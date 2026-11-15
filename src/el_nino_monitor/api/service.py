"""Acesso local aos NetCDF e cache compartilhado pelos endpoints."""

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import xarray as xr

from el_nino_monitor.config import Settings
from el_nino_monitor.data.noaa_client import NOAAClient, DatasetValidationError
from el_nino_monitor.processing.history import MonthPeriod
from el_nino_monitor.processing.sst import (
    calculate_monthly_climatology,
    prepare_published_noaa_climatology,
)


CLIMATOLOGY_FILENAME = "sst.mon.ltm.1991-2020.nc"


class LocalDataService:
    """Lê somente dados locais; observa alterações por mtime dos arquivos."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = NOAAClient(settings)
        self._lock = RLock()
        self._aggregate: xr.Dataset | None = None
        self._aggregate_stamp: int | None = None

    @staticmethod
    def _stamp(path: Path) -> int:
        return path.stat().st_mtime_ns

    @staticmethod
    def updated_at(paths: list[Path]) -> str:
        return datetime.fromtimestamp(
            max(path.stat().st_mtime for path in paths), tz=timezone.utc
        ).isoformat()

    @lru_cache(maxsize=48)
    def _month(self, path: Path, stamp: int) -> xr.Dataset:
        with self.client.open_dataset(path) as dataset:
            result = dataset.load()
            expected = path.stem.removeprefix("ersst.v5.")
            if result.sizes["time"] != 1:
                raise DatasetValidationError(f"Arquivo mensal com dimensão time inválida: {path.name}")
            actual = f"{int(result.time.dt.year.item()):04d}{int(result.time.dt.month.item()):02d}"
            if actual != expected:
                raise DatasetValidationError(f"Arquivo mensal com data inesperada: {path.name}")
            return result

    @lru_cache(maxsize=2)
    def _published_climatology(self, path: Path, stamp: int) -> xr.DataArray:
        with xr.open_dataset(
            path, engine="netcdf4",
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        ) as dataset:
            return prepare_published_noaa_climatology(dataset)

    def _aggregate_data(self) -> xr.Dataset:
        path = self.client.dataset_path
        if not path.is_file():
            raise FileNotFoundError("Nenhum arquivo agregado ERSSTv5 local")
        stamp = self._stamp(path)
        with self._lock:
            if self._aggregate is None or self._aggregate_stamp != stamp:
                replacement = self.client.open_dataset(path)
                if self._aggregate is not None:
                    self._aggregate.close()
                self._aggregate = replacement
                self._aggregate_stamp = stamp
            return self._aggregate

    def close(self) -> None:
        with self._lock:
            if self._aggregate is not None:
                self._aggregate.close()
                self._aggregate = None

    def observations(
        self, period: MonthPeriod
    ) -> tuple[xr.Dataset, str, list[Path]]:
        paths = [
            self.settings.raw_dir / f"ersst.v5.{month.year:04d}{month.month:02d}.nc"
            for month in period.months
        ]
        if all(path.is_file() for path in paths):
            with self._lock:
                datasets = [self._month(path, self._stamp(path)) for path in paths]
            if len(datasets) == 1:
                data = datasets[0]
            else:
                data = xr.concat(
                    datasets, dim="time", join="exact", data_vars="minimal",
                    coords="minimal", compat="equals",
                )
                self.client.get_metadata(data)
            return data, "NOAA NCEI ERSSTv5", paths
        if self.client.dataset_path.is_file():
            return self._aggregate_data(), "NOAA PSL ERSSTv5", [self.client.dataset_path]
        missing = next(path.name for path in paths if not path.is_file())
        raise FileNotFoundError(f"Mês {missing} ausente no cache local")

    def climatology(self) -> tuple[xr.DataArray, list[Path]]:
        published = self.settings.raw_dir / CLIMATOLOGY_FILENAME
        if published.is_file():
            with self._lock:
                return self._published_climatology(
                    published, self._stamp(published)
                ), [published]
        if not self.client.dataset_path.is_file():
            raise FileNotFoundError("Climatologia NOAA 1991–2020 ausente no cache local")
        return self._calculated_climatology(
            self.client.dataset_path, self._stamp(self.client.dataset_path)
        ), [self.client.dataset_path]

    @lru_cache(maxsize=1)
    def _calculated_climatology(self, path: Path, stamp: int) -> xr.DataArray:
        return calculate_monthly_climatology(self._aggregate_data()).load()

    def available_datasets(self) -> list[dict[str, object]]:
        """Descreve arquivos locais sem inferir continuidade entre meses."""
        records: list[dict[str, object]] = []
        aggregate_path = self.client.dataset_path
        if aggregate_path.is_file():
            records.append(self._record(
                self._aggregate_data(), "NOAA PSL ERSSTv5",
                [aggregate_path], "aggregated",
            ))
        monthly = sorted(self.settings.raw_dir.glob("ersst.v5.??????.nc"))
        if monthly:
            with self._lock:
                first = self._month(monthly[0], self._stamp(monthly[0]))
                last = self._month(monthly[-1], self._stamp(monthly[-1]))
            record = self._record(
                first, "NOAA NCEI ERSSTv5", monthly, "monthly_files"
            )
            record["time_end"] = self.client.get_metadata(last).time_end
            record["available_months"] = [
                f"{path.stem[-6:-2]}-{path.stem[-2:]}" for path in monthly
            ]
            records.append(record)
        return records

    def _record(
        self, data: xr.Dataset, source: str, paths: list[Path], kind: str
    ) -> dict[str, object]:
        metadata = self.client.get_metadata(data)
        lat = np.asarray(data[metadata.latitude_name].values, dtype=float)
        lon = np.asarray(data[metadata.longitude_name].values, dtype=float)
        return {
            "id": kind,
            "source": source,
            "time_start": metadata.time_start,
            "time_end": metadata.time_end,
            "spatial_resolution_degrees": {
                "latitude": float(np.median(np.abs(np.diff(lat)))) if len(lat) > 1 else None,
                "longitude": float(np.median(np.abs(np.diff(lon)))) if len(lon) > 1 else None,
            },
            "dimensions": metadata.dimensions,
            "local_updated_at_utc": self.updated_at(paths),
            "available_months": None,
        }

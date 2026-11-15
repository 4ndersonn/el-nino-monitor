"""Download e inspeção do arquivo mensal NOAA ERSSTv5."""

from dataclasses import dataclass
from datetime import date
import logging
import os
from pathlib import Path
from collections.abc import Iterable
import tempfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr

from el_nino_monitor.config import Settings, get_settings


ERSST_V5_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
ERSST_V5_FILENAME = "sst.mnmean.nc"
NCEI_MONTHLY_BASE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf"
LOGGER = logging.getLogger("el_nino_monitor.data.noaa_client")
CELSIUS_UNITS = {"degc", "degreec", "degreesc", "degreecelsius", "degreescelsius", "celsius", "°c"}


class NOAAClientError(RuntimeError):
    """Falha ao obter dados da fonte configurada."""


class DatasetValidationError(ValueError):
    """Arquivo NetCDF ou conteúdo científico inválido."""


class DateUnavailableError(LookupError):
    """Mês solicitado não está presente no arquivo."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Resumo verificado de um arquivo ERSSTv5."""

    dimensions: dict[str, int]
    latitude_name: str
    longitude_name: str
    sst_variable: str
    temperature_units: str
    time_start: str
    time_end: str
    missing_value_count: int


class NOAAClient:
    """Obtém e valida ERSSTv5 sem trocar de fonte implicitamente."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        url: str = ERSST_V5_URL,
        timeout: tuple[float, float] = (10.0, 120.0),
        chunk_size: int = 1024 * 1024,
        session: requests.Session | None = None,
    ) -> None:
        if chunk_size <= 0 or any(value <= 0 for value in timeout):
            raise ValueError("chunk_size e timeout devem ser positivos")
        self.settings = settings or get_settings()
        self.url = url
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.session = session or requests.Session()
        if session is None:
            retry_policy = Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            self.session.mount(
                "https://", HTTPAdapter(max_retries=retry_policy)
            )

    @property
    def dataset_path(self) -> Path:
        """Caminho do arquivo bruto no cache local."""
        return self.settings.raw_dir / ERSST_V5_FILENAME

    def download_ersst(self, *, force: bool = False) -> Path:
        """Baixa em streaming, valida e publica o arquivo de modo atômico."""
        return self._download_file(self.url, self.dataset_path, force=force)

    def download_ncei_month(
        self, year: int, month: int, *, force: bool = False
    ) -> Path:
        """Baixa um mês ERSSTv5 do arquivo oficial NCEI, com cache validado."""
        date(year, month, 1)
        filename = f"ersst.v5.{year:04d}{month:02d}.nc"
        destination = self.settings.raw_dir / filename
        url = f"{NCEI_MONTHLY_BASE_URL}/{filename}"
        return self._download_file(
            url, destination, force=force, expected_month=f"{year:04d}-{month:02d}"
        )

    def _validate_file(self, path: Path, expected_month: str | None) -> None:
        with self.open_dataset(path) as dataset:
            if expected_month is not None:
                metadata = self.get_metadata(dataset)
                if (
                    dataset.sizes["time"] != 1
                    or metadata.time_start[:7] != expected_month
                ):
                    raise DatasetValidationError(
                        f"Arquivo mensal não corresponde a {expected_month}"
                    )

    def _download_file(
        self,
        url: str,
        destination: Path,
        *,
        force: bool,
        expected_month: str | None = None,
    ) -> Path:
        if destination.exists() and not force:
            try:
                self._validate_file(destination, expected_month)
            except DatasetValidationError:
                LOGGER.warning("Cache inválido em %s; baixando novamente", destination)
            else:
                LOGGER.info("Usando arquivo ERSSTv5 em cache: %s", destination)
                return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                try:
                    with self.session.get(
                        url, stream=True, timeout=self.timeout
                    ) as response:
                        response.raise_for_status()
                        received = 0
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                temporary_file.write(chunk)
                                received += len(chunk)
                        advertised = response.headers.get("Content-Length")
                        if advertised is not None and received != int(advertised):
                            raise DatasetValidationError(
                                f"Download incompleto: {received} de {advertised} bytes"
                            )
                except requests.RequestException as exc:
                    raise NOAAClientError(f"Falha de conexão com {url}: {exc}") from exc

            self._validate_file(temporary_path, expected_month)
            os.replace(temporary_path, destination)
            LOGGER.info("ERSSTv5 validado e salvo em %s", destination)
            return destination
        except OSError as exc:
            raise NOAAClientError(f"Falha ao salvar ERSSTv5 em {destination}: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def open_dataset(self, path: str | Path) -> xr.Dataset:
        """Abre e valida um NetCDF; o chamador deve fechar o dataset."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Arquivo NetCDF ausente: {file_path}")
        try:
            dataset = xr.open_dataset(file_path, engine="netcdf4")
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"NetCDF corrompido ou ilegível: {file_path}") from exc
        try:
            self.get_metadata(dataset)
        except Exception:
            dataset.close()
            raise
        return dataset

    def load_monthly_files(self, paths: Iterable[str | Path]) -> xr.Dataset:
        """Carrega arquivos mensais locais em ordem, mantendo a grade exata.

        Os arquivos são validados individualmente e fechados após a leitura.
        A série retornada fica em memória e deve conter um calendário
        compatível entre seus meses.
        """
        ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
        if not ordered:
            raise FileNotFoundError("Nenhum arquivo mensal NetCDF informado")
        loaded: list[xr.Dataset] = []
        for path in ordered:
            with self.open_dataset(path) as dataset:
                if dataset.sizes["time"] != 1:
                    raise DatasetValidationError(
                        f"Esperado um único mês em {path}"
                    )
                loaded.append(dataset.load())
        try:
            combined = xr.concat(
                loaded, dim="time", join="exact",
                data_vars="minimal", coords="minimal", compat="equals",
            )
            self.get_metadata(combined)
        except (ValueError, TypeError) as exc:
            raise DatasetValidationError(
                "Arquivos mensais com grades, calendários ou datas incompatíveis"
            ) from exc
        return combined

    def get_metadata(self, dataset: xr.Dataset) -> DatasetMetadata:
        """Inspeciona dimensões, coordenadas, SST, unidades, datas e ausências."""
        latitude = self._coordinate(dataset, ("lat", "latitude"), -90.0, 90.0)
        longitude = self._coordinate(dataset, ("lon", "longitude"), -180.0, 360.0)
        if (
            "time" not in dataset.coords
            or dataset["time"].ndim != 1
            or dataset["time"].dims != ("time",)
        ):
            raise DatasetValidationError("Coordenada temporal 'time' ausente ou inesperada")
        try:
            years = np.asarray(dataset.time.dt.year.values, dtype=int)
            months = np.asarray(dataset.time.dt.month.values, dtype=int)
            days = np.asarray(dataset.time.dt.day.values, dtype=int)
            times = list(zip(years, months, days))
        except (AttributeError, TypeError, ValueError) as exc:
            raise DatasetValidationError("Datas inválidas em 'time'") from exc
        if not times or any(
            not (1 <= month <= 12 and 1 <= day <= 31)
            for _, month, day in times
        ) or times != sorted(set(times)):
            raise DatasetValidationError("Datas ausentes, duplicadas ou fora de ordem")

        variable_name = self._sst_variable(dataset)
        sst = dataset[variable_name]
        required_dims = {"time", latitude, longitude}
        extra_dims = set(sst.dims) - required_dims
        if (
            not required_dims.issubset(sst.dims)
            or any(dataset.sizes[dimension] != 1 for dimension in extra_dims)
        ):
            raise DatasetValidationError(f"Dimensões de {variable_name} inesperadas: {sst.dims}")
        units = str(sst.attrs.get("units", "")).strip()
        normalized_units = units.lower().replace("_", "").replace(" ", "")
        if normalized_units not in CELSIUS_UNITS:
            raise DatasetValidationError(f"Unidade de SST não reconhecida: {units!r}")
        if not np.issubdtype(sst.dtype, np.number):
            raise DatasetValidationError("SST deve conter valores numéricos")
        values = np.asarray(sst.values)
        if np.isinf(values).any():
            raise DatasetValidationError("SST contém valores infinitos")
        missing_count = int(np.isnan(values).sum())
        if missing_count == sst.size:
            raise DatasetValidationError("Todos os valores de SST estão ausentes")

        return DatasetMetadata(
            dimensions=dict(dataset.sizes),
            latitude_name=latitude,
            longitude_name=longitude,
            sst_variable=variable_name,
            temperature_units=units,
            time_start=f"{times[0][0]:04d}-{times[0][1]:02d}-{times[0][2]:02d}",
            time_end=f"{times[-1][0]:04d}-{times[-1][1]:02d}-{times[-1][2]:02d}",
            missing_value_count=missing_count,
        )

    def get_month(self, dataset: xr.Dataset, year: int, month: int) -> xr.Dataset:
        """Seleciona um mês disponível; informa quando ele não existe."""
        date(year, month, 1)
        matches = np.flatnonzero(
            (dataset.time.dt.year.values == year)
            & (dataset.time.dt.month.values == month)
        )
        if len(matches) == 0:
            raise DateUnavailableError(f"ERSSTv5 não contém {year:04d}-{month:02d}")
        return dataset.isel(time=int(matches[0]))

    @staticmethod
    def _coordinate(
        dataset: xr.Dataset, names: tuple[str, ...], minimum: float, maximum: float
    ) -> str:
        name = next((candidate for candidate in names if candidate in dataset.coords), None)
        if name is None:
            raise DatasetValidationError(f"Coordenada ausente: {names[0]}")
        coordinate = dataset[name]
        if coordinate.ndim != 1 or coordinate.size == 0 or coordinate.dims != (name,):
            raise DatasetValidationError(f"Coordenada inesperada: {name}")
        try:
            values = np.asarray(coordinate.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(f"Coordenada não numérica: {name}") from exc
        if (
            not np.isfinite(values).all()
            or np.any(values < minimum)
            or np.any(values > maximum)
            or (values.size > 1 and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)))
        ):
            raise DatasetValidationError(f"Valores inesperados na coordenada {name}")
        return name

    @staticmethod
    def _sst_variable(dataset: xr.Dataset) -> str:
        if "sst" in dataset.data_vars:
            return "sst"
        matches = [
            name
            for name, data in dataset.data_vars.items()
            if data.attrs.get("standard_name") == "sea_surface_temperature"
        ]
        if len(matches) != 1:
            raise DatasetValidationError("Variável de SST ausente ou ambígua")
        return matches[0]

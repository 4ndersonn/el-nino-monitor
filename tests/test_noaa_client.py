"""Dados NetCDF sintéticos usados somente para testar o cliente NOAA."""

from pathlib import Path

import cftime
import numpy as np
import pytest
import requests
import xarray as xr

from el_nino_monitor.config import Settings
from el_nino_monitor.data.noaa_client import (
    DateUnavailableError,
    DatasetValidationError,
    NOAAClient,
    NOAAClientError,
)


def synthetic_dataset(*, name: str = "sst", units: str = "degC") -> xr.Dataset:
    """Cria uma grade minúscula; estes valores não são observações reais."""
    values = np.array(
        [[[26.0, np.nan], [27.0, 28.0]], [[26.5, 27.0], [27.5, 28.5]]],
        dtype=np.float32,
    )
    return xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                values,
                {"units": units, "standard_name": "sea_surface_temperature"},
            )
        },
        coords={
            "time": np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"),
            "lat": [1.0, -1.0],
            "lon": [190.0, 192.0],
        },
    )


def write_synthetic(path: Path, dataset: xr.Dataset | None = None) -> Path:
    (dataset if dataset is not None else synthetic_dataset()).to_netcdf(
        path, engine="netcdf4"
    )
    return path


def all_missing_dataset() -> xr.Dataset:
    dataset = synthetic_dataset()
    dataset["sst"] = dataset["sst"] * np.nan
    dataset["sst"].attrs["units"] = "degC"
    return dataset


class FakeResponse:
    def __init__(self, content: bytes, *, failure: Exception | None = None) -> None:
        self.content = content
        self.failure = failure
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        if self.failure is not None:
            raise self.failure
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    def __init__(self, content: bytes, *, failure: Exception | None = None) -> None:
        self.content = content
        self.failure = failure
        self.calls = 0

    def get(self, url: str, *, stream: bool, timeout: tuple[float, float]) -> FakeResponse:
        assert url.startswith("https://")
        assert stream
        assert timeout == (10.0, 120.0)
        self.calls += 1
        return FakeResponse(self.content, failure=self.failure)


@pytest.fixture
def client(tmp_path: Path) -> NOAAClient:
    return NOAAClient(
        Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    )


def test_reads_synthetic_netcdf_and_metadata(client: NOAAClient, tmp_path: Path) -> None:
    path = write_synthetic(tmp_path / "synthetic.nc")
    with client.open_dataset(path) as dataset:
        metadata = client.get_metadata(dataset)
        assert metadata.dimensions == {"time": 2, "lat": 2, "lon": 2}
        assert metadata.latitude_name == "lat"
        assert metadata.longitude_name == "lon"
        assert metadata.sst_variable == "sst"
        assert metadata.temperature_units == "degC"
        assert metadata.time_start == "2020-01-01"
        assert metadata.time_end == "2020-02-01"
        assert metadata.missing_value_count == 1
        assert client.get_month(dataset, 2020, 2)["sst"].size == 4
        with pytest.raises(DateUnavailableError):
            client.get_month(dataset, 2020, 3)


def test_identifies_sst_by_standard_name(client: NOAAClient, tmp_path: Path) -> None:
    path = write_synthetic(tmp_path / "synthetic.nc", synthetic_dataset(name="tos"))
    with client.open_dataset(path) as dataset:
        assert client.get_metadata(dataset).sst_variable == "tos"


def test_reads_360_day_calendar(client: NOAAClient, tmp_path: Path) -> None:
    dataset = synthetic_dataset().assign_coords(
        time=[
            cftime.Datetime360Day(1997, 12, 1),
            cftime.Datetime360Day(1998, 1, 1),
        ]
    )
    path = write_synthetic(tmp_path / "synthetic-360-day.nc", dataset)
    with client.open_dataset(path) as opened:
        metadata = client.get_metadata(opened)
        assert metadata.time_start == "1997-12-01"
        assert metadata.time_end == "1998-01-01"
        assert client.get_month(opened, 1997, 12)["sst"].size == 4


def test_accepts_singleton_depth_axis(client: NOAAClient, tmp_path: Path) -> None:
    dataset = synthetic_dataset().expand_dims(lev=[0.0])
    path = write_synthetic(tmp_path / "synthetic-with-depth.nc", dataset)
    with client.open_dataset(path) as opened:
        assert client.get_metadata(opened).dimensions["lev"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda ds: ds.assign_coords(lat=[91.0, -1.0]),
        lambda ds: ds.assign_coords(lon=[190.0, 190.0]),
        lambda ds: ds.drop_vars("lon"),
        lambda ds: ds.assign_coords(time=np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]")),
        lambda ds: ds.assign(sst=ds.sst.assign_attrs(units="kelvin")),
        lambda ds: all_missing_dataset(),
    ],
)
def test_rejects_invalid_scientific_content(
    client: NOAAClient, tmp_path: Path, change
) -> None:
    path = write_synthetic(tmp_path / "invalid.nc", change(synthetic_dataset()))
    with pytest.raises(DatasetValidationError):
        client.open_dataset(path)


def test_rejects_missing_and_corrupt_files(client: NOAAClient, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        client.open_dataset(tmp_path / "missing.nc")
    corrupt = tmp_path / "corrupt.nc"
    corrupt.write_bytes(b"not a NetCDF file")
    with pytest.raises(DatasetValidationError):
        client.open_dataset(corrupt)


def test_cache_and_forced_refresh(tmp_path: Path) -> None:
    source = write_synthetic(tmp_path / "source.nc")
    session = FakeSession(source.read_bytes())
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    client = NOAAClient(settings, session=session, chunk_size=32)

    destination = client.download_ersst()
    assert destination.is_file()
    assert session.calls == 1
    assert client.download_ersst() == destination
    assert session.calls == 1
    assert client.download_ersst(force=True) == destination
    assert session.calls == 2
    assert list(settings.raw_dir.glob("*.part")) == []


def test_invalid_download_does_not_replace_valid_cache(tmp_path: Path) -> None:
    source = write_synthetic(tmp_path / "source.nc")
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    client = NOAAClient(settings, session=FakeSession(source.read_bytes()))
    destination = client.download_ersst()
    original = destination.read_bytes()
    client.session = FakeSession(b"invalid")

    with pytest.raises(DatasetValidationError):
        client.download_ersst(force=True)

    assert destination.read_bytes() == original
    assert not list(settings.raw_dir.glob("*.part"))


def test_connection_failure_cleans_temporary_file(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    session = FakeSession(b"", failure=requests.ConnectionError("offline"))
    client = NOAAClient(settings, session=session)
    with pytest.raises(NOAAClientError, match="Falha de conexão"):
        client.download_ersst()
    assert not client.dataset_path.exists()
    assert not list(settings.raw_dir.glob("*.part"))


def test_download_ncei_month_uses_validated_cache(tmp_path: Path) -> None:
    one_month = synthetic_dataset().isel(time=[0])
    source = write_synthetic(tmp_path / "source.nc", one_month)
    session = FakeSession(source.read_bytes())
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    client = NOAAClient(settings, session=session)
    path = client.download_ncei_month(2020, 1)
    assert path.name == "ersst.v5.202001.nc"
    assert client.download_ncei_month(2020, 1) == path
    assert session.calls == 1


def test_download_ncei_month_rejects_wrong_month(tmp_path: Path) -> None:
    source = write_synthetic(tmp_path / "source.nc", synthetic_dataset().isel(time=[1]))
    client = NOAAClient(
        Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs"),
        session=FakeSession(source.read_bytes()),
    )
    with pytest.raises(DatasetValidationError, match="não corresponde"):
        client.download_ncei_month(2020, 1)
    assert not client.settings.raw_dir.joinpath("ersst.v5.202001.nc").exists()


def test_load_monthly_files_combines_realistic_single_month_shapes(
    client: NOAAClient, tmp_path: Path
) -> None:
    source = synthetic_dataset()
    january = write_synthetic(tmp_path / "ersst.v5.202001.nc", source.isel(time=[0]))
    february = write_synthetic(tmp_path / "ersst.v5.202002.nc", source.isel(time=[1]))
    combined = client.load_monthly_files([february, january])
    try:
        assert combined.sizes["time"] == 2
        assert client.get_metadata(combined).time_start == "2020-01-01"
        assert client.get_metadata(combined).time_end == "2020-02-01"
    finally:
        combined.close()

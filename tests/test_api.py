"""Integração HTTP com NetCDF sintético, exclusivamente para testes."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, box

from el_nino_monitor.api.main import create_app
from el_nino_monitor.config import Settings
from el_nino_monitor.visualization.maps import MapFeatures


def _synthetic_files(raw_dir: Path) -> None:
    """Valores artificiais não são usados fora desta suíte."""
    raw_dir.mkdir(parents=True)
    latitude = [-4.0, 4.0]
    longitude = [190.0, 220.0, 240.0]
    for date, anomaly in (("1997-11-15", 1.0), ("1997-12-15", 2.0), ("1998-01-15", 3.0),
                          ("2015-11-15", 0.0), ("2015-12-15", 1.0), ("2016-01-15", 2.0)):
        values = np.full((1, 2, 3), 25.0 + anomaly)
        dataset = xr.Dataset(
            {"sst": (("time", "lat", "lon"), values, {"units": "degree_C"})},
            coords={"time": [pd.Timestamp(date)], "lat": latitude, "lon": longitude},
        )
        dataset.to_netcdf(raw_dir / f"ersst.v5.{date[:4]}{date[5:7]}.nc")
    baseline = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.full((12, 2, 3), 25.0),
                 {"units": "degree_C"})},
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS"),
                "lat": latitude, "lon": longitude},
    )
    baseline.time.attrs["climo_period"] = "1991/01/01 - 2020/12/31"
    baseline.to_netcdf(raw_dir / "sst.mon.ltm.1991-2020.nc")


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data",
                        output_dir=tmp_path / "outputs")
    _synthetic_files(settings.raw_dir)
    features = MapFeatures(
        land=cfeature.ShapelyFeature([box(-80, -20, -60, 20)], ccrs.PlateCarree(),
                                     facecolor="#ddd"),
        coastlines=cfeature.ShapelyFeature(
            [LineString([(-80, -20), (-60, -20)])], ccrs.PlateCarree(),
            facecolor="none"),
    )
    app = create_app(settings, map_features=features)
    with TestClient(app) as http:
        yield http


def test_health_datasets_and_openapi(client: TestClient) -> None:
    assert client.get("/api/v1/health").json()["status"] == "ok"
    datasets = client.get("/api/v1/datasets").json()["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["spatial_resolution_degrees"]["longitude"] == 25.0
    assert datasets[0]["available_months"][0] == "1997-11"
    assert datasets[0]["time_end"] == "2016-01-15"
    assert datasets[0]["local_updated_at_utc"]
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_nino34_and_history_with_cache(client: TestClient) -> None:
    one = client.get("/api/v1/enso/nino34", params={"year": 1997, "month": 12})
    assert one.status_code == 200, one.text
    body = one.json()
    assert body["observations"][0]["monthly_anomaly"]["value"] == pytest.approx(2)
    assert body["observations"][0]["observation_date"] == "1997-12-15"
    assert body["observations"][0]["three_month_mean"]["value"] is None
    assert body["reference_period"] == "1991–2020"
    history = client.get("/api/v1/enso/history", params={"start": "1997-11", "end": "1998-01"})
    assert history.status_code == 200, history.text
    values = history.json()["observations"]
    assert [row["monthly_anomaly"]["value"] for row in values] == pytest.approx([1, 2, 3])
    assert values[1]["three_month_mean"]["value"] == pytest.approx(2)
    assert values[1]["month_to_month_change"]["value"] == pytest.approx(1)
    service = client.app.state.data_service
    assert service._published_climatology.cache_info().hits >= 1


def test_compare_and_map_png(client: TestClient) -> None:
    response = client.get("/api/v1/enso/compare", params={
        "start": "1997-11", "end": "1998-01",
        "other_start": "2015-11", "other_end": "2016-01",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["first"]["mean_anomaly"]["value"] == pytest.approx(2)
    assert body["second"]["mean_anomaly"]["value"] == pytest.approx(1)
    assert body["source"] == "NOAA NCEI ERSSTv5"
    image = client.get("/api/v1/enso/map", params={"year": 1997, "month": 12,
                                                      "region": "pacific", "visualization": "anomaly"})
    assert image.status_code == 200, image.text
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    assert image.headers["x-observation-month"] == "1997-12"
    paired = client.get("/api/v1/enso/compare/map", params={
        "start": "1997-11", "end": "1998-01",
        "other_start": "2015-11", "other_end": "2016-01",
    })
    assert paired.status_code == 200, paired.text
    assert paired.content.startswith(b"\x89PNG")
    assert paired.headers["x-reference-period"] == "1991-2020"


def test_invalid_dates_missing_data_and_missing_baseline(client: TestClient) -> None:
    assert client.get("/api/v1/enso/nino34", params={"year": 1997, "month": 13}).status_code == 422
    assert client.get("/api/v1/enso/history", params={"start": "1997-13", "end": "1998-01"}).status_code == 422
    assert client.get("/api/v1/enso/nino34", params={"year": 1997, "month": 10}).status_code == 404
    assert client.get("/api/v1/enso/history", params={"start": "1997-11", "end": "1998-02"}).status_code == 404
    assert client.get("/api/v1/enso/map", params={"year": 1997, "month": 12,
                                                      "visualization": "absolute"}).status_code == 422
    baseline = client.app.state.data_service.settings.raw_dir / "sst.mon.ltm.1991-2020.nc"
    baseline.unlink()
    assert client.get("/api/v1/enso/nino34", params={"year": 1997, "month": 12}).status_code == 503


def test_corrupt_month_is_reported(client: TestClient) -> None:
    path = client.app.state.data_service.settings.raw_dir / "ersst.v5.199712.nc"
    path.write_bytes(b"corrupt test fixture")
    response = client.get("/api/v1/enso/nino34", params={"year": 1997, "month": 12})
    assert response.status_code == 422

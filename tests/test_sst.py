"""Cálculos científicos com grades sintéticas, nunca observações reais."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import cftime

from el_nino_monitor.processing.sst import (
    ScientificDataError,
    calculate_monthly_climatology,
    calculate_nino34_mean,
    calculate_nino34_timeseries,
    calculate_sst_anomaly,
    calculate_three_month_running_mean,
    prepare_published_noaa_climatology,
    select_nino34_region,
)


def synthetic_sst() -> xr.DataArray:
    """Dois anos completos para uma climatologia curta e configurada no teste."""
    times = pd.date_range("1991-01-01", "1992-12-01", freq="MS")
    values = np.empty((24, 2, 2), dtype=float)
    for index, timestamp in enumerate(times):
        values[index, :, :] = timestamp.month + (-1 if timestamp.year == 1991 else 1)
    values[:, 1, 1] = np.nan  # Célula sempre ausente, como uma máscara de terra.
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-4.0, 4.0], "lon": [190.0, 192.0]},
        name="sst",
        attrs={"units": "degree_C"},
    )


def test_monthly_climatology_preserves_spatial_grid_and_missing_values() -> None:
    climatology = calculate_monthly_climatology(
        synthetic_sst(), start_year=1991, end_year=1992
    )
    assert climatology.dims == ("month", "lat", "lon")
    assert climatology.sizes["month"] == 12
    assert climatology.sel(month=1, lat=-4, lon=190).item() == pytest.approx(1.0)
    assert climatology.sel(month=12, lat=4, lon=190).item() == pytest.approx(12.0)
    assert np.isnan(climatology.sel(month=1, lat=4, lon=192).item())


def test_anomalies_positive_negative_and_near_zero() -> None:
    baseline = synthetic_sst()
    climatology = calculate_monthly_climatology(
        baseline, start_year=1991, end_year=1992
    )
    observed = baseline.isel(time=slice(0, 3)).copy()
    observed = observed.assign_coords(time=pd.date_range("1993-01-01", periods=3, freq="MS"))
    observed.loc[{"time": "1993-01-01", "lat": -4, "lon": 190}] = 3.0
    observed.loc[{"time": "1993-02-01", "lat": -4, "lon": 190}] = 0.0
    observed.loc[{"time": "1993-03-01", "lat": -4, "lon": 190}] = 3.0 + 1e-9
    anomaly = calculate_sst_anomaly(observed, climatology)
    assert anomaly.sel(time="1993-01-01", lat=-4, lon=190).item() == pytest.approx(2.0)
    assert anomaly.sel(time="1993-02-01", lat=-4, lon=190).item() == pytest.approx(-2.0)
    assert anomaly.sel(time="1993-03-01", lat=-4, lon=190).item() == pytest.approx(1e-9)
    assert np.isnan(anomaly.sel(time="1993-01-01", lat=4, lon=192).item())


def test_reference_period_must_have_every_month() -> None:
    missing_february = synthetic_sst().drop_sel(time="1991-02-01")
    with pytest.raises(ScientificDataError, match="1991-02"):
        calculate_monthly_climatology(
            missing_february, start_year=1991, end_year=1992
        )
    with pytest.raises(ScientificDataError, match="1993-01"):
        calculate_monthly_climatology(
            synthetic_sst(), start_year=1991, end_year=1993
        )


def test_reference_month_without_any_sst_is_rejected() -> None:
    data = synthetic_sst()
    data.loc[{"time": "1991-01-01"}] = np.nan
    with pytest.raises(ScientificDataError, match="sem qualquer SST"):
        calculate_monthly_climatology(data, start_year=1991, end_year=1992)


@pytest.mark.parametrize(
    ("longitudes", "expected"),
    [
        ([180.0, 190.0, 220.0, 240.0, 250.0], [190.0, 220.0, 240.0]),
        ([-180.0, -170.0, -140.0, -120.0, -110.0], [-170.0, -140.0, -120.0]),
    ],
)
def test_nino34_selection_accepts_both_longitude_conventions(
    longitudes: list[float], expected: list[float]
) -> None:
    data = xr.DataArray(
        np.ones((1, 4, 5)),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2020-01-01")],
            "lat": [6.0, 4.0, -4.0, -6.0],
            "lon": longitudes,
        },
        attrs={"units": "degC"},
    )
    selected = select_nino34_region(data)
    assert selected.lat.values.tolist() == [4.0, -4.0]
    assert selected.lon.values.tolist() == expected


def test_nino34_mean_uses_cosine_weights_and_skips_missing() -> None:
    data = xr.DataArray(
        np.array([[[10.0], [20.0]], [[np.nan], [20.0]]]),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="MS"),
            "lat": [0.0, 4.0],
            "lon": [200.0],
        },
        attrs={"units": "degC"},
    )
    mean = calculate_nino34_mean(data)
    expected = (10.0 + 20.0 * np.cos(np.deg2rad(4.0))) / (
        1.0 + np.cos(np.deg2rad(4.0))
    )
    assert mean.values[0] == pytest.approx(expected)
    assert mean.values[1] == pytest.approx(20.0)


def test_nino34_timeseries_reports_sst_and_anomaly() -> None:
    data = synthetic_sst()
    climatology = calculate_monthly_climatology(
        data, start_year=1991, end_year=1992
    )
    series = calculate_nino34_timeseries(data, climatology)
    assert set(series.data_vars) == {"sst_mean", "sst_anomaly"}
    assert series.sst_anomaly.isel(time=0).item() == pytest.approx(-1.0)
    assert series.sst_anomaly.isel(time=12).item() == pytest.approx(1.0)
    assert series.attrs["reference_start_year"] == 1991
    assert "não é ONI" in series.attrs["warning"]


def test_three_month_mean_is_centered_and_respects_gaps() -> None:
    series = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=5, freq="MS")},
        attrs={"units": "degree_C"},
    )
    running = calculate_three_month_running_mean(series)
    np.testing.assert_allclose(running.values, [np.nan, 2.0, 3.0, 4.0, np.nan], equal_nan=True)
    assert "não é ONI" in running.attrs["warning"]

    with_gap = series.isel(time=[0, 1, 3, 4])
    assert np.isnan(calculate_three_month_running_mean(with_gap).values).all()


def test_anomaly_rejects_mismatched_grid() -> None:
    baseline = synthetic_sst()
    climatology = calculate_monthly_climatology(
        baseline, start_year=1991, end_year=1992
    ).assign_coords(lon=[191.0, 193.0])
    with pytest.raises(ScientificDataError, match="Grade espacial incompatível"):
        calculate_sst_anomaly(baseline, climatology)


def test_climatology_rejects_non_celsius_units() -> None:
    data = synthetic_sst()
    data.attrs["units"] = "kelvin"
    with pytest.raises(ScientificDataError, match="graus Celsius"):
        calculate_monthly_climatology(data, start_year=1991, end_year=1992)


def test_processing_accepts_360_day_calendar() -> None:
    data = synthetic_sst().assign_coords(
        time=[
            cftime.Datetime360Day(year, month, 1)
            for year in (1991, 1992)
            for month in range(1, 13)
        ]
    )
    climatology = calculate_monthly_climatology(
        data, start_year=1991, end_year=1992
    )
    assert climatology.sizes["month"] == 12
    assert calculate_nino34_timeseries(data, climatology).sst_anomaly.size == 24


def test_anomaly_aligns_reversed_latitudes_without_interpolation() -> None:
    baseline = synthetic_sst()
    climatology = calculate_monthly_climatology(
        baseline, start_year=1991, end_year=1992
    ).sortby("lat", ascending=False)
    anomaly = calculate_sst_anomaly(baseline, climatology)
    assert anomaly.sel(time="1991-01-01", lat=-4, lon=190).item() == pytest.approx(-1.0)


def test_prepare_published_climatology_checks_declared_period() -> None:
    months = pd.date_range("2001-01-01", periods=12, freq="MS")
    dataset = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.ones((12, 2, 2)), {"units": "degC"})},
        coords={"time": months, "lat": [-1.0, 1.0], "lon": [190.0, 192.0]},
    )
    dataset.time.attrs["climo_period"] = "1991/01/01 - 2020/12/31"
    climatology = prepare_published_noaa_climatology(dataset)
    assert climatology.sizes["month"] == 12
    assert climatology.attrs["method"].startswith("climatologia mensal pré-calculada")
    with pytest.raises(ScientificDataError, match="Período declarado"):
        prepare_published_noaa_climatology(dataset, start_year=1981, end_year=2010)


def test_three_month_mean_keeps_missing_window_empty() -> None:
    series = xr.DataArray(
        [1.0, np.nan, 3.0, 4.0, 5.0],
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=5, freq="MS")},
    )
    result = calculate_three_month_running_mean(series)
    np.testing.assert_allclose(
        result.values, [np.nan, np.nan, np.nan, 4.0, np.nan], equal_nan=True
    )

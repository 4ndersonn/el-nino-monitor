"""Análise histórica com SST sintética exclusivamente para testes."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, box

from el_nino_monitor.processing.history import (
    MonthPeriod,
    build_nino34_history,
    calculate_period_statistics,
    compare_periods,
    export_history_csv,
    select_period,
)
from el_nino_monitor.processing.sst import ScientificDataError
from el_nino_monitor.visualization.comparison import (
    create_comparison_map_figure,
    save_comparison_map,
)
from el_nino_monitor.visualization.history import (
    create_history_figure,
    save_history_plot,
    save_period_comparison_plot,
)
from el_nino_monitor.visualization.maps import MapFeatures


def synthetic_period(start: str, anomalies: list[float]) -> xr.DataArray:
    """Grade pequena para testes; não contém observações científicas reais."""
    times = pd.date_range(start, periods=len(anomalies), freq="MS")
    values = np.array(anomalies, dtype=float)[:, None, None]
    values = np.broadcast_to(values + 25.0, (len(anomalies), 2, 3)).copy()
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-4.0, 4.0], "lon": [190.0, 220.0, 240.0]},
        attrs={"units": "degC"},
    )


def synthetic_climatology() -> xr.DataArray:
    return xr.DataArray(
        np.full((12, 2, 3), 25.0),
        dims=("month", "lat", "lon"),
        coords={"month": range(1, 13), "lat": [-4.0, 4.0], "lon": [190.0, 220.0, 240.0]},
        attrs={
            "units": "degC",
            "reference_start_year": 1991,
            "reference_end_year": 2020,
            "method": "climatologia sintética de teste",
        },
    )


def local_features() -> MapFeatures:
    land = cfeature.ShapelyFeature(
        [box(-80, -20, -60, 20)], ccrs.PlateCarree(),
        facecolor="#ecebe7", edgecolor="none",
    )
    coast = cfeature.ShapelyFeature(
        [LineString([(-80, -20), (-60, -20), (-60, 20)])],
        ccrs.PlateCarree(), facecolor="none", edgecolor="#40484d",
    )
    return MapFeatures(land=land, coastlines=coast)


def test_select_period_and_validate_missing_month() -> None:
    data = synthetic_period("1997-10", [1, 2, 3, 4, 5, 6])
    period = MonthPeriod("1997-11", "1998-02")
    selected = select_period(data, period)
    assert selected.sizes["time"] == 4
    with pytest.raises(ScientificDataError, match="1998-01"):
        select_period(data.drop_sel(time="1998-01-01"), period)
    with pytest.raises(ValueError):
        MonthPeriod("1998-02", "1997-11")


def test_history_series_statistics_and_monthly_change() -> None:
    period = MonthPeriod("1997-10", "1998-03")
    series = build_nino34_history(
        synthetic_period("1997-10", [1, 2, 3, 4, 5, 6]),
        synthetic_climatology(), period, source="SINTÉTICO DE TESTE",
    )
    np.testing.assert_allclose(series.sst_anomaly.values, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(
        series.sst_anomaly_3month_mean.values,
        [np.nan, 2, 3, 4, 5, np.nan], equal_nan=True,
    )
    np.testing.assert_allclose(
        series.month_to_month_change.values,
        [np.nan, 1, 1, 1, 1, 1], equal_nan=True,
    )
    stats = calculate_period_statistics(series)
    assert stats.area.startswith("Niño 3.4")
    assert stats.mean_anomaly_c == pytest.approx(3.5)
    assert stats.maximum_anomaly_c == pytest.approx(6)
    assert stats.maximum_month == "1998-03"
    assert stats.minimum_anomaly_c == pytest.approx(1)
    assert stats.minimum_month == "1997-10"
    assert stats.valid_months == 6


def test_missing_values_are_not_interpolated() -> None:
    data = synthetic_period("1997-10", [1, 2, 3, 4, 5, 6])
    data.loc[{"time": "1997-12-01"}] = np.nan
    series = build_nino34_history(
        data, synthetic_climatology(),
        MonthPeriod("1997-10", "1998-03"),
    )
    assert np.isnan(series.sst_anomaly.sel(time="1997-12-01").item())
    assert np.isnan(series.sst_anomaly_3month_mean.sel(time="1997-12-01").item())
    assert np.isnan(series.month_to_month_change.sel(time="1998-01-01").item())
    assert calculate_period_statistics(series).valid_months == 5


def test_period_with_only_decreases_has_no_monthly_increase() -> None:
    series = build_nino34_history(
        synthetic_period("2015-11", [3.0, 2.5, 2.0]),
        synthetic_climatology(), MonthPeriod("2015-11", "2016-01"),
    )
    stats = calculate_period_statistics(series)
    assert stats.largest_monthly_increase_c is None
    assert stats.largest_monthly_decrease_c == pytest.approx(-0.5)


def test_comparison_requires_same_duration_and_reference() -> None:
    baseline = synthetic_climatology()
    first = build_nino34_history(
        synthetic_period("1997-10", [1, 2, 3]),
        baseline, MonthPeriod("1997-10", "1997-12"),
    )
    second = build_nino34_history(
        synthetic_period("2015-10", [0, 1, 2]),
        baseline, MonthPeriod("2015-10", "2015-12"),
    )
    comparison = compare_periods(first, second)
    assert comparison.first.mean_anomaly_c == pytest.approx(2)
    assert comparison.second.mean_anomaly_c == pytest.approx(1)
    with pytest.raises(ScientificDataError, match="durações diferentes"):
        compare_periods(first, second.isel(time=slice(0, 2)).assign_attrs(period_end="2015-11"))
    with pytest.raises(ScientificDataError, match="Referências"):
        compare_periods(first, second.assign_attrs(reference_end_year=2010))


def test_exports_csv_and_temporal_plots(tmp_path: Path) -> None:
    first = build_nino34_history(
        synthetic_period("1997-10", [1, 2, 3]),
        synthetic_climatology(), MonthPeriod("1997-10", "1997-12"),
        source="SINTÉTICO DE TESTE",
    )
    second = build_nino34_history(
        synthetic_period("2015-10", [0, 1, 2]),
        synthetic_climatology(), MonthPeriod("2015-10", "2015-12"),
        source="SINTÉTICO DE TESTE",
    )
    csv_path = export_history_csv(first, tmp_path)
    table = pd.read_csv(csv_path)
    assert len(table) == 3
    assert table["area"].str.contains("Niño 3.4").all()
    assert table["monthly_anomaly_c"].tolist() == pytest.approx([1, 2, 3])
    assert pd.isna(table["three_month_mean_c"].iloc[0])
    paths = (
        *save_history_plot(first, output_dir=tmp_path, dpi=80),
        *save_period_comparison_plot(first, second, output_dir=tmp_path, dpi=80),
    )
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    assert paths[1].read_bytes().startswith(b"%PDF")
    fig = create_history_figure(first)
    try:
        assert "1991–2020" in fig.axes[0].get_title()
    finally:
        fig.clear()


def test_comparison_maps_share_scale_without_network(tmp_path: Path) -> None:
    first_period = MonthPeriod("1997-10", "1997-12")
    second_period = MonthPeriod("2015-10", "2015-12")
    first = synthetic_period("1997-10", [1, 2, 3])
    second = synthetic_period("2015-10", [-1, 0, 1])
    features = local_features()
    fig = create_comparison_map_figure(
        first, second, synthetic_climatology(),
        first_period, second_period,
        map_type="pacific", source="SINTÉTICO DE TESTE", features=features,
    )
    try:
        assert fig.axes[0].collections[0].norm is fig.axes[1].collections[0].norm
        assert "1997-10" in fig.axes[0].get_title()
        assert "2015-10" in fig.axes[1].get_title()
    finally:
        fig.clear()
    paths = save_comparison_map(
        first, second, synthetic_climatology(),
        first_period, second_period,
        output_dir=tmp_path, dpi=80,
        source="SINTÉTICO DE TESTE", features=features,
    )
    assert len(paths) == 2
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    assert paths[1].read_bytes().startswith(b"%PDF")

"""Mapas com SST sintética e feições locais; nenhum acesso à rede."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, box

from el_nino_monitor.data.noaa_client import DateUnavailableError
from el_nino_monitor.processing.sst import ScientificDataError
from el_nino_monitor.visualization.maps import (
    MapFeatures,
    create_sst_anomaly_figure,
    save_sst_anomaly_map,
)


@pytest.fixture
def synthetic_inputs() -> tuple[xr.DataArray, xr.DataArray]:
    """Pequena grade de teste; seus valores não são observações."""
    latitudes = [-40.0, -4.0, 4.0, 40.0]
    longitudes = [0.0, 120.0, 190.0, 220.0, 240.0, 290.0, 358.0]
    values = np.full((1, 4, 7), 27.0)
    values[0, 0, 0] = np.nan
    observed = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [np.datetime64("1997-12-01")], "lat": latitudes, "lon": longitudes},
        attrs={"units": "degC"},
    )
    climatology = xr.DataArray(
        np.full((12, 4, 7), 25.0),
        dims=("month", "lat", "lon"),
        coords={"month": np.arange(1, 13), "lat": latitudes, "lon": longitudes},
        attrs={"units": "degC", "reference_start_year": 1991, "reference_end_year": 2020},
    )
    return observed, climatology


@pytest.fixture
def local_features() -> MapFeatures:
    """Geometrias locais simples evitam downloads do Natural Earth nos testes."""
    land = cfeature.ShapelyFeature(
        [box(-80, -20, -60, 20)],
        ccrs.PlateCarree(),
        facecolor="#ecebe7",
        edgecolor="none",
    )
    coastlines = cfeature.ShapelyFeature(
        [LineString([(-80, -20), (-60, -20), (-60, 20), (-80, 20), (-80, -20)])],
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#40484d",
    )
    return MapFeatures(land=land, coastlines=coastlines)


@pytest.mark.parametrize(
    ("map_type", "projection"),
    [("pacific", ccrs.PlateCarree), ("global", ccrs.Robinson)],
)
def test_projection_and_scientific_labels(
    synthetic_inputs: tuple[xr.DataArray, xr.DataArray],
    local_features: MapFeatures,
    map_type: str,
    projection: type,
) -> None:
    observed, climatology = synthetic_inputs
    figure = create_sst_anomaly_figure(
        observed, climatology, year=1997, month=12,
        map_type=map_type, source="DADOS SINTÉTICOS DE TESTE",
        features=local_features,
    )
    try:
        assert isinstance(figure.axes[0].projection, projection)
        labels = " ".join(text.get_text() for text in figure.texts)
        assert "1997-12" in labels
        assert "1991–2020" in labels
        assert "Niño 3.4" in labels
        assert "DADOS SINTÉTICOS DE TESTE" in labels
        assert len(figure.axes[0].lines) == 2  # Duplo contorno Niño 3.4.
        if map_type == "global":
            assert "não representa efeitos globais" in labels
    finally:
        figure.clear()


def test_exports_png_and_pdf_without_network(
    synthetic_inputs: tuple[xr.DataArray, xr.DataArray],
    local_features: MapFeatures,
    tmp_path: Path,
) -> None:
    observed, climatology = synthetic_inputs
    paths = save_sst_anomaly_map(
        observed, climatology, year=1997, month=12,
        map_type="pacific", dpi=90, output_dir=tmp_path,
        source="DADOS SINTÉTICOS DE TESTE", features=local_features,
    )
    assert len(paths) == 2
    assert paths[0].name == "sst_anomaly_pacific_1997-12_ref_1991-2020.png"
    assert paths[1].name == "sst_anomaly_pacific_1997-12_ref_1991-2020.pdf"
    assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths[1].read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob("*.part"))


def test_rejects_invalid_or_unavailable_dates(
    synthetic_inputs: tuple[xr.DataArray, xr.DataArray],
    local_features: MapFeatures,
) -> None:
    observed, climatology = synthetic_inputs
    with pytest.raises(ValueError):
        create_sst_anomaly_figure(
            observed, climatology, year=1997, month=13, features=local_features
        )
    with pytest.raises(DateUnavailableError):
        create_sst_anomaly_figure(
            observed, climatology, year=1998, month=1, features=local_features
        )


def test_rejects_invalid_options_and_missing_reference(
    synthetic_inputs: tuple[xr.DataArray, xr.DataArray],
    local_features: MapFeatures,
    tmp_path: Path,
) -> None:
    observed, climatology = synthetic_inputs
    with pytest.raises(ValueError, match="Tipo de mapa"):
        create_sst_anomaly_figure(
            observed, climatology, year=1997, month=12,
            map_type="unknown", features=local_features,
        )
    with pytest.raises(ValueError, match="resolução"):
        save_sst_anomaly_map(
            observed, climatology, year=1997, month=12,
            dpi=0, output_dir=tmp_path, features=local_features,
        )
    with pytest.raises(ScientificDataError, match="período de referência"):
        no_reference = climatology.copy()
        no_reference.attrs = {"units": "degC"}
        create_sst_anomaly_figure(
            observed, no_reference,
            year=1997, month=12, features=local_features,
        )

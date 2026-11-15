"""Mapas geográficos de anomalias SST calculadas a partir de dados NOAA."""

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from el_nino_monitor.config import get_settings
from el_nino_monitor.data.noaa_client import DateUnavailableError
from el_nino_monitor.processing.sst import (
    ScientificDataError,
    calculate_nino34_mean,
    calculate_sst_anomaly,
)


MapType = Literal["pacific", "global"]
OutputFormat = Literal["png", "pdf"]


@dataclass(frozen=True)
class MapFeatures:
    """Feições cartográficas; permitem testes sem downloads geográficos."""

    land: cfeature.Feature
    coastlines: cfeature.Feature

    @classmethod
    def natural_earth(cls) -> "MapFeatures":
        """Usa feições Natural Earth de 110 m armazenadas em data/raw/cartopy."""
        data_dir = get_settings().raw_dir / "cartopy"
        data_dir.mkdir(parents=True, exist_ok=True)
        cartopy.config["data_dir"] = data_dir
        return cls(
            land=cfeature.NaturalEarthFeature(
                "physical", "land", "110m", facecolor="#ecebe7", edgecolor="none"
            ),
            coastlines=cfeature.NaturalEarthFeature(
                "physical", "coastline", "110m", facecolor="none", edgecolor="#40484d"
            ),
        )


def _selected_anomaly(
    data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    year: int,
    month: int,
) -> tuple[xr.DataArray, float]:
    """Retorna campo mensal e média Niño 3.4, sem inventar meses ausentes."""
    date(year, month, 1)
    if "time" not in data.coords:
        raise ScientificDataError("Dados sem coordenada temporal")
    positions = np.flatnonzero(
        (data.time.dt.year.values == year) & (data.time.dt.month.values == month)
    )
    if len(positions) == 0:
        raise DateUnavailableError(f"Não há SST para {year:04d}-{month:02d}")
    if len(positions) != 1:
        raise ScientificDataError(f"Há mais de uma observação em {year:04d}-{month:02d}")
    selected = data.isel(time=positions)
    anomaly = calculate_sst_anomaly(selected, climatology)
    mean = float(calculate_nino34_mean(anomaly).item())
    if not np.isfinite(mean):
        raise ScientificDataError("Niño 3.4 sem anomalias válidas neste mês")
    return anomaly.isel(time=0), mean


def create_sst_anomaly_figure(
    data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    *,
    year: int,
    month: int,
    map_type: MapType = "pacific",
    source: str = "NOAA ERSSTv5",
    features: MapFeatures | None = None,
) -> Figure:
    """Cria mapa Pacífico ou global com escala divergente simétrica."""
    if map_type not in ("pacific", "global"):
        raise ValueError(f"Tipo de mapa inválido: {map_type}")
    if not source.strip():
        raise ValueError("A fonte dos dados deve ser informada")
    reference_start = climatology.attrs.get("reference_start_year")
    reference_end = climatology.attrs.get("reference_end_year")
    if not isinstance(reference_start, int) or not isinstance(reference_end, int):
        raise ScientificDataError("Climatologia sem período de referência declarado")
    anomaly, nino34_mean = _selected_anomaly(data, climatology, year, month)
    latitude = "lat" if "lat" in anomaly.coords else "latitude"
    longitude = "lon" if "lon" in anomaly.coords else "longitude"
    field = anomaly.transpose(latitude, longitude)
    latitudes = np.asarray(field[latitude].values, dtype=float)
    longitudes = np.asarray(field[longitude].values, dtype=float)
    values = np.asarray(field.values, dtype=float)

    if map_type == "pacific":
        visible = (
            (latitudes[:, None] >= -35)
            & (latitudes[:, None] <= 35)
            & (np.mod(longitudes[None, :], 360) >= 120)
            & (np.mod(longitudes[None, :], 360) <= 290)
        )
        scale_values = values[visible & np.isfinite(values)]
        projection = ccrs.PlateCarree(central_longitude=180)
        title = "Anomalia mensal da SST — Pacífico equatorial"
    else:
        scale_values = values[np.isfinite(values)]
        projection = ccrs.Robinson(central_longitude=180)
        title = "Anomalia mensal da SST — contexto oceânico global"
        if (
            longitudes.size > 2
            and abs(longitudes[-1] - longitudes[0]) > 300
            and np.allclose(np.diff(longitudes), np.diff(longitudes)[0])
        ):
            values, longitudes = add_cyclic_point(values, coord=longitudes, axis=-1)
    if scale_values.size == 0:
        raise ScientificDataError("Sem anomalias válidas na área do mapa")
    limit = max(1.0, float(np.max(np.abs(scale_values))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig = Figure(figsize=(12, 7), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection=projection)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.83, bottom=0.22)
    mesh = ax.pcolormesh(
        longitudes,
        latitudes,
        np.ma.masked_invalid(values),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        rasterized=True,
        zorder=1,
    )
    if map_type == "pacific":
        ax.set_extent([120, 290, -35, 35], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    geography = features or MapFeatures.natural_earth()
    ax.add_feature(geography.land, zorder=3)
    ax.add_feature(geography.coastlines, linewidth=0.55, zorder=4)
    grid = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.45,
        color="#626a70",
        alpha=0.55,
        linestyle=":",
    )
    grid.top_labels = False
    grid.right_labels = False
    grid.xlabel_style = {"size": 8}
    grid.ylabel_style = {"size": 8}

    box_lon = [190, 240, 240, 190, 190]
    box_lat = [-5, -5, 5, 5, -5]
    ax.plot(
        box_lon, box_lat, transform=ccrs.PlateCarree(),
        color="#202428", linewidth=2.6, zorder=5,
    )
    ax.plot(
        box_lon, box_lat, transform=ccrs.PlateCarree(),
        color="#fff1a8", linewidth=1.25, zorder=6,
    )

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.96)
    fig.text(0.5, 0.89, f"{year:04d}-{month:02d}  |  Referência climatológica: {reference_start}–{reference_end}",
             ha="center", fontsize=11)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.055, pad=0.08, shrink=0.78)
    colorbar.set_label("Anomalia da temperatura da superfície do mar (°C)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.05, 0.075,
        f"Região Niño 3.4: 5°S–5°N, 170°W–120°W  |  Anomalia média mensal: {nino34_mean:+.2f} °C",
        fontsize=10, fontweight="bold",
    )
    fig.text(0.05, 0.045, f"Fonte: {source}", fontsize=8.5, color="#3d454a")
    if map_type == "global":
        fig.text(
            0.05, 0.02,
            "Mapa contextual de SST; não representa efeitos globais causados pelo El Niño.",
            fontsize=8, color="#3d454a",
        )
    return fig


def save_sst_anomaly_map(
    data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    *,
    year: int,
    month: int,
    map_type: MapType = "pacific",
    dpi: int = 150,
    output_dir: str | Path | None = None,
    formats: Sequence[OutputFormat] = ("png", "pdf"),
    source: str = "NOAA ERSSTv5",
    features: MapFeatures | None = None,
) -> tuple[Path, ...]:
    """Salva PNG/PDF em outputs/maps, com nomes que incluem mês e referência."""
    if not 50 <= dpi <= 600:
        raise ValueError("A resolução deve estar entre 50 e 600 DPI")
    selected_formats = tuple(dict.fromkeys(formats))
    if not selected_formats or any(fmt not in ("png", "pdf") for fmt in selected_formats):
        raise ValueError("Formatos permitidos: png e pdf")
    fig = create_sst_anomaly_figure(
        data, climatology, year=year, month=month,
        map_type=map_type, source=source, features=features,
    )
    destination = Path(output_dir) if output_dir is not None else get_settings().maps_dir
    destination.mkdir(parents=True, exist_ok=True)
    start = climatology.attrs["reference_start_year"]
    end = climatology.attrs["reference_end_year"]
    stem = f"sst_anomaly_{map_type}_{year:04d}-{month:02d}_ref_{start}-{end}"
    paths: list[Path] = []
    try:
        for fmt in selected_formats:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{fmt}.part", dir=destination, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                fig.savefig(temporary_path, format=fmt, dpi=dpi, facecolor="white")
                final_path = destination / f"{stem}.{fmt}"
                os.replace(temporary_path, final_path)
                paths.append(final_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    finally:
        fig.clear()
    return tuple(paths)

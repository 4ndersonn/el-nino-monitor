"""Mapas lado a lado de anomalias médias de dois períodos comparáveis."""

import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from el_nino_monitor.config import get_settings
from el_nino_monitor.processing.history import MonthPeriod, select_period
from el_nino_monitor.processing.sst import (
    ScientificDataError,
    calculate_nino34_mean,
    calculate_sst_anomaly,
)
from el_nino_monitor.visualization.maps import MapFeatures


def _period_field(
    data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    period: MonthPeriod,
) -> tuple[xr.DataArray, float]:
    selected = select_period(data, period)
    anomaly = calculate_sst_anomaly(selected, climatology)
    field = anomaly.mean(dim="time", skipna=True)
    nino = float(calculate_nino34_mean(anomaly).mean(dim="time", skipna=True).item())
    if not np.isfinite(nino):
        raise ScientificDataError(f"Niño 3.4 sem anomalias válidas em {period.label}")
    return field, nino


def _visible_values(
    values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
    map_type: str,
) -> np.ndarray:
    if map_type == "global":
        return values[np.isfinite(values)]
    visible = (
        (latitudes[:, None] >= -35)
        & (latitudes[:, None] <= 35)
        & (np.mod(longitudes[None, :], 360) >= 120)
        & (np.mod(longitudes[None, :], 360) <= 290)
    )
    return values[visible & np.isfinite(values)]


def create_comparison_map_figure(
    first_data: xr.Dataset | xr.DataArray,
    second_data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    first_period: MonthPeriod,
    second_period: MonthPeriod,
    *,
    map_type: Literal["pacific", "global"] = "pacific",
    source: str = "NOAA ERSSTv5",
    features: MapFeatures | None = None,
) -> Figure:
    """Mostra duas médias espaciais com a mesma normalização de cores."""
    if map_type not in ("pacific", "global"):
        raise ValueError(f"Tipo de mapa inválido: {map_type}")
    if len(first_period.months) != len(second_period.months):
        raise ScientificDataError("Mapas comparativos exigem períodos de mesma duração")
    if not source.strip():
        raise ValueError("A fonte dos dados deve ser informada")
    start = climatology.attrs.get("reference_start_year")
    end = climatology.attrs.get("reference_end_year")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ScientificDataError("Climatologia sem período declarado")

    fields = [
        _period_field(first_data, climatology, first_period),
        _period_field(second_data, climatology, second_period),
    ]
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    scale_values: list[np.ndarray] = []
    for field, nino in fields:
        latitude = "lat" if "lat" in field.coords else "latitude"
        longitude = "lon" if "lon" in field.coords else "longitude"
        field = field.transpose(latitude, longitude)
        latitudes = np.asarray(field[latitude].values, dtype=float)
        longitudes = np.asarray(field[longitude].values, dtype=float)
        values = np.asarray(field.values, dtype=float)
        scale_values.append(
            _visible_values(values, latitudes, longitudes, map_type)
        )
        if (
            map_type == "global"
            and longitudes.size > 2
            and abs(longitudes[-1] - longitudes[0]) > 300
            and np.allclose(np.diff(longitudes), np.diff(longitudes)[0])
        ):
            values, longitudes = add_cyclic_point(
                values, coord=longitudes, axis=-1
            )
        prepared.append((values, latitudes, longitudes, nino))
    valid = [item for item in scale_values if item.size]
    if not valid:
        raise ScientificDataError("Sem anomalias válidas nos períodos")
    limit = max(1.0, max(float(np.max(np.abs(item))) for item in valid))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    projection = (
        ccrs.PlateCarree(central_longitude=180)
        if map_type == "pacific"
        else ccrs.Robinson(central_longitude=180)
    )
    geography = features or MapFeatures.natural_earth()
    fig = Figure(figsize=(14, 6.8), facecolor="white")
    FigureCanvasAgg(fig)
    axes = [
        fig.add_subplot(1, 2, index, projection=projection)
        for index in (1, 2)
    ]
    fig.subplots_adjust(left=0.04, right=0.96, top=0.81, bottom=0.25, wspace=0.06)
    mesh = None
    for ax, (values, latitudes, longitudes, nino), period in zip(
        axes, prepared, (first_period, second_period)
    ):
        mesh = ax.pcolormesh(
            longitudes, latitudes, np.ma.masked_invalid(values),
            transform=ccrs.PlateCarree(), cmap="RdBu_r", norm=norm,
            shading="auto", rasterized=True, zorder=1,
        )
        if map_type == "pacific":
            ax.set_extent([120, 290, -35, 35], crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        ax.add_feature(geography.land, zorder=3)
        ax.add_feature(geography.coastlines, linewidth=0.45, zorder=4)
        grid = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.35, color="#626a70", alpha=0.5, linestyle=":",
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"size": 7}
        grid.ylabel_style = {"size": 7}
        box_lon = [190, 240, 240, 190, 190]
        box_lat = [-5, -5, 5, 5, -5]
        ax.plot(
            box_lon, box_lat, transform=ccrs.PlateCarree(),
            color="#202428", linewidth=2.2, zorder=5,
        )
        ax.plot(
            box_lon, box_lat, transform=ccrs.PlateCarree(),
            color="#fff1a8", linewidth=1.0, zorder=6,
        )
        ax.set_title(
            f"{period.label}\nNiño 3.4: {nino:+.2f} °C (média mensal do período)",
            fontsize=10, pad=12,
        )
    fig.suptitle(
        "Comparação espacial de anomalias SST" +
        (" — Pacífico" if map_type == "pacific" else " — contexto oceânico global"),
        fontsize=16, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.89, f"Mesma escala de cores  |  Climatologia {start}–{end}",
        ha="center", fontsize=10,
    )
    assert mesh is not None
    colorbar = fig.colorbar(
        mesh, ax=axes, orientation="horizontal",
        fraction=0.055, pad=0.12, shrink=0.74,
    )
    colorbar.set_label("Anomalia média da SST no período (°C)", fontsize=10)
    fig.text(0.04, 0.07, f"Fonte: {source}", fontsize=8.5)
    fig.text(
        0.04, 0.045,
        "Comparação de médias mensais na mesma região e referência; não é classificação ENSO.",
        fontsize=8, color="#4a5056",
    )
    if map_type == "global":
        fig.text(
            0.04, 0.02,
            "Mapa contextual de SST; não representa efeitos globais causados pelo El Niño.",
            fontsize=8, color="#4a5056",
        )
    return fig


def save_comparison_map(
    first_data: xr.Dataset | xr.DataArray,
    second_data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    first_period: MonthPeriod,
    second_period: MonthPeriod,
    *,
    map_type: Literal["pacific", "global"] = "pacific",
    output_dir: str | Path | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 150,
    source: str = "NOAA ERSSTv5",
    features: MapFeatures | None = None,
) -> tuple[Path, ...]:
    """Salva painéis comparativos PNG/PDF em outputs/maps."""
    if not 50 <= dpi <= 600:
        raise ValueError("A resolução deve estar entre 50 e 600 DPI")
    selected_formats = tuple(dict.fromkeys(formats))
    if not selected_formats or any(fmt not in ("png", "pdf") for fmt in selected_formats):
        raise ValueError("Formatos permitidos: png e pdf")
    fig = create_comparison_map_figure(
        first_data, second_data, climatology, first_period, second_period,
        map_type=map_type, source=source, features=features,
    )
    destination = Path(output_dir) if output_dir else get_settings().maps_dir
    destination.mkdir(parents=True, exist_ok=True)
    start = climatology.attrs["reference_start_year"]
    end = climatology.attrs["reference_end_year"]
    stem = (
        f"sst_anomaly_compare_{map_type}_"
        f"{first_period.start}_to_{first_period.end}_vs_"
        f"{second_period.start}_to_{second_period.end}_ref_{start}-{end}"
    )
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

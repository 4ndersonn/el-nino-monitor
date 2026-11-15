"""Gráficos temporais de anomalias calculadas para a região Niño 3.4."""

import os
from pathlib import Path
import tempfile
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from el_nino_monitor.config import get_settings
from el_nino_monitor.processing.history import (
    calculate_period_statistics,
    compare_periods,
)


def _month_labels(series: xr.Dataset) -> list[str]:
    return [
        f"{int(year):04d}-{int(month):02d}"
        for year, month in zip(
            series.time.dt.year.values, series.time.dt.month.values
        )
    ]


def _reference(series: xr.Dataset) -> str:
    return (
        f"{series.attrs['reference_start_year']}–"
        f"{series.attrs['reference_end_year']}"
    )


def create_history_figure(series: xr.Dataset) -> Figure:
    """Gráfico cronológico mensal com média móvel centrada de três meses."""
    statistics = calculate_period_statistics(series)
    labels = _month_labels(series)
    x = np.arange(len(labels))
    fig = Figure(figsize=(11, 5.8), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.axhline(0, color="#42484e", linewidth=0.9, linestyle="--", label="0 °C")
    ax.plot(
        x, series["sst_anomaly"].values,
        color="#9d4f45", linewidth=1.3, marker="o", markersize=3.5,
        label="Anomalia mensal",
    )
    ax.plot(
        x, series["sst_anomaly_3month_mean"].values,
        color="#172d58", linewidth=2.2, marker="o", markersize=4,
        label="Média móvel de 3 meses (centrada)",
    )
    tick_step = max(1, int(np.ceil(len(labels) / 10)))
    ticks = np.arange(0, len(labels), tick_step)
    ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=35, ha="right")
    ax.set_ylabel("Anomalia SST (°C)")
    ax.set_xlabel("Mês")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle("Série histórica de anomalias — região Niño 3.4",
                 fontsize=15, fontweight="bold", y=0.97)
    ax.set_title(
        f"{statistics.period.label}  |  Climatologia {_reference(series)}",
        fontsize=10, pad=12,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.23)
    fig.text(0.09, 0.08, f"Fonte: {series.attrs['source']}", fontsize=8.5)
    fig.text(
        0.09, 0.045,
        "Cálculo local de SST Niño 3.4; a média móvel não é ONI ou RONI oficial.",
        fontsize=8, color="#4a5056",
    )
    return fig


def create_period_comparison_figure(
    first: xr.Dataset, second: xr.Dataset
) -> Figure:
    """Compara dois períodos de mesma duração no eixo de mês relativo."""
    comparison = compare_periods(first, second)
    x = np.arange(comparison.first.total_months)
    fig = Figure(figsize=(11, 5.8), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.axhline(0, color="#42484e", linewidth=0.9, linestyle="--")
    for series, stats, color in (
        (first, comparison.first, "#b34b3b"),
        (second, comparison.second, "#24679b"),
    ):
        ax.plot(
            x, series["sst_anomaly"].values,
            color=color, alpha=0.4, linewidth=1.1, marker="o", markersize=3,
        )
        ax.plot(
            x, series["sst_anomaly_3month_mean"].values,
            color=color, linewidth=2.2, marker="o", markersize=5,
            label=f"{stats.period.label} — média de 3 meses",
        )
    ax.set_xticks(x, [str(index + 1) for index in x])
    ax.set_xlabel("Mês relativo ao início de cada período")
    ax.set_ylabel("Anomalia SST Niño 3.4 (°C)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", fontsize=8.5)
    fig.suptitle("Comparação de anomalias — região Niño 3.4",
                 fontsize=15, fontweight="bold", y=0.97)
    ax.set_title(
        f"Mesma climatologia {_reference(first)}  |  linhas tênues: anomalia mensal",
        fontsize=10, pad=12,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.23)
    fig.text(0.09, 0.08, f"Fonte: {first.attrs['source']}", fontsize=8.5)
    fig.text(
        0.09, 0.045,
        "Comparação descritiva local; não é ONI, RONI nem classificação operacional.",
        fontsize=8, color="#4a5056",
    )
    return fig


def _save_figure(
    figure: Figure,
    stem: str,
    *,
    output_dir: str | Path | None,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    if not 50 <= dpi <= 600:
        raise ValueError("A resolução deve estar entre 50 e 600 DPI")
    selected_formats = tuple(dict.fromkeys(formats))
    if not selected_formats or any(fmt not in ("png", "pdf") for fmt in selected_formats):
        raise ValueError("Formatos permitidos: png e pdf")
    destination = Path(output_dir) if output_dir else get_settings().reports_dir
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for fmt in selected_formats:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{fmt}.part", dir=destination, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                figure.savefig(
                    temporary_path, format=fmt, dpi=dpi, facecolor="white"
                )
                final_path = destination / f"{stem}.{fmt}"
                os.replace(temporary_path, final_path)
                paths.append(final_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    finally:
        figure.clear()
    return tuple(paths)


def save_history_plot(
    series: xr.Dataset,
    *,
    output_dir: str | Path | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 150,
) -> tuple[Path, ...]:
    """Exporta o gráfico cronológico PNG/PDF em outputs/reports."""
    stats = calculate_period_statistics(series)
    stem = (
        f"nino34_history_{stats.period.start}_to_{stats.period.end}_"
        f"ref_{stats.reference_period.replace('–', '-')}"
    )
    return _save_figure(
        create_history_figure(series), stem,
        output_dir=output_dir, formats=formats, dpi=dpi,
    )


def save_period_comparison_plot(
    first: xr.Dataset,
    second: xr.Dataset,
    *,
    output_dir: str | Path | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 150,
) -> tuple[Path, ...]:
    """Exporta a comparação temporal PNG/PDF em outputs/reports."""
    comparison = compare_periods(first, second)
    a = comparison.first.period
    b = comparison.second.period
    stem = (
        f"nino34_compare_{a.start}_to_{a.end}_"
        f"vs_{b.start}_to_{b.end}_"
        f"ref_{comparison.first.reference_period.replace('–', '-')}"
    )
    return _save_figure(
        create_period_comparison_figure(first, second), stem,
        output_dir=output_dir, formats=formats, dpi=dpi,
    )

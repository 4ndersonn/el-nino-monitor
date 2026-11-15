"""Análise temporal e comparação descritiva da região Niño 3.4."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from el_nino_monitor.config import get_settings
from el_nino_monitor.processing.sst import (
    ScientificDataError,
    calculate_nino34_timeseries,
    calculate_three_month_running_mean,
)


NINO34_AREA = "Niño 3.4 (5°S–5°N, 170°W–120°W)"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class MonthPeriod:
    """Período inclusivo em meses civis, por exemplo 1997-10 a 1998-03."""

    start: str
    end: str

    def __post_init__(self) -> None:
        if not MONTH_PATTERN.fullmatch(self.start) or not MONTH_PATTERN.fullmatch(self.end):
            raise ValueError("Datas do período devem usar YYYY-MM")
        if pd.Period(self.start, freq="M") > pd.Period(self.end, freq="M"):
            raise ValueError("Início do período posterior ao fim")

    @property
    def months(self) -> pd.PeriodIndex:
        return pd.period_range(self.start, self.end, freq="M")

    @property
    def label(self) -> str:
        return f"{self.start} a {self.end}"


@dataclass(frozen=True)
class PeriodStatistics:
    """Resumo de anomalias mensais, sempre com área e cobertura explícitas."""

    period: MonthPeriod
    area: str
    reference_period: str
    valid_months: int
    total_months: int
    mean_anomaly_c: float
    maximum_anomaly_c: float
    maximum_month: str
    minimum_anomaly_c: float
    minimum_month: str
    largest_monthly_increase_c: float | None
    largest_monthly_decrease_c: float | None


@dataclass(frozen=True)
class PeriodComparison:
    """Dois resumos comparáveis; não atribui classificação ENSO."""

    first: PeriodStatistics
    second: PeriodStatistics


def select_period(
    data: xr.Dataset | xr.DataArray, period: MonthPeriod
) -> xr.Dataset | xr.DataArray:
    """Seleciona todos os meses do período, rejeitando lacunas e duplicações."""
    if "time" not in data.coords:
        raise ScientificDataError("Dados sem coordenada temporal")
    try:
        years = np.asarray(data.time.dt.year.values, dtype=int)
        months = np.asarray(data.time.dt.month.values, dtype=int)
        available = pd.PeriodIndex(
            [f"{year:04d}-{month:02d}" for year, month in zip(years, months)],
            freq="M",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ScientificDataError("Datas mensais inválidas") from exc
    positions = np.flatnonzero(available.isin(period.months))
    selected_months = available[positions]
    if not selected_months.is_unique:
        raise ScientificDataError("Há meses duplicados no período selecionado")
    missing = period.months.difference(selected_months)
    if len(missing):
        raise ScientificDataError(
            f"Período incompleto; primeiro mês ausente: {missing[0]}"
        )
    if not selected_months.is_monotonic_increasing:
        raise ScientificDataError("Meses do período fora de ordem")
    return data.isel(time=positions)


def build_nino34_history(
    data: xr.Dataset | xr.DataArray,
    climatology: xr.DataArray,
    period: MonthPeriod,
    *,
    source: str = "NOAA ERSSTv5",
) -> xr.Dataset:
    """Calcula SST, anomalia mensal, média de 3 meses e variação mensal."""
    if not source.strip():
        raise ValueError("A fonte dos dados deve ser informada")
    selected = select_period(data, period)
    series = calculate_nino34_timeseries(selected, climatology)
    anomaly = series["sst_anomaly"]
    running = calculate_three_month_running_mean(anomaly)
    values = np.asarray(anomaly.values, dtype=float)
    changes = np.full(values.shape, np.nan, dtype=float)
    for index in range(1, len(values)):
        if np.isfinite(values[index - 1]) and np.isfinite(values[index]):
            changes[index] = values[index] - values[index - 1]
    series["sst_anomaly_3month_mean"] = running
    series["month_to_month_change"] = xr.DataArray(
        changes,
        dims=("time",),
        coords={"time": series.time},
        attrs={
            "long_name": "Variação da anomalia Niño 3.4 em relação ao mês anterior",
            "units": "°C",
        },
    )
    series.attrs.update(
        area=NINO34_AREA,
        period_start=period.start,
        period_end=period.end,
        source=source,
        warning="Série local; não é ONI ou RONI oficial da NOAA",
    )
    return series


def calculate_period_statistics(series: xr.Dataset) -> PeriodStatistics:
    """Resume a anomalia Niño 3.4 e a mudança entre meses válidos."""
    required = {"sst_anomaly", "month_to_month_change"}
    if not required.issubset(series.data_vars):
        raise ScientificDataError("Série histórica sem as variáveis necessárias")
    period = MonthPeriod(
        str(series.attrs.get("period_start", "")),
        str(series.attrs.get("period_end", "")),
    )
    values = np.asarray(series["sst_anomaly"].values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ScientificDataError("Período sem anomalias Niño 3.4 válidas")
    valid_positions = np.flatnonzero(finite)
    maximum_index = valid_positions[np.argmax(values[finite])]
    minimum_index = valid_positions[np.argmin(values[finite])]
    months = [
        f"{int(year):04d}-{int(month):02d}"
        for year, month in zip(
            series.time.dt.year.values, series.time.dt.month.values
        )
    ]
    changes = np.asarray(series["month_to_month_change"].values, dtype=float)
    valid_changes = changes[np.isfinite(changes)]
    increases = valid_changes[valid_changes > 0]
    decreases = valid_changes[valid_changes < 0]
    start = series.attrs.get("reference_start_year")
    end = series.attrs.get("reference_end_year")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ScientificDataError("Série sem referência climatológica")
    return PeriodStatistics(
        period=period,
        area=str(series.attrs.get("area", NINO34_AREA)),
        reference_period=f"{start}–{end}",
        valid_months=int(finite.sum()),
        total_months=len(values),
        mean_anomaly_c=float(np.mean(values[finite])),
        maximum_anomaly_c=float(values[maximum_index]),
        maximum_month=months[maximum_index],
        minimum_anomaly_c=float(values[minimum_index]),
        minimum_month=months[minimum_index],
        largest_monthly_increase_c=(
            float(np.max(increases)) if increases.size else None
        ),
        largest_monthly_decrease_c=(
            float(np.min(decreases)) if decreases.size else None
        ),
    )


def compare_periods(first: xr.Dataset, second: xr.Dataset) -> PeriodComparison:
    """Compara períodos de mesma duração, área e referência climatológica."""
    first_stats = calculate_period_statistics(first)
    second_stats = calculate_period_statistics(second)
    if first_stats.area != second_stats.area:
        raise ScientificDataError("Áreas distintas não são comparáveis nesta análise")
    if first_stats.reference_period != second_stats.reference_period:
        raise ScientificDataError("Referências climatológicas distintas")
    if first_stats.total_months != second_stats.total_months:
        raise ScientificDataError("Períodos de durações diferentes")
    if first.attrs.get("source") != second.attrs.get("source"):
        raise ScientificDataError("Fontes de dados distintas")
    return PeriodComparison(first=first_stats, second=second_stats)


def export_history_csv(
    series: xr.Dataset, output_dir: str | Path | None = None
) -> Path:
    """Exporta a série com área, fonte e referência em cada linha."""
    statistics = calculate_period_statistics(series)
    destination = Path(output_dir) if output_dir else get_settings().reports_dir
    destination.mkdir(parents=True, exist_ok=True)
    period = statistics.period
    filename = (
        f"nino34_history_{period.start}_to_{period.end}_"
        f"ref_{statistics.reference_period.replace('–', '-')}.csv"
    )
    labels = [
        f"{int(year):04d}-{int(month):02d}"
        for year, month in zip(
            series.time.dt.year.values, series.time.dt.month.values
        )
    ]
    table = pd.DataFrame({
        "month": labels,
        "area": statistics.area,
        "source": str(series.attrs.get("source", "")),
        "reference_period": statistics.reference_period,
        "sst_mean_c": series["sst_mean"].values,
        "monthly_anomaly_c": series["sst_anomaly"].values,
        "three_month_mean_c": series["sst_anomaly_3month_mean"].values,
        "month_to_month_change_c": series["month_to_month_change"].values,
    })
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv.part", dir=destination, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        table.to_csv(temporary_path, index=False, na_rep="", float_format="%.6f")
        final_path = destination / filename
        os.replace(temporary_path, final_path)
        return final_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

"""Climatologia, anomalias e médias espaciais de SST da região Niño 3.4.

Os resultados são cálculos locais sobre os dados fornecidos, não índices
operacionais publicados pela NOAA.
"""

import numpy as np
import pandas as pd
import xarray as xr

from el_nino_monitor.config import (
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
    Nino34Region,
)


SSTInput = xr.Dataset | xr.DataArray


class ScientificDataError(ValueError):
    """Dados insuficientes ou incompatíveis para o cálculo solicitado."""


def _require_celsius(array: xr.DataArray) -> None:
    units = str(array.attrs.get("units", "")).lower().replace("_", "").replace(" ", "")
    if units not in {
        "degc", "degreec", "degreesc", "degreecelsius", "degreescelsius", "celsius", "°c"
    }:
        raise ScientificDataError(f"Unidade de SST deve ser graus Celsius: {array.attrs.get('units')!r}")


def _sst_array(data: SSTInput) -> xr.DataArray:
    """Obtém a SST e remove somente dimensões extras de tamanho um."""
    if isinstance(data, xr.Dataset):
        if "sst" in data.data_vars:
            array = data["sst"]
        else:
            matches = [
                name
                for name, variable in data.data_vars.items()
                if variable.attrs.get("standard_name") == "sea_surface_temperature"
            ]
            if len(matches) != 1:
                raise ScientificDataError("Variável SST ausente ou ambígua")
            array = data[matches[0]]
    else:
        array = data

    latitude = _coordinate_name(array, ("lat", "latitude"))
    longitude = _coordinate_name(array, ("lon", "longitude"))
    if "time" not in array.dims or "time" not in array.coords:
        raise ScientificDataError("A SST deve ter coordenada mensal 'time'")
    required = {"time", latitude, longitude}
    extra = set(array.dims) - required
    if any(array.sizes[dimension] != 1 for dimension in extra):
        raise ScientificDataError("Dimensões extras da SST devem ter tamanho um")
    return array.squeeze(dim=list(extra), drop=True) if extra else array


def _coordinate_name(array: xr.DataArray, names: tuple[str, ...]) -> str:
    name = next((item for item in names if item in array.coords), None)
    if name is None or name not in array.dims or array[name].ndim != 1:
        raise ScientificDataError(f"Coordenada espacial ausente ou inválida: {names[0]}")
    return name


def _months(array: xr.DataArray) -> pd.PeriodIndex:
    """Verifica a ordem e a unicidade das observações mensais."""
    try:
        years = np.asarray(array.time.dt.year.values, dtype=int)
        months = np.asarray(array.time.dt.month.values, dtype=int)
        periods = pd.PeriodIndex(
            [f"{year:04d}-{month:02d}" for year, month in zip(years, months)],
            freq="M",
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ScientificDataError("Datas de SST inválidas") from exc
    if len(periods) == 0 or not periods.is_monotonic_increasing:
        raise ScientificDataError("Datas de SST ausentes ou fora de ordem")
    if not periods.is_unique:
        raise ScientificDataError("Há mais de uma observação no mesmo mês")
    return periods


def calculate_monthly_climatology(
    data: SSTInput,
    *,
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> xr.DataArray:
    """Calcula as 12 médias mensais espaciais do período inclusivo escolhido.

    Exige todos os meses de todos os anos do período. Células sem observações,
    como terra, permanecem NaN; dados faltantes em células individuais são
    ignorados pela média de cada mês.
    """
    if start_year < 1 or end_year < start_year:
        raise ValueError("Período climatológico inválido")
    sst = _sst_array(data)
    _require_celsius(sst)
    months = _months(sst)
    expected = pd.period_range(
        f"{start_year}-01", f"{end_year}-12", freq="M"
    )
    missing = expected.difference(months)
    if len(missing):
        raise ScientificDataError(
            f"Período climatológico incompleto; primeiro mês ausente: {missing[0]}"
        )
    reference = sst.where(
        (sst.time.dt.year >= start_year) & (sst.time.dt.year <= end_year),
        drop=True,
    )
    spatial_dims = [dimension for dimension in reference.dims if dimension != "time"]
    if not bool(reference.notnull().any(dim=spatial_dims).all().item()):
        raise ScientificDataError("Há meses sem qualquer SST válida no período")
    climatology = reference.groupby("time.month").mean(
        dim="time", skipna=True, keep_attrs=True
    )
    climatology.name = "monthly_sst_climatology"
    climatology.attrs.update(
        reference_start_year=start_year,
        reference_end_year=end_year,
        method="calculada a partir das observações mensais fornecidas",
        description="Média mensal espacial de SST; não é índice NOAA",
    )
    return climatology


def prepare_published_noaa_climatology(
    dataset: xr.Dataset,
    *,
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> xr.DataArray:
    """Converte uma climatologia mensal publicada pela NOAA em eixo 'month'.

    Esta função não recalcula a climatologia. Confere o período declarado no
    arquivo e mantém explícita a proveniência do produto pré-calculado.
    """
    if "sst" not in dataset or "time" not in dataset.coords:
        raise ScientificDataError("Climatologia NOAA sem SST ou eixo temporal")
    declared = dataset.time.attrs.get("climo_period")
    expected = f"{start_year}/01/01 - {end_year}/12/31"
    if declared != expected:
        raise ScientificDataError(
            f"Período declarado no arquivo ({declared!r}) difere de {expected}"
        )
    try:
        months = np.asarray(dataset.time.dt.month.values, dtype=int)
    except (AttributeError, TypeError) as exc:
        raise ScientificDataError("Meses da climatologia NOAA não decodificados") from exc
    if len(months) != 12 or set(months) != set(range(1, 13)):
        raise ScientificDataError("Climatologia NOAA deve conter os 12 meses")
    sst = dataset["sst"]
    if set(sst.dims) != {"time", "lat", "lon"}:
        raise ScientificDataError("Grade da climatologia NOAA inesperada")
    _require_celsius(sst)
    climatology = sst.rename(time="month").assign_coords(month=months).load()
    climatology.name = "monthly_sst_climatology"
    climatology.attrs.update(
        reference_start_year=start_year,
        reference_end_year=end_year,
        method="climatologia mensal pré-calculada publicada pela NOAA PSL",
    )
    return climatology


def calculate_sst_anomaly(
    data: SSTInput, climatology: xr.DataArray
) -> xr.DataArray:
    """Subtrai de cada observação a climatologia do mesmo mês do ano."""
    sst = _sst_array(data)
    _require_celsius(sst)
    _require_celsius(climatology)
    _months(sst)
    if "month" not in climatology.dims or set(climatology.month.values) != set(range(1, 13)):
        raise ScientificDataError("Climatologia deve conter exatamente os meses 1 a 12")
    latitude = _coordinate_name(sst, ("lat", "latitude"))
    longitude = _coordinate_name(sst, ("lon", "longitude"))
    for name in (latitude, longitude):
        if name not in climatology.coords or not np.array_equal(
            np.sort(sst[name].values), np.sort(climatology[name].values)
        ):
            raise ScientificDataError(f"Grade espacial incompatível na coordenada {name}")
        climatology = climatology.sel({name: sst[name]})
    anomaly = sst.groupby("time.month") - climatology
    anomaly.name = "sst_anomaly"
    anomaly.attrs = {
        "long_name": "Anomalia mensal de temperatura da superfície do mar",
        "units": str(sst.attrs.get("units", "")),
        "reference_start_year": climatology.attrs.get("reference_start_year"),
        "reference_end_year": climatology.attrs.get("reference_end_year"),
    }
    return anomaly


def select_nino34_region(
    data: SSTInput, region: Nino34Region | None = None
) -> xr.DataArray:
    """Seleciona 5°S–5°N e 170°W–120°W em ambas convenções de longitude."""
    sst = _sst_array(data)
    bounds = region or Nino34Region()
    latitude = _coordinate_name(sst, ("lat", "latitude"))
    longitude = _coordinate_name(sst, ("lon", "longitude"))
    lat_values = np.asarray(sst[latitude].values, dtype=float)
    lon_values = np.mod(np.asarray(sst[longitude].values, dtype=float), 360.0)
    if not np.isfinite(lat_values).all() or not np.isfinite(lon_values).all():
        raise ScientificDataError("Coordenadas espaciais não finitas")
    lat_mask = (lat_values >= bounds.latitude_min) & (lat_values <= bounds.latitude_max)
    west = bounds.longitude_min % 360
    east = bounds.longitude_max % 360
    lon_mask = (
        (lon_values >= west) & (lon_values <= east)
        if west <= east
        else (lon_values >= west) | (lon_values <= east)
    )
    if not lat_mask.any() or not lon_mask.any():
        raise ScientificDataError("Grade sem células na região Niño 3.4")
    return sst.isel(
        {latitude: np.flatnonzero(lat_mask), longitude: np.flatnonzero(lon_mask)}
    )


def calculate_nino34_mean(
    data: SSTInput, region: Nino34Region | None = None
) -> xr.DataArray:
    """Calcula média espacial com pesos proporcionais a cos(latitude).

    Para grade regular, cos(latitude) aproxima a área relativa das células.
    Células NaN são excluídas do numerador e do denominador a cada mês.
    """
    subset = select_nino34_region(data, region)
    latitude = _coordinate_name(subset, ("lat", "latitude"))
    longitude = _coordinate_name(subset, ("lon", "longitude"))
    weights = np.cos(np.deg2rad(subset[latitude]))
    mean = subset.weighted(weights).mean(dim=(latitude, longitude), skipna=True)
    mean.name = "nino34_mean"
    mean.attrs = {
        "long_name": "Média espacial mensal da região Niño 3.4",
        "units": str(subset.attrs.get("units", "")),
        "weighting": "cos(latitude)",
    }
    return mean


def calculate_nino34_timeseries(
    data: SSTInput, climatology: xr.DataArray
) -> xr.Dataset:
    """Retorna SST média e anomalia mensal médias na região Niño 3.4."""
    sst_mean = calculate_nino34_mean(data).rename("sst_mean")
    anomaly = calculate_sst_anomaly(data, climatology)
    anomaly_mean = calculate_nino34_mean(anomaly).rename("sst_anomaly")
    result = xr.Dataset({"sst_mean": sst_mean, "sst_anomaly": anomaly_mean})
    result.attrs = {
        "reference_start_year": climatology.attrs.get("reference_start_year"),
        "reference_end_year": climatology.attrs.get("reference_end_year"),
        "climatology_method": climatology.attrs.get("method", "não informado"),
        "warning": "Série local; não é ONI ou RONI oficial da NOAA",
    }
    return result


def calculate_three_month_running_mean(
    monthly_anomaly: xr.DataArray,
) -> xr.DataArray:
    """Média centrada de três meses; lacunas e extremos permanecem NaN."""
    if monthly_anomaly.dims != ("time",):
        raise ScientificDataError("A série de anomalias deve ter apenas a dimensão time")
    months = _months(monthly_anomaly)
    result = monthly_anomaly.rolling(
        time=3, center=True, min_periods=3
    ).mean()
    valid = np.zeros(len(months), dtype=bool)
    for index in range(1, len(months) - 1):
        valid[index] = (
            months[index - 1] + 1 == months[index]
            and months[index] + 1 == months[index + 1]
        )
    result = result.where(
        xr.DataArray(valid, dims="time", coords={"time": monthly_anomaly.time})
    )
    result.name = "nino34_anomaly_3month_mean"
    result.attrs = {
        "long_name": "Média móvel centrada de 3 meses da anomalia Niño 3.4",
        "units": str(monthly_anomaly.attrs.get("units", "")),
        "warning": "Cálculo local; não é ONI ou RONI oficial da NOAA",
    }
    return result

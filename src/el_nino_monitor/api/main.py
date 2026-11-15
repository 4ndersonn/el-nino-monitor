"""Endpoints HTTP de leitura para resultados científicos locais."""

from contextlib import asynccontextmanager
from io import BytesIO
import logging
import math
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from el_nino_monitor.api.service import LocalDataService
from el_nino_monitor.config import Settings, get_settings
from el_nino_monitor.data.noaa_client import DateUnavailableError, DatasetValidationError
from el_nino_monitor.processing.history import (
    MonthPeriod,
    build_nino34_history,
    calculate_period_statistics,
    compare_periods,
)
from el_nino_monitor.processing.sst import ScientificDataError
from el_nino_monitor.visualization.comparison import create_comparison_map_figure
from el_nino_monitor.visualization.maps import MapFeatures, create_sst_anomaly_figure


LOGGER = logging.getLogger("el_nino_monitor.api")
METHODOLOGY = "ERSSTv5; anomalia mensal SST menos climatologia; Niño 3.4 ponderado por cos(latitude)"


class ScientificValue(BaseModel):
    value: float | None
    unit: str = "°C"


class Observation(BaseModel):
    month: str
    observation_date: str
    sst_mean: ScientificValue
    monthly_anomaly: ScientificValue
    three_month_mean: ScientificValue
    month_to_month_change: ScientificValue


class HistoryResponse(BaseModel):
    area: str
    period_start: str
    period_end: str
    reference_period: str
    source: str
    methodology: str
    climatology_method: str
    local_updated_at_utc: str
    observations: list[Observation]
    note: str = "Cálculo local; média móvel não é ONI ou RONI oficial"


def _finite(value: object) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _period(start: str, end: str) -> MonthPeriod:
    try:
        return MonthPeriod(start, end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _history_payload(
    series: object, source: str, updated: str
) -> HistoryResponse:
    stats = calculate_period_statistics(series)
    observations = []
    for index in range(series.sizes["time"]):
        instant = series.time.isel(time=index)
        year, month, day = (
            int(instant.dt.year.item()),
            int(instant.dt.month.item()),
            int(instant.dt.day.item()),
        )
        observations.append(Observation(
            month=f"{year:04d}-{month:02d}",
            observation_date=f"{year:04d}-{month:02d}-{day:02d}",
            sst_mean=ScientificValue(value=_finite(series.sst_mean.values[index])),
            monthly_anomaly=ScientificValue(value=_finite(series.sst_anomaly.values[index])),
            three_month_mean=ScientificValue(
                value=_finite(series.sst_anomaly_3month_mean.values[index])
            ),
            month_to_month_change=ScientificValue(
                value=_finite(series.month_to_month_change.values[index])
            ),
        ))
    return HistoryResponse(
        area=stats.area,
        period_start=stats.period.start,
        period_end=stats.period.end,
        reference_period=stats.reference_period,
        source=source,
        methodology=METHODOLOGY,
        climatology_method=str(series.attrs["climatology_method"]),
        local_updated_at_utc=updated,
        observations=observations,
    )


def _stats_payload(series: object) -> dict[str, object]:
    stats = calculate_period_statistics(series)
    return {
        "area": stats.area,
        "period_start": stats.period.start,
        "period_end": stats.period.end,
        "valid_months": stats.valid_months,
        "total_months": stats.total_months,
        "mean_anomaly": ScientificValue(value=stats.mean_anomaly_c).model_dump(),
        "maximum_monthly_anomaly": {
            "month": stats.maximum_month,
            **ScientificValue(value=stats.maximum_anomaly_c).model_dump(),
        },
        "minimum_monthly_anomaly": {
            "month": stats.minimum_month,
            **ScientificValue(value=stats.minimum_anomaly_c).model_dump(),
        },
        "largest_monthly_increase": ScientificValue(
            value=stats.largest_monthly_increase_c
        ).model_dump(),
        "largest_monthly_decrease": ScientificValue(
            value=stats.largest_monthly_decrease_c
        ).model_dump(),
    }


def create_app(
    settings: Settings | None = None,
    *,
    map_features: MapFeatures | None = None,
) -> FastAPI:
    """Permite testar a API com dados locais isolados e feições sem rede."""
    service = LocalDataService(settings or get_settings())

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        service.close()

    application = FastAPI(
        title="El Niño Monitor API", version="1.0.0",
        description="Resultados locais ERSSTv5; não representa monitoramento em tempo real.",
        lifespan=lifespan,
    )
    application.state.data_service = service

    @application.exception_handler(FileNotFoundError)
    async def missing_file(request: Request, exc: FileNotFoundError) -> JSONResponse:
        LOGGER.warning("Dados locais ausentes em %s: %s", request.url.path, exc)
        status = 404 if str(exc).startswith("Mês ") else 503
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @application.exception_handler(DateUnavailableError)
    async def unavailable_date(request: Request, exc: DateUnavailableError) -> JSONResponse:
        LOGGER.warning("Data indisponível em %s: %s", request.url.path, exc)
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @application.exception_handler(ScientificDataError)
    async def invalid_science(request: Request, exc: ScientificDataError) -> JSONResponse:
        LOGGER.error("Dados científicos inválidos em %s: %s", request.url.path, exc)
        status = 404 if "primeiro mês ausente" in str(exc) else 422
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @application.exception_handler(DatasetValidationError)
    async def invalid_dataset(request: Request, exc: DatasetValidationError) -> JSONResponse:
        LOGGER.error("NetCDF inválido em %s: %s", request.url.path, exc)
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @application.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Falha inesperada em %s", request.url.path)
        return JSONResponse({"detail": "Falha interna"}, status_code=500)

    def calculate(period: MonthPeriod):
        data, source, paths = service.observations(period)
        climatology, baseline_paths = service.climatology()
        series = build_nino34_history(
            data, climatology, period, source=source
        )
        updated = service.updated_at(paths + baseline_paths)
        return data, climatology, series, source, updated

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local_cache", "realtime": "false"}

    @application.get("/api/v1/datasets")
    def datasets() -> dict[str, object]:
        return {"datasets": service.available_datasets(), "realtime": False}

    @application.get("/api/v1/enso/nino34", response_model=HistoryResponse)
    def nino34(
        year: int = Query(ge=1854, le=9999),
        month: int = Query(ge=1, le=12),
    ) -> HistoryResponse:
        period = _period(f"{year:04d}-{month:02d}", f"{year:04d}-{month:02d}")
        _, _, series, source, updated = calculate(period)
        return _history_payload(series, source, updated)

    @application.get("/api/v1/enso/history", response_model=HistoryResponse)
    def history(start: str, end: str) -> HistoryResponse:
        _, _, series, source, updated = calculate(_period(start, end))
        return _history_payload(series, source, updated)

    @application.get("/api/v1/enso/compare")
    def compare(
        start: str, end: str, other_start: str, other_end: str
    ) -> dict[str, object]:
        first_period = _period(start, end)
        second_period = _period(other_start, other_end)
        _, _, first, source_a, updated_a = calculate(first_period)
        _, _, second, source_b, updated_b = calculate(second_period)
        compare_periods(first, second)
        return {
            "first": _stats_payload(first),
            "second": _stats_payload(second),
            "reference_period": calculate_period_statistics(first).reference_period,
            "source": source_a,
            "methodology": METHODOLOGY,
            "climatology_method": first.attrs["climatology_method"],
            "local_updated_at_utc": max(updated_a, updated_b),
            "observation_dates_first": [
                item.observation_date for item in _history_payload(first, source_a, updated_a).observations
            ],
            "observation_dates_second": [
                item.observation_date for item in _history_payload(second, source_b, updated_b).observations
            ],
            "note": "Comparação descritiva local; não é classificação ENSO oficial",
        }

    @application.get(
        "/api/v1/enso/map", response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "Mapa científico PNG"}},
    )
    def map_png(
        year: int = Query(ge=1854, le=9999),
        month: int = Query(ge=1, le=12),
        region: Literal["pacific", "global"] = "pacific",
        visualization: Literal["anomaly"] = "anomaly",
    ) -> Response:
        period = _period(f"{year:04d}-{month:02d}", f"{year:04d}-{month:02d}")
        data, climatology, series, source, updated = calculate(period)
        figure = create_sst_anomaly_figure(
            data, climatology, year=year, month=month,
            map_type=region, source=source, features=map_features,
        )
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120, facecolor="white")
        finally:
            figure.clear()
        return Response(
            content=buffer.getvalue(), media_type="image/png",
            headers={
                "X-Observation-Month": period.start,
                "X-Reference-Period": calculate_period_statistics(series).reference_period.replace("–", "-"),
                "X-Data-Source": source,
                "X-Local-Updated-At": updated,
                "X-Methodology-Id": "ersstv5-monthly-nino34-coslat",
            },
        )

    @application.get(
        "/api/v1/enso/compare/map", response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "Mapas comparativos PNG"}},
    )
    def comparison_map_png(
        start: str, end: str, other_start: str, other_end: str,
        region: Literal["pacific", "global"] = "pacific",
    ) -> Response:
        first_period = _period(start, end)
        second_period = _period(other_start, other_end)
        first_data, climatology, first, source, first_updated = calculate(first_period)
        second_data, _, second, _, second_updated = calculate(second_period)
        compare_periods(first, second)
        figure = create_comparison_map_figure(
            first_data, second_data, climatology, first_period, second_period,
            map_type=region, source=source, features=map_features,
        )
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=120, facecolor="white")
        finally:
            figure.clear()
        return Response(
            content=buffer.getvalue(), media_type="image/png",
            headers={
                "X-Reference-Period": calculate_period_statistics(first).reference_period.replace("–", "-"),
                "X-Data-Source": source,
                "X-Local-Updated-At": max(first_updated, second_updated),
                "X-Methodology-Id": "ersstv5-period-anomaly-comparison",
            },
        )

    return application


app = create_app()

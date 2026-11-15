"""Cliente HTTP e seleção de meses para a interface científica."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests


class DashboardAPIError(RuntimeError):
    """Resposta indisponível ou inválida da API local."""


@dataclass
class DashboardAPIClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: tuple[float, float] = (5.0, 90.0)

    def _get(self, path: str, params: dict[str, object] | None = None) -> requests.Response:
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}{path}",
                params=params, timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = None
            if exc.response is not None:
                try:
                    detail = exc.response.json().get("detail")
                except (ValueError, AttributeError):
                    pass
            raise DashboardAPIError(str(detail or exc)) from exc
        except requests.RequestException as exc:
            raise DashboardAPIError(f"API indisponível: {exc}") from exc
        return response

    def json(self, path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        response = self._get(path, params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise DashboardAPIError("A API retornou JSON inválido") from exc
        if not isinstance(payload, dict):
            raise DashboardAPIError("Resposta JSON inesperada da API")
        return payload

    def png(self, path: str, params: dict[str, object] | None = None) -> bytes:
        response = self._get(path, params)
        if response.headers.get("content-type", "").split(";")[0] != "image/png":
            raise DashboardAPIError("A API não retornou um mapa PNG")
        if not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DashboardAPIError("Imagem PNG inválida")
        return response.content


def available_months(datasets: list[dict[str, Any]]) -> list[str]:
    """Retorna somente meses declarados, sem preencher lacunas implicitamente."""
    months = {
        month
        for dataset in datasets
        for month in (dataset.get("available_months") or [])
        if isinstance(month, str) and len(month) == 7
    }
    return sorted(months)


def contiguous_runs(months: list[str]) -> list[tuple[str, str]]:
    """Agrupa meses consecutivos presentes no inventário da API."""
    if not months:
        return []
    periods = sorted({pd.Period(month, freq="M") for month in months})
    runs: list[tuple[str, str]] = []
    start = previous = periods[0]
    for current in periods[1:]:
        if current != previous + 1:
            runs.append((str(start), str(previous)))
            start = current
        previous = current
    runs.append((str(start), str(previous)))
    return runs


def observation_is_stale(month: str, today: date | None = None, *, days: int = 60) -> bool:
    """Sinaliza arquivo cujo mês observado terminou há mais de 60 dias."""
    current = today or date.today()
    year, number = map(int, month.split("-"))
    end = date(year, number, monthrange(year, number)[1])
    return (current - end).days > days

"""Seleção de dados e interface Streamlit com respostas sintéticas da API."""

from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
import requests
from streamlit.testing.v1 import AppTest

from el_nino_monitor.dashboard.client import (
    DashboardAPIClient,
    DashboardAPIError,
    available_months,
    contiguous_runs,
    observation_is_stale,
)


def test_month_inventory_and_staleness() -> None:
    months = available_months([
        {"available_months": ["1997-11", "1997-12", "1998-01"]},
        {"available_months": ["2015-11", "2015-12", "2016-01", "1997-12"]},
    ])
    assert contiguous_runs(months) == [
        ("1997-11", "1998-01"), ("2015-11", "2016-01")
    ]
    assert observation_is_stale("2026-10", date(2026, 11, 15)) is False
    assert observation_is_stale("2026-08", date(2026, 11, 15)) is True
    from dashboard import _period_defaults
    assert _period_defaults([], "2026-10") == ("2026-10",) * 4


def test_http_client_reports_api_error_and_invalid_png() -> None:
    response = requests.Response()
    response.status_code = 404
    response._content = b'{"detail":"Mes ausente"}'
    with patch("requests.get", return_value=response):
        with pytest.raises(DashboardAPIError, match="Mes ausente"):
            DashboardAPIClient().json("/api/v1/enso/nino34")
    response.status_code = 200
    response.headers["content-type"] = "image/png"
    response._content = b"invalid"
    with patch("requests.get", return_value=response):
        with pytest.raises(DashboardAPIError, match="PNG inválida"):
            DashboardAPIClient().png("/api/v1/enso/map")


def test_streamlit_shows_api_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("teste: API desligada")
    monkeypatch.setattr(requests, "get", unavailable)
    from dashboard import api_json
    api_json.clear()
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(str(dashboard_path), default_timeout=30).run()
    assert not app.exception
    assert any("API" in message.value for message in app.error)


def test_streamlit_renders_data_from_api(monkeypatch: pytest.MonkeyPatch) -> None:
    months = ["1997-11", "1997-12", "1998-01", "2015-11", "2015-12", "2016-01"]
    picture = BytesIO()
    Image.new("RGB", (8, 8), "navy").save(picture, format="PNG")

    def response_json(url, params):
        if url.endswith("/datasets"):
            return {"datasets": [{
                "time_start": "1997-11-15", "time_end": "2016-01-15",
                "available_months": months,
            }]}
        if url.endswith("/enso/compare"):
            stats = {"area": "Niño 3.4", "mean_anomaly": {"value": 1.0}}
            return {"first": stats, "second": stats, "reference_period": "1991–2020",
                    "source": "NOAA ERSSTv5"}
        if url.endswith("/enso/nino34"):
            return {"source": "NOAA ERSSTv5", "reference_period": "1991–2020",
                    "local_updated_at_utc": "2026-11-15T00:00:00+00:00",
                    "observations": [{"observation_date": "2016-01-15",
                                      "monthly_anomaly": {"value": 1.0},
                                      "sst_mean": {"value": 26.0}}]}
        if url.endswith("/enso/history"):
            return {"source": "NOAA ERSSTv5", "area": "Niño 3.4",
                    "reference_period": "1991–2020", "observations": [
                        {"month": month, "monthly_anomaly": {"value": float(index)},
                         "three_month_mean": {"value": None},
                         "month_to_month_change": {"value": None}}
                        for index, month in enumerate(months[-3:])
                    ]}
        raise AssertionError(url)

    def fake_get(url, params=None, timeout=None):
        result = requests.Response()
        result.status_code = 200
        if url.endswith("/map"):
            result.headers["content-type"] = "image/png"
            result._content = picture.getvalue()
        else:
            import json
            result.headers["content-type"] = "application/json"
            result._content = json.dumps(response_json(url, params)).encode()
        return result

    monkeypatch.setattr(requests, "get", fake_get)
    from dashboard import api_json, api_png
    api_json.clear()
    api_png.clear()
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(str(dashboard_path), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) >= 4
    assert any("desatualizados" in item.value.lower() for item in app.warning)

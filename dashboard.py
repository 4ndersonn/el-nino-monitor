"""Dashboard Streamlit; toda observação e cálculo vêm da API FastAPI."""

import os

import pandas as pd
import streamlit as st

from el_nino_monitor.dashboard.client import (
    DashboardAPIClient,
    DashboardAPIError,
    available_months,
    contiguous_runs,
    observation_is_stale,
)


API_URL = os.getenv("EL_NINO_API_URL", "http://127.0.0.1:8000")
ROOT = "/api/v1"


@st.cache_data(ttl=60, show_spinner=False)
def api_json(path: str, params: tuple[tuple[str, object], ...] = ()) -> dict:
    return DashboardAPIClient(API_URL).json(ROOT + path, dict(params))


@st.cache_data(ttl=300, show_spinner=False)
def api_png(path: str, params: tuple[tuple[str, object], ...]) -> bytes:
    return DashboardAPIClient(API_URL).png(ROOT + path, dict(params))


def _params(**values: object) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(values.items()))


def _value(payload: dict, key: str) -> float | None:
    return payload.get(key, {}).get("value")


def _temperature(value: float | None) -> str:
    return "Indisponível" if value is None else f"{value:+.2f} °C"


def _style() -> None:
    st.markdown("""
    <style>
      .stApp {background: #091827; color: #e4edf5;}
      [data-testid="stSidebar"] {background: #102338;}
      [data-testid="stMetric"] {
        background: #172f47; border: 1px solid #2c4a63;
        border-radius: 14px; padding: 1.05rem 1.2rem; min-height: 125px;
      }
      [data-testid="stMetricLabel"] {color: #a9c2d4;}
      [data-testid="stMetricValue"] {color: #dff5ff;}
      h1, h2, h3 {color: #f3f8fc;}
      .hero {padding: .4rem 0 1.2rem; border-bottom: 1px solid #31506a; margin-bottom: 1.2rem;}
      .hero h1 {font-size: 2.45rem; letter-spacing: -.025em; margin: 0;}
      .hero p {font-size: 1.05rem; color: #aac9dc; margin: .35rem 0 0;}
      .eyebrow {color: #73d1ef; font-size: .78rem; letter-spacing: .15em; font-weight: 700;}
      .stTabs [data-baseweb="tab"] {font-weight: 650;}
      .science-note {background: #152c43; border-left: 3px solid #66c9e8;
                     border-radius: 6px; padding: .8rem 1rem; color: #c7dae8;}
    </style>
    """, unsafe_allow_html=True)


def _inventory() -> tuple[list[dict], list[str], str | None]:
    payload = api_json("/datasets")
    datasets = payload.get("datasets", [])
    months = available_months(datasets)
    latest = max((str(item.get("time_end", ""))[:7] for item in datasets), default="")
    return datasets, months, latest or None


def _period_defaults(months: list[str], latest: str) -> tuple[str, str, str, str]:
    runs = contiguous_runs(months)
    long_runs = [
        (start, end) for start, end in runs
        if len(pd.period_range(start, end, freq="M")) >= 3
    ]
    if len(long_runs) >= 2:
        first, second = long_runs[-2:]
        first_length = len(pd.period_range(*first, freq="M"))
        second_length = len(pd.period_range(*second, freq="M"))
        length = min(first_length, second_length)
        first_end = str(pd.Period(first[0], freq="M") + length - 1)
        second_end = str(pd.Period(second[0], freq="M") + length - 1)
        return first[0], first_end, second[0], second_end
    if len(months) >= 2:
        return months[-2], months[-2], months[-1], months[-1]
    return latest, latest, latest, latest


def _indicators(latest: str | None, months: list[str]) -> None:
    st.subheader("Indicadores principais")
    if latest is None:
        st.info("Nenhuma observação ERSSTv5 foi encontrada no cache da API.")
        return
    year, month = map(int, latest.split("-"))
    try:
        payload = api_json("/enso/nino34", _params(year=year, month=month))
        observation = payload["observations"][0]
    except (DashboardAPIError, KeyError, IndexError) as exc:
        st.warning(f"Indicadores indisponíveis: {exc}")
        return
    trend = None
    previous = str(pd.Period(latest, freq="M") - 1)
    if previous in months:
        try:
            recent = api_json("/enso/history", _params(start=previous, end=latest))
            trend = _value(recent["observations"][-1], "month_to_month_change")
        except (DashboardAPIError, KeyError, IndexError):
            pass
    columns = st.columns(4)
    columns[0].metric("Anomalia Niño 3.4", _temperature(_value(observation, "monthly_anomaly")))
    columns[1].metric("Temperatura média Niño 3.4", _temperature(_value(observation, "sst_mean")))
    columns[2].metric("Tendência recente · variação mensal", _temperature(trend))
    columns[3].metric("Último período observado", latest)
    st.caption(
        f"Fonte: {payload['source']} · Climatologia {payload['reference_period']} · "
        f"Observação: {observation['observation_date']} · "
        f"Atualização local: {payload['local_updated_at_utc']}. "
        "A tendência requer dois meses consecutivos."
    )


def _map_panel(selected: str, region: str) -> None:
    st.subheader("Anomalias da temperatura da superfície do mar")
    st.caption("A barra de cores do mapa expressa a anomalia SST em °C; azul indica valores negativos e vermelho, positivos.")
    year, month = map(int, selected.split("-"))
    try:
        with st.spinner("Carregando mapa científico da API…"):
            picture = api_png("/enso/map", _params(
                year=year, month=month, region=region, visualization="anomaly"
            ))
        st.image(picture, width="stretch", caption=f"Anomalia SST · {selected} · {region.title()}")
    except DashboardAPIError as exc:
        st.error(f"Mapa indisponível: {exc}")
    if region == "global":
        st.info("O mapa global contextualiza as anomalias da SST. Ele não representa efeitos globais causados pelo El Niño.")


def _history_panel(default_start: str, default_end: str) -> None:
    st.subheader("Histórico Niño 3.4")
    left, right = st.columns(2)
    start = left.text_input("Mês inicial · AAAA-MM", value=default_start, key="history_start")
    end = right.text_input("Mês final · AAAA-MM", value=default_end, key="history_end")
    try:
        with st.spinner("Carregando série temporal…"):
            payload = api_json("/enso/history", _params(start=start, end=end))
    except DashboardAPIError as exc:
        st.warning(f"Série indisponível: {exc}")
        return
    observations = payload.get("observations", [])
    if not observations:
        st.info("Nenhuma anomalia válida no período selecionado.")
        return
    table = pd.DataFrame({
        "Mês": pd.to_datetime([row["month"] for row in observations]),
        "Anomalia mensal": [_value(row, "monthly_anomaly") for row in observations],
        "Média móvel de 3 meses": [_value(row, "three_month_mean") for row in observations],
        "Referência 0 °C": [0.0] * len(observations),
    }).set_index("Mês")
    st.line_chart(table, color=["#f28b75", "#6ad0ef", "#8f9eac"], y_label="Anomalia SST (°C)")
    st.caption(
        f"Região: {payload['area']} · Climatologia {payload['reference_period']} · "
        f"Fonte: {payload['source']}. A média móvel é centrada e não é ONI/RONI oficial."
    )
    with st.expander("Ver valores mensais"):
        st.dataframe(table, width="stretch")


def _compare_panel(defaults: tuple[str, str, str, str], region: str) -> None:
    st.subheader("Comparador histórico")
    st.caption("Compare períodos de mesma duração e a mesma climatologia. O mapa usa uma única escala de cores nos dois painéis.")
    a, b = st.columns(2)
    with a:
        start = st.text_input("Período A · início", value=defaults[0])
        end = st.text_input("Período A · fim", value=defaults[1])
    with b:
        other_start = st.text_input("Período B · início", value=defaults[2])
        other_end = st.text_input("Período B · fim", value=defaults[3])
    if not st.button("Comparar períodos", type="primary"):
        return
    params = _params(start=start, end=end, other_start=other_start, other_end=other_end)
    try:
        with st.spinner("Calculando comparação e carregando mapas…"):
            comparison = api_json("/enso/compare", params)
            picture = api_png("/enso/compare/map", params + (("region", region),))
    except DashboardAPIError as exc:
        st.warning(f"Comparação indisponível: {exc}")
        return
    first, second = comparison["first"], comparison["second"]
    left, right = st.columns(2)
    left.metric("Anomalia média · período A", _temperature(_value(first, "mean_anomaly")))
    right.metric("Anomalia média · período B", _temperature(_value(second, "mean_anomaly")))
    st.image(picture, width="stretch", caption="Médias espaciais das anomalias SST · escala compartilhada")
    st.caption(
        f"Região: {first['area']} · Climatologia {comparison['reference_period']} · "
        f"Fonte: {comparison['source']}. Comparação descritiva; não é classificação ENSO."
    )


def _science_panel() -> None:
    st.subheader("Informações científicas")
    st.markdown("""
    **El Niño** é uma fase de aquecimento persistente do Pacífico equatorial central e oriental.
    **ENSO** é a oscilação acoplada oceano–atmosfera que inclui fases quentes, frias e neutras.

    **Anomalia SST** é a diferença entre a temperatura observada da superfície do mar e a
    média climatológica do mesmo mês. Aqui a referência é 1991–2020, quando disponível.

    **Niño 3.4** abrange 5°S–5°N e 170°W–120°W. As médias espaciais usam pesos
    proporcionais a `cos(latitude)`, aproximação para células regulares.

    **Fontes:** observações NOAA ERSSTv5 do NCEI ou PSL, e climatologia mensal
    1991–2020 publicada pela NOAA PSL ou calculada localmente quando há série completa.

    **Limites:** o painel mostra cálculos locais, sujeitos à cobertura e à atualização
    dos arquivos disponíveis. A média móvel de três meses não equivale necessariamente
    ao ONI ou RONI oficial. O painel não classifica operacionalmente ENSO, não faz
    previsões e não atribui desastres naturais ao El Niño.
    """)


def main() -> None:
    st.set_page_config(page_title="El Niño Monitor", page_icon="🌊", layout="wide")
    _style()
    st.markdown(
        '<div class="hero"><div class="eyebrow">MONITORAMENTO OCEÂNICO</div>'
        '<h1>El Niño Monitor</h1><p>Plataforma de Monitoramento Oceânico e ENSO</p></div>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Consultando o inventário local da API…"):
            datasets, months, latest = _inventory()
    except DashboardAPIError as exc:
        st.error(f"Não foi possível consultar a API: {exc}")
        st.info("Inicie a API FastAPI antes de abrir o dashboard. Consulte o README para os comandos.")
        return
    if not datasets or latest is None:
        st.warning("Nenhum conjunto ERSSTv5 local disponível na API. O dashboard não exibe valores estimados.")
        _science_panel()
        return
    stale = observation_is_stale(latest)
    st.caption(f"Última observação disponível: {latest} · Arquivos locais, sem atualização em tempo real.")
    if stale:
        st.warning(f"Dados desatualizados: o último mês observado é {latest}. Verifique a atualização dos arquivos NOAA locais.")
    st.sidebar.header("Explorar dados")
    st.sidebar.caption("Seleção baseada nos arquivos disponíveis na API.")
    aggregate = next((item for item in datasets if item.get("id") == "aggregated"), None)
    years_set = {int(item[:4]) for item in months}
    if aggregate is not None:
        years_set.update(range(
            int(aggregate["time_start"][:4]),
            int(aggregate["time_end"][:4]) + 1,
        ))
        st.sidebar.info("Arquivo agregado: a API confirmará a disponibilidade do mês escolhido.")
    years = sorted(years_set, reverse=True)
    year = st.sidebar.selectbox("Ano", years)
    available_for_year = [int(month[5:]) for month in months if int(month[:4]) == year]
    choices = (list(range(1, 13)) if aggregate is not None else available_for_year)
    preferred = int(latest[5:]) if year == int(latest[:4]) else choices[0]
    month = st.sidebar.selectbox(
        "Mês", choices, index=choices.index(preferred) if preferred in choices else 0
    )
    region_label = st.sidebar.radio("Área do mapa", ["Pacífico equatorial", "Global"])
    region = "pacific" if region_label == "Pacífico equatorial" else "global"
    selected = f"{year:04d}-{month:02d}"
    _indicators(latest, months)
    map_tab, history_tab, compare_tab, science_tab = st.tabs(
        ["Mapa SST", "Histórico", "Comparador", "Ciência e fontes"]
    )
    defaults = _period_defaults(months, latest)
    with map_tab:
        _map_panel(selected, region)
    with history_tab:
        _history_panel(defaults[2], defaults[3])
    with compare_tab:
        _compare_panel(defaults, region)
    with science_tab:
        _science_panel()


if __name__ == "__main__":
    main()

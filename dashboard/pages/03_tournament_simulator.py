"""Tournament Simulator — Monte Carlo WC 2026.

Funcionalidades:
- Tabla de favoritos con probabilidades por ronda
- Comparación contra cuotas outright (método Shin)
- Alertas EV+ para mercado de campeón

Usa el bracket oficial FIFA (partidos 73-104) con asignación de mejores
terceros por matching sobre los slots permitidos.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import PROCESSED_DIR

st.set_page_config(page_title="Tournament Simulator", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Simulator — Mundial FIFA 2026")

# ---- Carga de resultados (o ejecución en demanda) ----------------------

@st.cache_data(ttl=3600, show_spinner="Ejecutando simulacion Monte Carlo...")
def get_sim_results(n_sims: int = 10_000, seed: int = 42,
                    conditioned: bool = True) -> pd.DataFrame:
    import joblib
    from src.simulation.monte_carlo import load_tournament_state, run_monte_carlo
    predictor_path = PROCESSED_DIR / "calibrator.joblib"
    if not predictor_path.exists():
        predictor_path = PROCESSED_DIR / "ensemble.joblib"
    predictor = joblib.load(predictor_path)
    state = load_tournament_state() if conditioned else None
    return run_monte_carlo(predictor, n=n_sims, seed=seed, state=state)


@st.cache_data(ttl=900)
def get_state_counts() -> tuple[int, int]:
    from src.simulation.monte_carlo import load_tournament_state
    s = load_tournament_state()
    return s["n_group_played"], s["n_ko_played"]


@st.cache_data
def get_groups() -> dict:
    import json
    from config.settings import WC2026_FIXTURES
    return json.loads(WC2026_FIXTURES.read_text(encoding="utf-8"))["groups"]


# ---- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("Configuracion")
    n_sims = st.select_slider("Simulaciones",
                               options=[1_000, 5_000, 10_000, 25_000, 50_000],
                               value=10_000)
    seed = st.number_input("Semilla aleatoria", value=42, step=1)
    conditioned = st.toggle("Condicionar a resultados reales", value=True,
                            help="Los partidos ya jugados del torneo quedan "
                                 "fijos en todas las simulaciones")
    n_grp, n_ko = get_state_counts()
    if conditioned and (n_grp or n_ko):
        st.caption(f"📌 {n_grp} partidos de grupos + {n_ko} KO reales en la BD")
    elif conditioned:
        st.caption("Sin resultados reales en la BD todavía")
    run_btn = st.button("Ejecutar simulacion", type="primary")
    st.divider()
    st.subheader("Cuotas outright (campeón)")
    st.caption("Pega las cuotas decimales de tu casa de apuestas (una por equipo)")
    odds_input = st.text_area("Equipo: cuota  (ej. Brazil: 7.50)", height=200,
                               placeholder="Brazil: 7.50\nArgentina: 6.00\nFrance: 5.50\n...")
    st.divider()
    st.caption("Bracket oficial FIFA (R32 partidos 73-88). Los terceros se "
               "asignan a sus slots oficiales en cada simulación.")

# ---- Ejecutar simulación -----------------------------------------------
parquet = PROCESSED_DIR / "sim_results.parquet"
if run_btn:
    get_sim_results.clear()
    df = get_sim_results(n_sims, seed, conditioned)
elif parquet.exists() and conditioned:
    # el parquet lo genera refresh_all / run_simulator ya condicionado
    df = pd.read_parquet(parquet)
else:
    df = get_sim_results(n_sims, seed, conditioned)

groups = get_groups()
team_to_group = {t: g for g, teams in groups.items() for t in teams}
df["group"] = df["team"].map(team_to_group)

# ---- Parsear cuotas outright -------------------------------------------
def parse_odds_input(text: str) -> dict:
    odds = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        parts = line.split(":", 1)
        try:
            odds[parts[0].strip()] = float(parts[1].strip())
        except ValueError:
            pass
    return odds


odds_dict = parse_odds_input(odds_input) if odds_input else {}

# ---- Calcular EV outright -----------------------------------------------
if odds_dict:
    from src.betting.odds_parser import remove_vig_shin
    from src.betting.ev_calculator import classify_ev, expected_value

    teams_with_odds = list(odds_dict.keys())
    raw_odds = [odds_dict[t] for t in teams_with_odds]
    try:
        implied = remove_vig_shin(raw_odds)
        ev_map = {}
        for t, imp, o in zip(teams_with_odds, implied, raw_odds):
            row = df[df["team"] == t]
            if not row.empty:
                model_p = row.iloc[0]["champion_pct"] / 100.0
                ev = expected_value(model_p, o)
                ev_map[t] = {"ev": ev, "implied": imp, "signal": classify_ev(ev, outright=True)}
    except Exception:
        ev_map = {}
else:
    ev_map = {}

# ---- Tabla principal ---------------------------------------------------
st.subheader("Probabilidades por equipo")

view = st.radio("Vista", ["Acumulada (llega al menos a la ronda)",
                          "Exclusiva (ronda de salida — cada fila suma 100%)"],
                horizontal=True, label_visibility="collapsed")

if view.startswith("Acumulada"):
    cols_show = ["rank", "team", "group", "champion_pct", "finalist_pct",
                 "sf_pct", "qf_pct", "r32_pct", "group_exit_pct"]
    col_labels = {
        "rank": "#", "team": "Equipo", "group": "Grupo",
        "champion_pct": "Campeon%", "finalist_pct": "Finalista%",
        "sf_pct": "Semis%", "qf_pct": "Cuartos%",
        "r32_pct": "R32%", "group_exit_pct": "Sale en grupo%"
    }
    df_display = df[cols_show].rename(columns=col_labels).copy()
    pct_cols = ["Campeon%", "Finalista%", "Semis%", "Cuartos%",
                "R32%", "Sale en grupo%"]
    legend = ("**Vista acumulada**: cada celda es P(llegar *al menos* a esa ronda). "
              "Son probabilidades anidadas (campeón ⊂ finalista ⊂ semis...), por eso "
              "una fila NO suma 100. Suma por columna = equipos que alcanzan la ronda "
              "× 100 (campeón=100, finalista=200, semis=400...). "
              "El complemento por fila sí cierra: R32% + Sale en grupo% = 100.")
else:
    # Vista exclusiva: P(ronda exacta de salida) — diferencias de acumuladas
    df_ex = df[["rank", "team", "group"]].copy()
    df_ex["Campeon"] = df["champion_pct"]
    df_ex["Subcampeon"] = df["finalist_pct"] - df["champion_pct"]
    df_ex["Cae en semis"] = df["sf_pct"] - df["finalist_pct"]
    df_ex["Cae en cuartos"] = df["qf_pct"] - df["sf_pct"]
    df_ex["Cae en octavos"] = df["r16_pct"] - df["qf_pct"]
    df_ex["Cae en R32"] = df["r32_pct"] - df["r16_pct"]
    df_ex["Cae en grupos"] = df["group_exit_pct"]
    df_display = df_ex.rename(columns={"rank": "#", "team": "Equipo",
                                       "group": "Grupo"})
    pct_cols = ["Campeon", "Subcampeon", "Cae en semis", "Cae en cuartos",
                "Cae en octavos", "Cae en R32", "Cae en grupos"]
    legend = ("**Vista exclusiva**: cada celda es P(que el torneo del equipo termine "
              "exactamente en esa ronda). Cada fila suma 100%.")

if ev_map:
    df_display["EV campeon"] = df_display["Equipo"].map(
        lambda t: f"{ev_map[t]['ev']:+.1%} [{ev_map[t]['signal']}]" if t in ev_map else "-")

# Color coding
def color_cell(val):
    if isinstance(val, str) and "HIGH" in val:
        return "background-color: #d4edda; color: #155724"
    return ""

st.dataframe(
    df_display.style.format({c: "{:.1f}%" for c in pct_cols})
        .map(color_cell, subset=["EV campeon"] if "EV campeon" in df_display.columns else []),
    hide_index=True, use_container_width=True, height=600
)
st.caption(legend)

# ---- Gráficos ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Favoritos al titulo", "Distribucion por ronda", "EV Outrights"])

with tab1:
    top15 = df.nlargest(15, "champion_pct")
    fig = px.bar(top15, x="champion_pct", y="team", orientation="h",
                 color="champion_pct", color_continuous_scale="Blues",
                 labels={"champion_pct": "P(campeon) %", "team": ""},
                 text="champion_pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    top10 = df.head(10)
    rounds = ["champion_pct", "finalist_pct", "sf_pct", "qf_pct", "r32_pct"]
    round_labels = ["Campeon", "Finalista", "Semis", "Cuartos", "R32"]

    fig2 = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, team in enumerate(top10["team"]):
        vals = [top10[top10["team"] == team][r].values[0] for r in rounds]
        fig2.add_trace(go.Scatter(x=round_labels, y=vals, mode="lines+markers",
                                  name=team, line=dict(color=colors[i % len(colors)])))
    fig2.update_layout(title="Probabilidad acumulada de avanzar (Top 10)",
                       yaxis_title="%", height=450, legend_title="Equipo")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if not ev_map:
        st.info("Pega cuotas outright en el panel lateral para ver oportunidades EV+.")
    else:
        rows_ev = []
        for t, data in ev_map.items():
            row = df[df["team"] == t]
            if not row.empty:
                rows_ev.append({
                    "Equipo": t,
                    "Prob. modelo": f"{row.iloc[0]['champion_pct']:.1f}%",
                    "Prob. implícita (Shin)": f"{data['implied']:.1%}",
                    "Cuota": odds_dict.get(t, "-"),
                    "EV": f"{data['ev']:+.1%}",
                    "Señal": data["signal"],
                })
        df_ev = pd.DataFrame(rows_ev).sort_values("EV", ascending=False)
        st.dataframe(
            df_ev.style.map(
                lambda v: "background-color: #d4edda" if v == "HIGH" else
                          "background-color: #fff3cd" if v == "MEDIUM" else "",
                subset=["Señal"]),
            hide_index=True, use_container_width=True)
        st.caption("Umbral EV+ outright: >15% (margen extra por incertidumbre acumulada en 7 rondas). "
                   "Metodo de remocion de vig: Shin (corrige sesgo longshot).")

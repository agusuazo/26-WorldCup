"""Vista de Bracket — un torneo simulado completo: grupos, llaves y caminos.

Dos modos:
- "Escenario más probable": determinista — standings por puntos esperados,
  en eliminatorias avanza el favorito de cada cruce.
- "Simulación aleatoria": una iteración Monte Carlo; re-simular genera otro
  escenario posible.

El escenario más probable es UNA trayectoria ilustrativa (el camino modal),
no la predicción agregada: para probabilidades reales por equipo y ronda
está la página Tournament Simulator.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vista de Bracket", page_icon="🗂️", layout="wide")
st.title("🗂️ Vista de Bracket — WC 2026 simulado")

# ---- Simulación cacheada ------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Simulando torneo...")
def get_detail(mode: str, seed: int, conditioned: bool) -> dict:
    from dashboard.components.model_store import load_best_predictor
    from src.simulation.monte_carlo import (load_tournament_state,
                                            simulate_tournament_detail)
    predictor = load_best_predictor()
    state = load_tournament_state() if conditioned else None
    return simulate_tournament_detail(predictor, seed=seed, mode=mode,
                                      state=state)


# ---- Sidebar -------------------------------------------------------------
if "bracket_seed" not in st.session_state:
    st.session_state.bracket_seed = 42

with st.sidebar:
    st.header("Configuracion")
    mode_label = st.radio("Modo", ["Escenario más probable", "Simulación aleatoria"])
    mode = "expected" if mode_label == "Escenario más probable" else "sample"
    if mode == "sample":
        if st.button("🎲 Re-simular", type="primary"):
            st.session_state.bracket_seed += 1
        st.caption(f"Seed actual: {st.session_state.bracket_seed}")
    conditioned = st.toggle("Condicionar a resultados reales", value=True,
                            help="Los partidos ya jugados del torneo se "
                                 "muestran con su resultado real (📌)")
    st.divider()

detail = get_detail(mode, st.session_state.bracket_seed, conditioned)
played_pairs = detail.get("played_pairs", set())

all_wc_teams = sorted({t for g in detail["groups"].values()
                       for t, *_ in g["standings"]})
with st.sidebar:
    highlight = st.selectbox("Resaltar equipo", ["(ninguno)"] + all_wc_teams)
    if highlight == "(ninguno)":
        highlight = None

if mode == "expected":
    st.info("**Escenario más probable**: en cada cruce avanza el favorito del modelo. "
            "Es una trayectoria ilustrativa, no la predicción agregada — "
            "las probabilidades reales por ronda están en Tournament Simulator.")

# ---- Fase de grupos -------------------------------------------------------
st.subheader("Fase de grupos — clasificación final")
is_expected = detail["mode"] == "expected"
fmt_pts = "{:.1f}" if is_expected else "{:.0f}"

group_names = sorted(detail["groups"].keys())
for row_start in range(0, 12, 4):
    cols = st.columns(4)
    for col, g_name in zip(cols, group_names[row_start:row_start + 4]):
        g = detail["groups"][g_name]
        q = g["qualified"]
        with col:
            n_real = sum(1 for h, a, *_ in g["results"]
                         if frozenset((h, a)) in played_pairs)
            pin = f" 📌{n_real}" if n_real else ""
            st.markdown(f"**Grupo {g_name}**{pin}")
            lines = []
            for pos, (team, pts, gd, gf) in enumerate(g["standings"], 1):
                if team == q["first"] or team == q["second"]:
                    mark = "✅"
                elif team == q["third"]:
                    mark = "3️⃣"
                else:
                    mark = "❌"
                style = "**" if team == highlight else ""
                lines.append(f"{pos}. {mark} {style}{team}{style} "
                             f"({fmt_pts.format(pts)} pts, {gd:+.1f})")
            st.markdown("\n".join(f"- {l}" for l in lines))

st.caption("✅ clasifica directo (1° y 2°) · 3️⃣ clasifica entre los 8 mejores terceros · "
           "❌ eliminado" + (" · puntos esperados (no enteros)" if is_expected else ""))

st.divider()

# ---- Llaves ---------------------------------------------------------------
st.subheader("Llaves de eliminación directa")

ROUND_LABELS = [("r32", "Ronda de 32"), ("r16", "Octavos"), ("qf", "Cuartos"),
                ("sf", "Semifinal"), ("final", "Final")]


def match_html(a: str, b: str, w: str, highlight: str | None,
               is_real: bool = False) -> str:
    """Bloque HTML de un cruce; ganador en negrita, resaltado si participa,
    📌 si es un resultado real (no simulado)."""
    involved = highlight in (a, b) if highlight else False
    bg = "#fff3cd" if involved else "#f8f9fa"
    border = "#e0a800" if involved else "#dee2e6"
    pin = ("<div style='font-size:0.68rem;color:#888'>📌 resultado real</div>"
           if is_real else "")

    def line(t):
        is_w = t == w
        weight = "700" if is_w else "400"
        color = "#1a7f37" if is_w else "#555"
        icon = " ✓" if is_w else ""
        return (f"<div style='font-weight:{weight};color:{color};"
                f"font-size:0.82rem;line-height:1.5'>{t}{icon}</div>")

    return (f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:6px;padding:6px 8px;margin-bottom:8px'>"
            f"{line(a)}{line(b)}{pin}</div>")


cols = st.columns([3, 3, 3, 3, 3, 2])
for col, (rname, rlabel) in zip(cols, ROUND_LABELS):
    with col:
        st.markdown(f"**{rlabel}**")
        html = "".join(
            match_html(a, b, w, highlight,
                       is_real=frozenset((a, b)) in played_pairs)
            for a, b, w in detail["rounds"][rname])
        st.markdown(html, unsafe_allow_html=True)

with cols[5]:
    st.markdown("**🏆 Campeón**")
    st.markdown(
        f"<div style='background:#d4edda;border:2px solid #1a7f37;"
        f"border-radius:8px;padding:14px 10px;text-align:center;"
        f"font-weight:700;font-size:1.05rem;color:#155724'>"
        f"{detail['champion']}</div>", unsafe_allow_html=True)

# ---- Camino del equipo resaltado -------------------------------------------
if highlight:
    st.divider()
    st.subheader(f"Camino de {highlight}")

    # Posición en su grupo
    path_parts = []
    for g_name, g in detail["groups"].items():
        teams_in_group = [t for t, *_ in g["standings"]]
        if highlight in teams_in_group:
            pos = teams_in_group.index(highlight) + 1
            q = g["qualified"]
            if highlight in (q["first"], q["second"]) or highlight == q["third"]:
                suffix = "" if pos <= 2 else " (mejor tercero)"
                path_parts.append(f"{pos}° del Grupo {g_name}{suffix}")
            else:
                path_parts.append(f"{pos}° del Grupo {g_name} — eliminado en grupos")
            break

    eliminated = False
    round_es = {"r32": "R32", "r16": "octavos", "qf": "cuartos",
                "sf": "semifinal", "final": "la final"}
    for rname, _label in ROUND_LABELS:
        for a, b, w in detail["rounds"][rname]:
            if highlight in (a, b):
                rival = b if a == highlight else a
                if w == highlight:
                    path_parts.append(f"vence a {rival} en {round_es[rname]}")
                else:
                    path_parts.append(f"cae ante {rival} en {round_es[rname]}")
                    eliminated = True
        if eliminated:
            break

    if detail["champion"] == highlight:
        path_parts.append("🏆 **CAMPEÓN DEL MUNDO**")

    st.markdown(" → ".join(path_parts))

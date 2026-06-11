"""Dashboard principal — Sistema Predictivo Mundial FIFA 2026.

Ejecutar desde la raíz del proyecto: streamlit run app/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components.model_store import load_elo_table, load_wc2026

st.set_page_config(page_title="Mundial FIFA 2026 — EV+", page_icon="⚽",
                   layout="wide")

st.title("⚽ Sistema Predictivo — Mundial FIFA 2026")
st.caption("Probabilidades propias vs. cuotas de mercado. Detección de EV+ con Kelly fraccional.")

wc = load_wc2026()
elo = load_elo_table()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ranking Elo — selecciones mundialistas")
    teams_wc = {t for g in wc["groups"].values() for t in g}
    elo_wc = elo[elo["team"].isin(teams_wc)].reset_index(drop=True)
    elo_wc.index += 1
    st.dataframe(elo_wc.style.format({"elo": "{:.0f}"}), height=500)

with col2:
    st.subheader("Grupos oficiales")
    for letter, teams in wc["groups"].items():
        st.markdown(f"**Grupo {letter}:** {', '.join(teams)}")

st.info("👈 Páginas: **home** (próximos partidos + EV+), **match predictor**, "
        "**tournament simulator** (bracket oficial FIFA), **backtesting** y "
        "**bankroll tracker**.")
st.warning("⚠️ Protocolo: paper trading la primera semana. Gate Brier < 0.20 "
           "superado (hold-out 0.1652 · walk-forward trimestral 0.1713). "
           "Dinero real solo con 1/8 Kelly y sin EV+ sistemático en longshots.")

"""Dashboard principal — Sistema Predictivo Mundial FIFA 2026.

Ejecutar desde la raíz del proyecto: streamlit run dashboard/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.components.model_store import load_elo_table, load_wc2026
from dashboard.components.styles import inject_global_css, info_box

st.set_page_config(page_title="Mundial FIFA 2026 — Predicciones", page_icon="⚽",
                   layout="wide")
inject_global_css()

st.title("⚽ Predicciones — Mundial FIFA 2026")
st.caption("Probabilidades propias basadas en modelos estadísticos · comparación contra cuotas de mercado")

with st.expander("ℹ️ ¿Qué es este sitio?", expanded=False):
    info_box("""
    <b>Sistema de predicción estadística del Mundial FIFA 2026.</b><br><br>
    Usamos tres modelos matemáticos combinados para estimar qué tan probable es que cada equipo
    gane, empate o pierda cada partido, y quién tiene más probabilidades de ser campeón del mundo.
    <ul>
      <li><span class="glossary-term">Elo:</span> Sistema de puntuación que actualiza el "nivel" de cada selección tras cada partido.
          Un equipo sube puntos al ganar y los pierde al perder, ponderado por la importancia del torneo.</li>
      <li><span class="glossary-term">Poisson:</span> Modelo que estima cuántos goles va a meter cada equipo según su ataque y la defensa rival.</li>
      <li><span class="glossary-term">Dixon-Coles:</span> Versión mejorada del Poisson que aprende el estilo ofensivo/defensivo de cada selección.</li>
      <li><span class="glossary-term">Brier Score:</span> Nota de calidad del modelo (0 = perfecto, 0.222 = adivinar al azar). Cuanto más bajo, mejor.</li>
      <li><span class="glossary-term">Monte Carlo:</span> Técnica que simula el torneo completo 10.000 veces y cuenta cuántas veces gana cada equipo.</li>
    </ul>
    Usá el menú lateral para explorar predicciones de partidos, el simulador del torneo, el bracket y más.
    """)

wc = load_wc2026()
elo = load_elo_table()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ranking Elo — selecciones mundialistas")
    st.caption("Puntuación de nivel de cada equipo. Mayor Elo = mayor historial de victorias en torneos importantes.")
    teams_wc = {t for g in wc["groups"].values() for t in g}
    elo_wc = elo[elo["team"].isin(teams_wc)].reset_index(drop=True)
    elo_wc.index += 1
    st.dataframe(elo_wc.style.format({"elo": "{:.0f}"}), height=500,
                 use_container_width=True)

with col2:
    st.subheader("Grupos oficiales")
    st.caption("Los 48 equipos divididos en 12 grupos de 4. Clasifican los 2 primeros de cada grupo más los 8 mejores terceros.")
    for letter, teams in wc["groups"].items():
        st.markdown(f"**Grupo {letter}:** {', '.join(teams)}")

st.info("👈 Navegá por las páginas del menú lateral: predictor de partidos, simulador del torneo, bracket y calidad del modelo.")

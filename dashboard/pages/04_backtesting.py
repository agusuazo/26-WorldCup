"""Backtesting — calidad probabilística y simulación de apuestas.

Sprint 1: backtest de Brier/LogLoss sobre hold-out 2023-2025.
Sprint 2: backtesting con cuotas históricas (requiere dataset de cuotas).
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
from dashboard.components.styles import inject_global_css, info_box

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")
inject_global_css()
st.title("📊 Calidad del Modelo")

with st.expander("ℹ️ Cómo leer esta página", expanded=False):
    info_box("""
    <b>¿Qué tan bueno es el modelo?</b> Esta página muestra métricas de calidad estadística evaluadas
    sobre miles de partidos históricos que el modelo <b>nunca vio durante el entrenamiento</b>.<br><br>
    <ul>
      <li><span class="glossary-term">Brier Score:</span>
          Mide la precisión de las probabilidades predichas. Escala de 0 (perfecto) a 1 (peor posible).
          <b>0.222</b> es la nota de "adivinar al azar" (decir siempre 33%/33%/33%).
          Nuestro modelo está en <b>~0.163</b>, una mejora significativa sobre el azar.</li>
      <li><span class="glossary-term">Baseline uniforme (0.222):</span>
          Referencia: si siempre predijeras exactamente 1/3 de probabilidad para cada resultado,
          obtendrías Brier = 0.222. Cualquier modelo útil debe estar por debajo de ese valor.</li>
      <li><span class="glossary-term">Log Loss:</span>
          Otra métrica de calidad, más sensible a errores con alta confianza. Menor = mejor.</li>
      <li><span class="glossary-term">Curva de calibración:</span>
          Verifica que las probabilidades sean "reales". Si el modelo dice 70% de victoria local,
          ¿realmente ganó el local el 70% de las veces? Una curva pegada a la diagonal diagonal = modelo bien calibrado.</li>
      <li><span class="glossary-term">Hold-out:</span>
          Conjunto de partidos (2023–2025) que se guardaron <b>sin usar</b> para entrenar el modelo.
          Evaluar sobre hold-out garantiza que no estamos midiendo memoria sino predicción real.</li>
    </ul>
    """)

# ---- Carga de resultados precomputados ---------------------------------

@st.cache_data(ttl=1800)
def load_backtest_results():
    summary_p = PROCESSED_DIR / "backtest_results.parquet"
    by_year_p = PROCESSED_DIR / "backtest_by_year.parquet"
    cal_p = PROCESSED_DIR / "calibration_curve.parquet"
    out = {}
    if summary_p.exists():
        out["summary"] = pd.read_parquet(summary_p)
    if by_year_p.exists():
        out["by_year"] = pd.read_parquet(by_year_p)
    if cal_p.exists():
        out["calibration"] = pd.read_parquet(cal_p)
    return out


@st.cache_data(ttl=1800, show_spinner="Ejecutando backtest probabilistico...")
def run_backtest_live(start: str, end: str):
    import duckdb
    import joblib
    from config.settings import DB_PATH, ELO_WARMUP_END
    from src.backtesting.engine import backtest_probability_quality

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT m.match_id, m.date, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.tournament, m.neutral
            FROM matches m
            WHERE m.home_score IS NOT NULL AND m.date >= ?
            ORDER BY m.date
        """, [ELO_WARMUP_END]).df()
    finally:
        con.close()

    cal_path = PROCESSED_DIR / "calibrator.joblib"
    ens_path = PROCESSED_DIR / "ensemble.joblib"
    if cal_path.exists():
        predictor = joblib.load(cal_path)
    elif ens_path.exists():
        predictor = joblib.load(ens_path)
    else:
        return None

    return backtest_probability_quality(predictor, df, start=start, end=end)


# ---- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("Parametros")
    start_date = st.date_input("Desde", value=pd.Timestamp("2023-01-01"))
    end_date = st.date_input("Hasta", value=pd.Timestamp("2025-12-31"))
    run_live = st.button("Recalcular backtest", type="primary")
    st.divider()
    st.caption("Backtest de calidad probabilística: evalúa Brier Score y Log Loss "
               "en partidos pasados sin usar cuotas históricas.")

# ---- Cargar / ejecutar -------------------------------------------------
data = load_backtest_results()

if run_live:
    load_backtest_results.clear()
    run_backtest_live.clear()
    results = run_backtest_live(str(start_date), str(end_date))
    if results and "error" not in results:
        data["by_year"] = results["by_year"]
        data["calibration"] = results["calibration"]
        data["summary"] = pd.DataFrame([{
            "start": str(start_date), "end": str(end_date),
            "n_matches": results["n_matches"],
            "brier_score": results["brier_score"],
            "log_loss": results["log_loss"],
        }])
    else:
        st.error("Backtest fallido. Asegúrate de haber entrenado los modelos con run_training.py")

# ---- KPIs globales ----------------------------------------------------
if "summary" in data and not data["summary"].empty:
    row = data["summary"].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    brier = row["brier_score"]
    baseline = 0.2222
    col1.metric("Brier Score", f"{brier:.4f}",
                delta=f"{brier - baseline:+.4f} vs baseline",
                delta_color="inverse")
    col2.metric("Log Loss", f"{row['log_loss']:.4f}")
    col3.metric("Partidos evaluados", f"{int(row['n_matches']):,}")
    col4.metric("Periodo",
                f"{row.get('start', '?')} → {row.get('end', '?')}")

    if brier < 0.220:
        st.success(f"✅ Brier {brier:.4f} — por debajo del umbral de calidad (0.220). "
                   "El modelo supera significativamente la predicción al azar.")
    elif brier < 0.235:
        st.warning(f"⚠️ Brier {brier:.4f} — dentro del rango aceptable pero mejorable. "
                   "El modelo predice mejor que el azar pero con margen ajustado.")
    else:
        st.error(f"❌ Brier {brier:.4f} — por encima del umbral de calidad (0.235). "
                 "El modelo necesita mejoras antes de ser confiable para predicciones.")
else:
    st.info("No hay resultados de backtest. Ejecuta `python scripts/run_backtest.py` "
            "o pulsa 'Recalcular' en el panel lateral.")

st.divider()

# ---- Tabs de análisis -------------------------------------------------
tab_year, tab_cal, tab_about = st.tabs(["Brier por año", "Curva de calibración", "Interpretacion"])

with tab_year:
    if "by_year" in data and not data["by_year"].empty:
        df_yr = data["by_year"]
        fig = px.bar(df_yr, x="year", y="brier",
                     labels={"brier": "Brier Score", "year": "Año"},
                     color="brier", color_continuous_scale="RdYlGn_r",
                     text="brier")
        fig.add_hline(y=0.2222, line_dash="dash", line_color="gray",
                      annotation_text="Baseline uniforme (0.2222)")
        fig.add_hline(y=0.220, line_dash="dot", line_color="blue",
                      annotation_text="Umbral calidad (0.220)")
        fig.update_traces(texttemplate="%{text:.4f}")
        fig.update_layout(height=400, showlegend=False,
                          yaxis_range=[0.18, 0.26])
        st.plotly_chart(fig, use_container_width=True)

        for _, row in df_yr.iterrows():
            improvement = 0.2222 - row["brier"]
            st.caption(f"{int(row['year'])}: Brier={row['brier']:.4f} "
                       f"(mejora vs baseline: {improvement:+.4f})  n={int(row['n'])}")
    else:
        st.info("Sin datos de Brier por año.")

with tab_cal:
    if "calibration" in data and not data["calibration"].empty:
        df_cal = data["calibration"]
        fig2 = go.Figure()
        # Diagonal perfecta
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"),
                                  name="Perfectamente calibrado"))
        # Curva real
        fig2.add_trace(go.Scatter(x=df_cal["mean_pred"], y=df_cal["actual_freq"],
                                  mode="lines+markers",
                                  marker=dict(size=df_cal["count"].clip(upper=500) / 50 + 4),
                                  name="Modelo",
                                  hovertemplate="Pred: %{x:.2f}<br>Real: %{y:.2f}<br>n=%{customdata}",
                                  customdata=df_cal["count"]))
        fig2.update_layout(
            title="Curva de calibración — probabilidad de victoria local",
            xaxis_title="Probabilidad predicha", yaxis_title="Frecuencia real",
            height=450, xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("El tamaño de los marcadores es proporcional al número de muestras en cada bin. "
                   "Una curva pegada a la diagonal indica buena calibración.")
    else:
        st.info("Sin datos de calibración.")

with tab_about:
    st.markdown("""
    ### Interpretación del Brier Score

    El **Brier Score** mide qué tan precisas son las probabilidades predichas:
    - **0.0**: predicciones perfectas (imposible en la práctica)
    - **0.163**: nuestro modelo actual (hold-out 2023-2025)
    - **0.215 – 0.220**: rango de modelos estadísticos bien calibrados
    - **0.222**: baseline uniforme — adivinar siempre 1/3 para cada resultado
    - **0.235**: modelo Elo básico típico
    - **1.0**: peor caso — 100% de confianza siempre equivocada

    **Fórmula:** `Brier = media(suma((p_i − y_i)²)) / 3`
    donde `p_i` es la probabilidad predicha e `y_i` es el resultado real (1 o 0).

    ### Curva de calibración

    Un modelo bien calibrado tiene su curva pegada a la diagonal.
    Si predecimos 60%, queremos que el local gane realmente el 60% de esas veces.
    El tamaño de cada punto indica cuántas muestras hay en ese rango de probabilidades.

    ### ¿Por qué evaluar en hold-out?

    Para evitar "trampa": si evaluáramos en los mismos datos con los que se entrenó el modelo,
    mediríamos memoria en vez de predicción real. El hold-out (2023-2025) es un conjunto
    de partidos que el modelo nunca procesó durante el aprendizaje.
    """)

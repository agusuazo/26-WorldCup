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

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")
st.title("📊 Backtesting — Calidad del Modelo")

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
        st.success(f"Brier {brier:.4f} — por debajo del umbral de calidad (0.220). "
                   "El modelo es apto para uso real.")
    elif brier < 0.235:
        st.warning(f"Brier {brier:.4f} — dentro del rango aceptable pero mejora posible. "
                   "Usar con Kelly conservador (1/8 fracción).")
    else:
        st.error(f"Brier {brier:.4f} — por encima del umbral de seguridad (0.235). "
                 "No apostar dinero real hasta mejorar el modelo.")
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
    ### Interpretación del Brier Score multiclase

    El **Brier Score** mide la precisión probabilística:
    - **0.0**: predicciones perfectas (imposible en la práctica)
    - **0.2222**: baseline uniforme (1/3, 1/3, 1/3 para todos los partidos)
    - **0.235**: modelo Elo básico típico
    - **0.215 - 0.220**: rango de modelos bien calibrados con Dixon-Coles + ensemble
    - **1.0**: worst case (100% de confianza siempre errónea)

    **Fórmula:** `Brier = mean(sum((p_i - y_i)^2)) / 3`

    ### Protocolo paper trading

    La primera semana del torneo opera con `paper=TRUE` en el bet_log.
    Requisitos para pasar a dinero real:
    1. Brier hold-out < **0.220**
    2. Sin EV+ sistemático en longshots (>7.0) en paper trading
    3. Máximo **1/8 Kelly** las primeras 2 semanas

    ### Backtest con cuotas (Sprint 2)

    Actualmente solo se evalúa calidad probabilística (Brier/LogLoss).
    El backtest con cuotas reales (ROI, drawdown, Sharpe) requiere un
    dataset de cuotas históricas de torneos internacionales —
    disponible en el script `run_backtest.py` con `--mode bets` una vez
    que se haya integrado el dataset de odds.
    """)

"""Home — próximos partidos, top EV+ del día y favoritos al título."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.model_store import (load_best_predictor,
                                              load_upcoming_wc_matches)
from dashboard.components.styles import inject_global_css, info_box
from config.settings import PROCESSED_DIR
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import remove_vig_multiplicative

st.set_page_config(page_title="Home — Mundial 2026", page_icon="🏠", layout="wide")
inject_global_css()
st.title("🏠 Home — Mundial FIFA 2026")

with st.expander("ℹ️ Cómo leer esta página", expanded=False):
    info_box("""
    <b>Panel de control del Mundial.</b> Acá encontrás las predicciones del modelo para los próximos partidos
    y quiénes son los favoritos a ganar el campeonato.<br><br>
    <ul>
      <li><span class="glossary-term">P(local) / P(empate) / P(visita):</span>
          Probabilidad (en %) que el modelo le asigna a cada resultado posible.
          Por ejemplo, P(local) = 55% significa que en 100 partidos similares, el local ganaría ~55 veces.</li>
      <li><span class="glossary-term">Mejor EV:</span>
          Valor Esperado — mide si una cuota de apuesta ofrece ganancia a largo plazo.
          EV positivo = la cuota paga más de lo que el riesgo justificaría según el modelo.
          <b>Requiere snapshot de cuotas actualizado.</b></li>
      <li><span class="glossary-term">Señal HIGH / MEDIUM:</span>
          Clasificación del EV. HIGH (>10%) = valor alto. MEDIUM (5–10%) = valor moderado. Sin señal = sin ventaja detectada.</li>
      <li><span class="glossary-term">Favoritos al título:</span>
          Porcentaje de veces que cada equipo ganó el torneo en 10.000 simulaciones completas.</li>
    </ul>
    """)

predictor = load_best_predictor()
upcoming = load_upcoming_wc_matches(limit=20)

try:
    from src.ingestion.updater import get_last_refresh
    _last = get_last_refresh()
    if _last:
        st.caption(f"🔄 Último recálculo: "
                   f"{pd.to_datetime(_last['refreshed_at']):%d-%b %H:%M} UTC · "
                   f"Brier {_last['brier_holdout']:.4f} · condicionado a "
                   f"{_last['n_group_played'] + _last['n_ko_played']} partidos reales. "
                   "Actualizar en la página 'Actualizar Datos'.")
except Exception:
    pass

# ---- Cuotas en vivo (si existen) ----------------------------------------
odds_df = None
try:
    from src.ingestion.odds_api import best_odds_per_match, load_latest_odds
    raw_odds = load_latest_odds()
    if raw_odds is not None and not raw_odds.empty:
        odds_df = best_odds_per_match(raw_odds)
        fetched = pd.to_datetime(raw_odds["fetched_at"].iloc[0])
        st.caption(f"Cuotas: snapshot de {fetched:%d-%b %H:%M} UTC "
                   f"({raw_odds['bookmaker'].nunique()} casas de apuesta). "
                   "Actualizar: `python scripts/fetch_odds.py`")
except Exception:
    pass

if odds_df is None:
    st.info("Sin snapshot de cuotas activas. Las columnas EV y Señal aparecerán "
            "cuando se cargue un snapshot con `python scripts/fetch_odds.py`.")

# ---- Próximos partidos con predicciones ----------------------------------
st.subheader("Próximos partidos — predicciones del modelo")

if upcoming.empty:
    st.warning("No hay partidos futuros del Mundial en la base. "
               "¿Falta re-ejecutar la ingesta en Actualizar Datos?")
else:
    rows = []
    for m in upcoming.itertuples():
        try:
            ph, pd_, pa = predictor.predict_proba(m.home_team, m.away_team, True)
        except Exception:
            continue
        row = {
            "Fecha": pd.to_datetime(m.date).strftime("%d-%b"),
            "Partido": f"{m.home_team} vs {m.away_team}",
            "P(local)": ph, "P(empate)": pd_, "P(visita)": pa,
        }
        if odds_df is not None:
            match_odds = odds_df[
                (odds_df["home_team"] == m.home_team) &
                (odds_df["away_team"] == m.away_team)]
            if not match_odds.empty:
                mo = match_odds.iloc[0]
                best_ev, best_label = -1.0, ""
                for p, o, label in [(ph, mo["home_odds"], m.home_team),
                                    (pd_, mo["draw_odds"], "Empate"),
                                    (pa, mo["away_odds"], m.away_team)]:
                    if o and o > 1:
                        ev = expected_value(p, o)
                        if ev > best_ev:
                            best_ev, best_label = ev, f"{label} @{o:.2f}"
                row["Mejor EV"] = best_ev
                row["Apuesta sugerida"] = best_label
                row["Señal"] = classify_ev(best_ev)
        rows.append(row)

    df_pred = pd.DataFrame(rows)
    fmt = {"P(local)": "{:.0%}", "P(empate)": "{:.0%}", "P(visita)": "{:.0%}"}
    if "Mejor EV" in df_pred.columns:
        fmt["Mejor EV"] = "{:+.1%}"
        styled = df_pred.style.format(fmt, na_rep="—").map(
            lambda v: "background-color: #d4edda" if v == "HIGH" else
                      ("background-color: #fff3cd" if v == "MEDIUM" else ""),
            subset=["Señal"])
    else:
        styled = df_pred.style.format(fmt)
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("EV = Valor Esperado. Solo visible cuando hay cuotas cargadas. "
               "Verde = señal HIGH (EV > 10%), amarillo = MEDIUM (EV 5–10%).")

st.divider()

# ---- Favoritos al título ------------------------------------------------
st.subheader("Favoritos al título (Monte Carlo — 10.000 simulaciones)")
sim_path = PROCESSED_DIR / "sim_results.parquet"
if sim_path.exists():
    sim = pd.read_parquet(sim_path).head(8)
    fig = go.Figure(go.Bar(
        x=sim["champion_pct"], y=sim["team"], orientation="h",
        marker_color="#2563eb",
        text=[f"{v:.1f}%" for v in sim["champion_pct"]],
        textposition="outside"))
    fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Probabilidad de ser campeón (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Porcentaje de torneos simulados donde cada equipo resultó campeón. "
               "Detalle completo por ronda en la página Tournament Simulator.")
else:
    st.info("Simulación no disponible. Ejecutá el recálculo en 'Actualizar Datos'.")

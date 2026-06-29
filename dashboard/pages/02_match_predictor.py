"""Match Predictor: probabilidades 1X2, comparación con cuotas y análisis de valor."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.model_store import (load_best_predictor, load_models,
                                              wc_teams)
from dashboard.components.styles import inject_global_css, info_box
from config.settings import (EV_THRESHOLD_MED, KELLY_CAP, KELLY_FRACTION)
from src.betting.bankroll import fractional_kelly, kelly_stake
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import overround, remove_vig_multiplicative

st.set_page_config(page_title="Match Predictor", page_icon="🎯", layout="wide")
inject_global_css()
st.title("🎯 Match Predictor")

with st.expander("ℹ️ Cómo leer esta página", expanded=False):
    info_box("""
    <b>Analizá cualquier partido del Mundial.</b> El modelo combina tres enfoques estadísticos
    y los compara contra las cuotas del mercado para detectar si hay valor en apostar.<br><br>
    <ul>
      <li><span class="glossary-term">Ensemble calibrado:</span>
          Combinación de Elo, Poisson y Dixon-Coles ponderados y ajustados. Es la predicción más confiable.</li>
      <li><span class="glossary-term">Goles esperados (λ):</span>
          Promedio de goles que el modelo predice para cada equipo en ese partido.</li>
      <li><span class="glossary-term">Cuota decimal:</span>
          Número que multiplica tu apuesta si ganás. Cuota 2.50 significa: apostás $1, ganás $2.50 (ganancia neta $1.50).</li>
      <li><span class="glossary-term">Probabilidad implícita:</span>
          La probabilidad que la casa de apuesta "asume" con esa cuota, descontando su margen. Si la cuota es 2.50, la prob. implícita es ~40%.</li>
      <li><span class="glossary-term">Edge (ventaja):</span>
          Diferencia entre la probabilidad del modelo y la implícita. Edge positivo = el modelo ve más probabilidad de la que paga la cuota.</li>
      <li><span class="glossary-term">EV (Valor Esperado):</span>
          EV = prob_modelo × cuota − 1. Si EV > 0, la apuesta tiene ganancia esperada a largo plazo.</li>
      <li><span class="glossary-term">Apuesta de referencia (Kelly/8):</span>
          Tamaño de apuesta sugerido por la fórmula Kelly, reducido a 1/8 para mayor seguridad. Es solo una referencia.</li>
      <li><span class="glossary-term">Overround:</span>
          Margen de ganancia de la casa de apuesta. Ej: 5% significa que las cuotas suman un 5% más que 100%.</li>
    </ul>
    """)

predictor = load_best_predictor()
elo_pred, poisson_model = load_models()
teams = wc_teams()

c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Equipo 1", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
away = c2.selectbox("Equipo 2", teams, index=teams.index("France") if "France" in teams else 1)
neutral = c3.toggle("Cancha neutral", value=True,
                    help="WC 2026: todo es neutral salvo que juegue el anfitrión en casa")

if home == away:
    st.error("Elige dos equipos distintos.")
    st.stop()

# --- Probabilidades del ensemble calibrado ---
probs = predictor.predict_proba(home, away, neutral=neutral)
lam_h, lam_a = predictor.predict_lambdas(home, away, neutral=neutral)
rh, ra = elo_pred.ratings.get(home, 1500.0), elo_pred.ratings.get(away, 1500.0)

st.subheader("Probabilidades del modelo (ensemble calibrado)")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Gana {home}", f"{probs[0]:.1%}")
m2.metric("Empate", f"{probs[1]:.1%}")
m3.metric(f"Gana {away}", f"{probs[2]:.1%}")
m4.metric("Goles esperados", f"{lam_h:.2f} – {lam_a:.2f}",
          help="Promedio de goles que el modelo predice para cada equipo")

with st.expander("📊 Detalle por modelo y matriz de marcadores"):
    st.caption("Comparación de las tres fuentes del ensemble. La fila 'Ensemble calibrado' es la predicción final.")
    p_elo = elo_pred.predict_proba(home, away, neutral=neutral)
    p_pois = poisson_model.predict_proba(rh, ra, neutral=neutral)
    st.dataframe(pd.DataFrame(
        [p_elo, p_pois, probs],
        index=["Elo + draw model", "Poisson global", "Ensemble calibrado"],
        columns=[home, "Empate", away],
    ).style.format("{:.1%}"), use_container_width=True)

    st.caption("Probabilidad de cada marcador exacto (eje X = goles visita, eje Y = goles local):")
    matrix = poisson_model.score_matrix(lam_h, lam_a, max_goals=6)
    fig = px.imshow(matrix * 100, text_auto=".1f",
                    labels=dict(x=f"Goles {away}", y=f"Goles {home}", color="%"),
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# --- Cuotas y EV ---
st.subheader("Comparación contra cuotas de mercado")
st.caption("Ingresá las cuotas decimales de tu casa de apuesta para ver si hay valor.")

o1, o2, o3 = st.columns(3)
odds_h = o1.number_input(f"Cuota {home}", min_value=1.01, value=2.50, step=0.01)
odds_d = o2.number_input("Cuota empate", min_value=1.01, value=3.20, step=0.01)
odds_a = o3.number_input(f"Cuota {away}", min_value=1.01, value=2.80, step=0.01)

odds = [odds_h, odds_d, odds_a]
implied = remove_vig_multiplicative(odds)
ovr = overround(odds)

rows = []
for i, (label, p, o) in enumerate(zip([home, "Empate", away], probs, odds)):
    ev = expected_value(p, o)
    stake_ref = kelly_stake(1000.0, p, o) if ev > 0 else 0.0
    rows.append({
        "Resultado": label,
        "Prob. modelo": p,
        "Prob. implícita (sin vig)": implied[i],
        "Edge": p - implied[i],
        "EV": ev,
        "Señal": classify_ev(ev),
        "Apuesta ref. (Kelly/8)": stake_ref,
    })
df = pd.DataFrame(rows)
st.dataframe(df.style.format({
    "Prob. modelo": "{:.1%}", "Prob. implícita (sin vig)": "{:.1%}",
    "Edge": "{:+.1%}", "EV": "{:+.1%}", "Apuesta ref. (Kelly/8)": "{:.2f}",
}).map(lambda v: "background-color: #d4edda" if v in ("HIGH", "MEDIUM")
       else ("background-color: #fff3cd" if v == "LOW" else ""),
       subset=["Señal"]), hide_index=True, use_container_width=True)
st.caption("Verde = señal HIGH (EV > 10%), amarillo = MEDIUM (5–10%), sin color = sin ventaja detectada. "
           "La columna 'Apuesta ref.' es orientativa sobre €1000 de bankroll hipotético.")

# --- Recomendación del modelo ---
st.subheader("Recomendación del modelo")

best_i = int(np.argmax([r["EV"] for r in rows]))
best = rows[best_i]
best_ev = best["EV"]
best_label = best["Resultado"]
best_odds_val = odds[best_i]
best_prob = probs[best_i]
kelly_f = fractional_kelly(best_prob, best_odds_val, KELLY_FRACTION, KELLY_CAP)

if best_ev >= EV_THRESHOLD_MED:
    st.success(
        f"**Hay valor: {best_label} @ {best_odds_val:.2f}**\n\n"
        f"EV {best_ev:+.1%} ({best['Señal']}) · probabilidad del modelo: {best_prob:.1%} "
        f"vs probabilidad implícita de la cuota: {best['Prob. implícita (sin vig)']:.1%}\n\n"
        f"Apuesta de referencia (Kelly/8, sobre $1000): **${best['Apuesta ref. (Kelly/8)']:.2f}**")
elif best_ev > 0:
    st.info(
        f"**Valor marginal: {best_label} @ {best_odds_val:.2f}** — "
        f"EV {best_ev:+.1%} (por debajo del umbral del 5%). "
        f"El edge es pequeño y el margen de error del modelo puede neutralizarlo.")
else:
    st.warning(
        f"**Sin valor detectado** — el mejor resultado ({best_label}) tiene EV {best_ev:+.1%}. "
        f"Las cuotas actuales no ofrecen ventaja estadística según el modelo.")

st.caption(f"Overround del mercado: {(ovr - 1):.1%} · "
           f"Elo: {home} {rh:.0f} vs {away} {ra:.0f} · "
           f"Kelly fraccional: 1/8 del Kelly completo, cap {KELLY_CAP:.0%}")

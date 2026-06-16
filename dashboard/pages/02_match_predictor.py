"""Match Predictor: probabilidades 1X2, comparación con cuotas, EV y Kelly.

Usa el ensemble calibrado (Elo + Poisson + Dixon-Coles → calibración
multinomial). Tras la tabla de EV muestra la apuesta recomendada y permite
registrarla en el bet_log (paper trading).
"""
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
from config.settings import (EV_THRESHOLD_MED, KELLY_CAP, KELLY_FRACTION)
from src.betting.bankroll import fractional_kelly, kelly_stake
from src.betting.bet_log import add_bet, current_bankroll
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import overround, remove_vig_multiplicative

st.set_page_config(page_title="Match Predictor", page_icon="🎯", layout="wide")
st.title("🎯 Match Predictor")

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
m4.metric("Goles esperados", f"{lam_h:.2f} – {lam_a:.2f}")

with st.expander("Detalle por modelo y matriz de marcadores"):
    p_elo = elo_pred.predict_proba(home, away, neutral=neutral)
    p_pois = poisson_model.predict_proba(rh, ra, neutral=neutral)
    st.dataframe(pd.DataFrame(
        [p_elo, p_pois, probs],
        index=["Elo + draw model", "Poisson global", "Ensemble calibrado"],
        columns=[home, "Empate", away],
    ).style.format("{:.1%}"))
    matrix = poisson_model.score_matrix(lam_h, lam_a, max_goals=6)
    fig = px.imshow(matrix * 100, text_auto=".1f",
                    labels=dict(x=f"Goles {away}", y=f"Goles {home}", color="%"),
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# --- Cuotas y EV ---
st.subheader("Comparación contra cuotas (decimal)")
o1, o2, o3 = st.columns(3)
odds_h = o1.number_input(f"Cuota {home}", min_value=1.01, value=2.50, step=0.01)
odds_d = o2.number_input("Cuota empate", min_value=1.01, value=3.20, step=0.01)
odds_a = o3.number_input(f"Cuota {away}", min_value=1.01, value=2.80, step=0.01)

odds = [odds_h, odds_d, odds_a]
implied = remove_vig_multiplicative(odds)
ovr = overround(odds)

paper_bankroll = current_bankroll(paper=True)
bankroll = st.number_input("Bankroll (unidades)", min_value=1.0,
                           value=float(round(paper_bankroll, 2)), step=50.0,
                           help="Default: bankroll actual de paper trading")

rows = []
for i, (label, p, o) in enumerate(zip([home, "Empate", away], probs, odds)):
    ev = expected_value(p, o)
    rows.append({
        "Resultado": label,
        "Prob. modelo": p,
        "Prob. implícita (sin vig)": implied[i],
        "Edge": p - implied[i],
        "EV": ev,
        "Señal": classify_ev(ev),
        "Stake Kelly 1/8": kelly_stake(bankroll, p, o) if ev > 0 else 0.0,
    })
df = pd.DataFrame(rows)
st.dataframe(df.style.format({
    "Prob. modelo": "{:.1%}", "Prob. implícita (sin vig)": "{:.1%}",
    "Edge": "{:+.1%}", "EV": "{:+.1%}", "Stake Kelly 1/8": "{:.2f}",
}).map(lambda v: "background-color: #d4edda" if v in ("HIGH", "MEDIUM")
       else ("background-color: #fff3cd" if v == "LOW" else ""),
       subset=["Señal"]), hide_index=True, use_container_width=True)

# --- Apuesta recomendada ---
st.subheader("Apuesta recomendada")

MARKETS = ["1X2-home", "1X2-draw", "1X2-away"]
best_i = int(np.argmax([r["EV"] for r in rows]))
best = rows[best_i]
best_ev = best["EV"]
best_label = best["Resultado"]
best_odds = odds[best_i]
best_prob = probs[best_i]
best_stake = best["Stake Kelly 1/8"]
kelly_f = fractional_kelly(best_prob, best_odds, KELLY_FRACTION, KELLY_CAP)

if best_ev >= EV_THRESHOLD_MED:
    st.success(
        f"**APOSTAR: {best_label} @ {best_odds:.2f}**\n\n"
        f"EV {best_ev:+.1%} ({best['Señal']}) · prob. modelo {best_prob:.1%} "
        f"vs implícita {best['Prob. implícita (sin vig)']:.1%} · "
        f"stake Kelly 1/8: **{best_stake:.2f}** "
        f"({kelly_f:.2%} del bankroll)")
elif best_ev > 0:
    st.info(
        f"**Valor marginal: {best_label} @ {best_odds:.2f}** — "
        f"EV {best_ev:+.1%} (LOW, por debajo del umbral del 5%). "
        f"Opcional con stake reducido: {best_stake:.2f}. "
        f"Con edges tan finos el error del modelo puede comerse el valor.")
else:
    st.warning(
        f"**No apostar** — ninguna cuota ofrece valor. "
        f"El mejor resultado ({best_label}) tiene EV {best_ev:+.1%}. "
        f"Pasar también es una decisión correcta.")

if best_ev > 0:
    if st.button(f"Registrar apuesta (paper): {best_label} @ {best_odds:.2f} "
                 f"por {best_stake:.2f}", type="primary"):
        bet_id = add_bet(home, away, MARKETS[best_i],
                         stake=float(best_stake), odds=float(best_odds),
                         model_prob=float(best_prob), ev=float(best_ev),
                         kelly_fraction=float(kelly_f), paper=True)
        st.success(f"Apuesta #{bet_id} registrada en el bet log (paper). "
                   "Liquídala en Bankroll Tracker cuando termine el partido.")

st.caption(f"Overround del mercado: {(ovr - 1):.1%} · Kelly fraccional: "
           f"{KELLY_FRACTION:.0%} del Kelly completo, cap {KELLY_CAP:.0%} del bankroll · "
           f"Elo: {home} {rh:.0f} vs {away} {ra:.0f}")

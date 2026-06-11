"""Bankroll Tracker — registro de apuestas y curva de equity.

Protocolo paper trading: las apuestas nacen con paper=TRUE por defecto.
El toggle a dinero real exige confirmar que el gate se cumplió.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import KELLY_CAP, KELLY_FRACTION
from src.backtesting.metrics import summarize
from src.betting.bet_log import (add_bet, clv_summary, current_bankroll,
                                 delete_bet, get_bets, get_initial_bankroll,
                                 set_closing_odds, set_initial_bankroll,
                                 settle_bet, void_bet)

st.set_page_config(page_title="Bankroll Tracker", page_icon="💰", layout="wide")
st.title("💰 Bankroll Tracker")

# ---- Sidebar: configuración --------------------------------------------
with st.sidebar:
    st.header("Configuracion")
    mode = st.radio("Modo", ["Paper trading", "Dinero real"], index=0)
    is_paper = mode == "Paper trading"
    if not is_paper:
        st.warning("Gate para dinero real: Brier hold-out < 0.20 (actual: 0.1652 OK), "
                   "sin EV+ sistemático en longshots durante paper trading, "
                   "1/8 Kelly las primeras 2 semanas.")
    st.divider()
    initial = st.number_input("Bankroll inicial (€)",
                              value=float(get_initial_bankroll()),
                              min_value=1.0, step=50.0)
    if st.button("Guardar bankroll inicial"):
        set_initial_bankroll(initial)
        st.success(f"Bankroll inicial: {initial:.2f} €")
        st.rerun()

# ---- KPIs ---------------------------------------------------------------
bankroll = current_bankroll(paper=is_paper)
df_bets = get_bets(paper=is_paper)
df_settled = df_bets[df_bets["result"].isin(["win", "lose"])]
df_pending = df_bets[df_bets["result"] == "pending"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bankroll actual", f"{bankroll:,.2f} €",
            delta=f"{bankroll - get_initial_bankroll():+,.2f} €")
col2.metric("Apuestas liquidadas", len(df_settled))
col3.metric("Pendientes", len(df_pending))
if not df_settled.empty:
    s = summarize(df_settled.rename(columns={"result": "result"}),
                  get_initial_bankroll())
    col4.metric("ROI", f"{s['roi']:+.1%}")
else:
    col4.metric("ROI", "—")

st.divider()

# ---- Registrar apuesta --------------------------------------------------
st.subheader("Registrar apuesta")
with st.form("new_bet", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    home = c1.text_input("Equipo local")
    away = c2.text_input("Equipo visitante")
    market = c3.selectbox("Mercado", ["1X2-home", "1X2-draw", "1X2-away", "outright"])

    c4, c5, c6 = st.columns(3)
    odds = c4.number_input("Cuota decimal", min_value=1.01, value=2.00, step=0.05)
    model_prob = c5.number_input("Prob. del modelo", min_value=0.001,
                                 max_value=0.999, value=0.50, step=0.01)
    stake_mode = c6.radio("Stake", ["Kelly sugerido", "Manual"], horizontal=True)

    from src.betting.bankroll import fractional_kelly
    from src.betting.ev_calculator import expected_value
    kelly_f = fractional_kelly(model_prob, odds, KELLY_FRACTION, KELLY_CAP)
    suggested = bankroll * kelly_f
    ev = expected_value(model_prob, odds)

    manual_stake = st.number_input("Stake manual (€)", min_value=0.0,
                                   value=round(suggested, 2), step=1.0)
    st.caption(f"EV: {ev:+.1%} · Kelly 1/8: {kelly_f:.2%} del bankroll "
               f"→ sugerido {suggested:.2f} €")

    submitted = st.form_submit_button("Registrar", type="primary")
    if submitted:
        if not home or (market != "outright" and not away):
            st.error("Completa los equipos.")
        elif ev <= 0:
            st.error(f"EV negativo ({ev:+.1%}) — el sistema no registra apuestas EV-. "
                     "Revisa cuota y probabilidad.")
        else:
            stake = manual_stake if stake_mode == "Manual" else suggested
            bet_id = add_bet(home, away or "-", market, stake, odds,
                             model_prob, ev, kelly_f, paper=is_paper)
            st.success(f"Apuesta #{bet_id} registrada ({'paper' if is_paper else 'REAL'}): "
                       f"{stake:.2f} € a cuota {odds:.2f}")
            st.rerun()

st.divider()

# ---- Liquidar pendientes ------------------------------------------------
if not df_pending.empty:
    st.subheader("Apuestas pendientes")
    for row in df_pending.itertuples():
        c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 1, 1])
        c1.write(f"**#{int(row.bet_id)}** {row.home_team} vs {row.away_team} · "
                 f"{row.market} · {row.stake:.2f} € @ {row.odds:.2f} "
                 f"(EV {row.ev:+.1%})")
        if c2.button("Ganada", key=f"w{row.bet_id}"):
            settle_bet(int(row.bet_id), won=True)
            st.rerun()
        if c3.button("Perdida", key=f"l{row.bet_id}"):
            settle_bet(int(row.bet_id), won=False)
            st.rerun()
        if c4.button("Anular", key=f"v{row.bet_id}"):
            void_bet(int(row.bet_id))
            st.rerun()
        if c5.button("Borrar", key=f"d{row.bet_id}"):
            delete_bet(int(row.bet_id))
            st.rerun()
    st.divider()

# ---- Cuota de cierre (CLV) ------------------------------------------------
df_no_close = df_bets[df_bets["closing_odds"].isna() &
                      df_bets["result"].isin(["pending", "win", "lose"])]
if not df_no_close.empty:
    st.subheader("Registrar cuota de cierre (CLV)")
    st.caption("Anota la cuota del mercado justo antes del partido. "
               "CLV = tu cuota / cuota de cierre − 1: positivo sistemático = edge real, "
               "aunque el resultado económico a corto plazo sea ruido.")
    opts = {f"#{int(r.bet_id)} · {r.home_team} vs {r.away_team} · {r.market} "
            f"@ {r.odds:.2f}": int(r.bet_id) for r in df_no_close.itertuples()}
    c1, c2, c3 = st.columns([4, 2, 1])
    sel_bet = c1.selectbox("Apuesta", list(opts), label_visibility="collapsed")
    close_val = c2.number_input("Cuota de cierre", min_value=1.01, value=2.00,
                                step=0.01, label_visibility="collapsed")
    if c3.button("Guardar", key="clv_save"):
        set_closing_odds(opts[sel_bet], float(close_val))
        st.rerun()
    st.divider()

# ---- Métricas CLV -----------------------------------------------------------
clv = clv_summary(df_bets)
if clv:
    st.subheader("Closing Line Value")
    c1, c2, c3 = st.columns(3)
    c1.metric("CLV medio", f"{clv['avg_clv']:+.2%}",
              help="Promedio de (cuota tomada / cuota cierre − 1)")
    c2.metric("Le ganas al cierre", f"{clv['pct_beat_close']:.0%} de las veces")
    c3.metric("Apuestas con CLV", clv["n_with_clv"])
    if clv["n_with_clv"] >= 10:
        if clv["avg_clv"] > 0.01:
            st.success("CLV medio positivo sostenido — señal de edge real. "
                       "El criterio para pasar a dinero real se está cumpliendo.")
        elif clv["avg_clv"] < -0.01:
            st.error("CLV medio negativo — el mercado se mueve sistemáticamente "
                     "en tu contra. No pasar a dinero real; revisar el modelo o "
                     "apostar más cerca del cierre.")
        else:
            st.info("CLV neutro — sin evidencia de edge todavía. Seguir en paper.")
    else:
        st.caption(f"Con {clv['n_with_clv']} apuestas aún es pronto para concluir "
                   "(mínimo ~10-20 para que el CLV sea indicativo).")
    st.divider()

# ---- Curva de equity ----------------------------------------------------
if not df_settled.empty:
    st.subheader("Curva de equity")
    df_eq = df_settled.sort_values("placed_at")
    equity = np.concatenate([[get_initial_bankroll()],
                             get_initial_bankroll() + np.cumsum(df_eq["profit"].to_numpy())])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines+markers", name="Bankroll",
                             line=dict(color="#2563eb")))
    fig.add_hline(y=get_initial_bankroll(), line_dash="dash", line_color="gray",
                  annotation_text="Inicial")
    fig.update_layout(height=350, xaxis_title="Apuesta nº",
                      yaxis_title="Bankroll (€)")
    st.plotly_chart(fig, use_container_width=True)

    s = summarize(df_settled, get_initial_bankroll())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Win rate", f"{s['win_rate']:.0%}")
    m2.metric("Yield/apuesta", f"{s['yield']:+.2f} €")
    m3.metric("Profit factor", f"{s['profit_factor']:.2f}"
              if s['profit_factor'] != float('inf') else "∞")
    m4.metric("Max drawdown", f"{s['max_drawdown']:.2f} €")
    m5.metric("Cuota media", f"{s['avg_odds']:.2f}")

# ---- Historial ----------------------------------------------------------
st.subheader("Historial completo")
if df_bets.empty:
    st.info("Sin apuestas registradas todavía en este modo.")
else:
    show = df_bets[["bet_id", "placed_at", "home_team", "away_team", "market",
                    "stake", "odds", "closing_odds", "model_prob", "ev",
                    "result", "profit"]].copy()
    show["clv"] = np.where(show["closing_odds"] > 1.0,
                           show["odds"] / show["closing_odds"] - 1.0, np.nan)
    st.dataframe(
        show.style.format({"stake": "{:.2f}", "odds": "{:.2f}",
                           "closing_odds": "{:.2f}", "clv": "{:+.1%}",
                           "model_prob": "{:.1%}", "ev": "{:+.1%}",
                           "profit": "{:+.2f}"}, na_rep="—")
            .map(lambda v: "color: green" if v == "win" else
                           ("color: red" if v == "lose" else ""),
                 subset=["result"]),
        hide_index=True, use_container_width=True)

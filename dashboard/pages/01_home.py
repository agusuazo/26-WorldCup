"""Home — próximos partidos, top EV+ del día y estado del bankroll."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.model_store import (load_best_predictor,
                                             load_upcoming_wc_matches)
from config.settings import PROCESSED_DIR
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import remove_vig_multiplicative

st.set_page_config(page_title="Home — Mundial 2026", page_icon="🏠", layout="wide")
st.title("🏠 Home — Mundial FIFA 2026")

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
                   "Actualizar en la página 'update results'.")
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
                   f"({raw_odds['bookmaker'].nunique()} bookmakers). "
                   "Actualizar: `python scripts/fetch_odds.py`")
except Exception:
    pass

if odds_df is None:
    st.info("Sin snapshot de cuotas. Ejecuta `python scripts/fetch_odds.py` "
            "(requiere ODDS_API_KEY) para ver EV+ automático, o usa el "
            "match predictor con cuotas manuales.")

# ---- Próximos partidos con predicciones ----------------------------------
st.subheader("Próximos partidos — predicciones del modelo")

if upcoming.empty:
    st.warning("No hay partidos futuros del Mundial en la base. "
               "¿Falta re-ejecutar la ingesta?")
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
        # EV+ si hay cuotas para este partido
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
                row["Apuesta"] = best_label
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

st.divider()

# ---- Dos columnas: favoritos del simulador + bankroll --------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Favoritos al título (Monte Carlo)")
    sim_path = PROCESSED_DIR / "sim_results.parquet"
    if sim_path.exists():
        sim = pd.read_parquet(sim_path).head(8)
        fig = go.Figure(go.Bar(
            x=sim["champion_pct"], y=sim["team"], orientation="h",
            marker_color="#2563eb",
            text=[f"{v:.1f}%" for v in sim["champion_pct"]],
            textposition="outside"))
        fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="P(campeón) %")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Detalle completo en Tournament Simulator.")
    else:
        st.info("Ejecuta `python scripts/run_simulator.py` para ver favoritos.")

with col_right:
    st.subheader("Bankroll (paper trading)")
    try:
        from src.betting.bet_log import current_bankroll, get_bets, get_initial_bankroll
        bk = current_bankroll(paper=True)
        initial = get_initial_bankroll()
        df_bets = get_bets(paper=True)
        settled = df_bets[df_bets["result"].isin(["win", "lose"])]
        c1, c2, c3 = st.columns(3)
        c1.metric("Bankroll", f"{bk:,.0f} €", delta=f"{bk - initial:+,.0f} €")
        c2.metric("Liquidadas", len(settled))
        c3.metric("Pendientes", int((df_bets["result"] == "pending").sum()))
        if not settled.empty:
            eq = np.concatenate([[initial],
                                 initial + np.cumsum(
                                     settled.sort_values("placed_at")["profit"].to_numpy())])
            fig2 = go.Figure(go.Scatter(y=eq, mode="lines", line=dict(color="#16a34a")))
            fig2.add_hline(y=initial, line_dash="dash", line_color="gray")
            fig2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Sin apuestas liquidadas. Registra apuestas en Bankroll Tracker.")
    except Exception as e:
        st.warning(f"Bet log no disponible: {e}")

st.divider()
st.caption("Protocolo: paper trading la primera semana · gate Brier < 0.20 "
           "(actual 0.1652 ✓) · 1/8 Kelly con cap 5% · "
           "umbral EV+ outright 15%.")

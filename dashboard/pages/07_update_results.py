"""Actualizar Datos — ingreso de resultados y recálculo completo.

Dos vías de ingreso:
- Descarga del dataset martj42 (GitHub) — fuente oficial, puede tardar en
  actualizarse tras cada jornada.
- Entrada manual — efecto inmediato; sobrevive a rebuilds vía manual_results.

El botón "Recalcular todo" ejecuta: ingesta + Elo → re-entrenamiento de
modelos → simulación Monte Carlo condicionada a los resultados reales.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd
import streamlit as st

from config.settings import DB_PATH
from src.ingestion.updater import (delete_manual_result, download_latest_results,
                                   get_last_refresh, get_manual_results,
                                   refresh_all, save_manual_result)

st.set_page_config(page_title="Actualizar Datos", page_icon="🔄", layout="wide")
st.title("🔄 Actualizar Datos del Torneo")

# ---- Estado actual --------------------------------------------------------
last = get_last_refresh()
col1, col2, col3 = st.columns(3)


def _load_wc_matches():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute("""
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup' AND date >= '2026-06-01'
            ORDER BY date
        """).df()
    finally:
        con.close()


wc = _load_wc_matches()
played = wc[wc["home_score"].notna()]
today = pd.Timestamp(date.today())
pending = wc[wc["home_score"].isna() & (wc["date"] <= today)]

col1.metric("Partidos WC jugados (en BD)", len(played))
col2.metric("Pendientes de resultado", len(pending),
            help="Partidos con fecha pasada y sin marcador cargado")
col3.metric("Último recálculo",
            pd.Timestamp(last["refreshed_at"]).strftime("%d-%b %H:%M UTC")
            if last else "nunca")

if last:
    st.caption(f"Último recálculo: Brier {last['brier_holdout']:.4f} · "
               f"condicionado a {last['n_group_played']} partidos de grupos "
               f"+ {last['n_ko_played']} eliminatorias · "
               f"favorito: {last['top5'][0]['team']} "
               f"({last['top5'][0]['champion_pct']:.1f}%)")

st.divider()

# ---- Sección 1: descarga del dataset ---------------------------------------
st.subheader("1 · Descargar dataset actualizado (GitHub)")
st.caption("Fuente: martj42/international_results. El dataset suele actualizarse "
           "horas o días después de cada jornada — para resultados inmediatos "
           "usa la entrada manual de abajo.")

if st.button("⬇️ Descargar últimos resultados", type="secondary"):
    with st.spinner("Descargando results.csv y shootouts.csv..."):
        try:
            info = download_latest_results()
            if info["rows_new"] > 0:
                st.success(f"{info['rows_new']:,} filas nuevas "
                           f"({info['rows_before']:,} → {info['rows_after']:,}). "
                           "Pulsa 'Recalcular todo' para incorporarlas.")
            else:
                st.info("El dataset no tiene filas nuevas todavía.")
            with st.expander("Últimos 10 resultados en el dataset"):
                st.dataframe(info["latest_results"], hide_index=True)
        except Exception as e:
            st.error(f"Descarga fallida: {e}")

st.divider()

# ---- Sección 2: entrada manual ----------------------------------------------
st.subheader("2 · Ingresar resultado manual")

tab_fixture, tab_free = st.tabs(["Partido del fixture", "Partido libre (KO)"])

with tab_fixture:
    if pending.empty:
        st.info("No hay partidos pendientes de resultado hasta hoy.")
    else:
        options = {
            f"{pd.Timestamp(r.date):%d-%b} · {r.home_team} vs {r.away_team}": r
            for r in pending.itertuples()}
        sel = st.selectbox("Partido", list(options))
        r = options[sel]
        c1, c2 = st.columns(2)
        hs = c1.number_input(f"Goles {r.home_team}", 0, 15, 0, key="fx_hs")
        aws = c2.number_input(f"Goles {r.away_team}", 0, 15, 0, key="fx_as")
        if st.button("Guardar resultado", type="primary", key="fx_save"):
            save_manual_result(str(pd.Timestamp(r.date).date()),
                               r.home_team, r.away_team, int(hs), int(aws))
            st.success(f"Guardado: {r.home_team} {hs}-{aws} {r.away_team}. "
                       "Recalcula para que los modelos lo incorporen.")
            st.rerun()

with tab_free:
    st.caption("Para eliminatorias que aún no existen como filas en el dataset.")
    from dashboard.components.model_store import wc_teams
    teams = wc_teams()
    c1, c2, c3 = st.columns([2, 2, 1])
    fh = c1.selectbox("Local", teams, key="fr_h")
    fa = c2.selectbox("Visitante", teams, index=1, key="fr_a")
    fdate = c3.date_input("Fecha", value=date.today(), key="fr_d")
    c4, c5 = st.columns(2)
    fhs = c4.number_input(f"Goles {fh}", 0, 15, 0, key="fr_hs")
    fas = c5.number_input(f"Goles {fa}", 0, 15, 0, key="fr_as")
    winner = None
    if fhs == fas:
        winner = st.selectbox("Ganador por penales (eliminatorias)",
                              [fh, fa], key="fr_w")
    if st.button("Guardar resultado", type="primary", key="fr_save"):
        if fh == fa:
            st.error("Elige dos equipos distintos.")
        else:
            save_manual_result(str(fdate), fh, fa, int(fhs), int(fas),
                               winner=winner)
            st.success(f"Guardado: {fh} {fhs}-{fas} {fa}"
                       + (f" (gana {winner} en penales)" if winner else "")
                       + ". Recalcula para incorporarlo.")
            st.rerun()

# Resultados manuales registrados
manual = get_manual_results()
if not manual.empty:
    with st.expander(f"Resultados manuales registrados ({len(manual)})"):
        for r in manual.itertuples():
            c1, c2 = st.columns([6, 1])
            score = f"{int(r.home_score)}-{int(r.away_score)}"
            pen = f" (penales: {r.winner})" if r.winner else ""
            c1.write(f"{pd.Timestamp(r.date):%d-%b} · {r.home_team} {score} "
                     f"{r.away_team}{pen}")
            if c2.button("Borrar", key=f"del_{r.Index}"):
                delete_manual_result(str(pd.Timestamp(r.date).date()),
                                     r.home_team, r.away_team)
                st.rerun()

st.divider()

# ---- Sección 3: recálculo completo -------------------------------------------
st.subheader("3 · Recalcular todo")
st.caption("Ingesta + Elo → re-entrenamiento completo de modelos → "
           "simulación condicionada a resultados reales. Tarda 2-4 minutos.")

n_sims = st.select_slider("Simulaciones Monte Carlo",
                          options=[5_000, 10_000, 25_000, 50_000], value=10_000)

if st.button("🔄 Recalcular todo", type="primary"):
    stages = {"ingesta": "Ingesta + Elo forward pass...",
              "entrenamiento": "Re-entrenando modelos (Poisson, DC, ensemble, calibración)...",
              "simulacion": "Simulación Monte Carlo condicionada..."}
    with st.status("Recalculando...", expanded=True) as status:
        def cb(stage):
            st.write(stages.get(stage, stage))

        try:
            summary = refresh_all(n_sims=n_sims, progress_cb=cb)
            status.update(label="Recálculo completo", state="complete")
        except Exception as e:
            status.update(label="Error en el recálculo", state="error")
            st.error(str(e))
            st.stop()

    st.cache_data.clear()
    st.cache_resource.clear()

    gate = "✅ PASA" if summary["gate_passed"] else "❌ NO PASA"
    st.success(
        f"**Recálculo completo.** Brier hold-out: {summary['brier_holdout']:.4f} "
        f"(gate {gate}) · condicionado a {summary['n_group_played']} partidos "
        f"de grupos + {summary['n_ko_played']} KO.")
    top = " · ".join(f"{t['team']} {t['champion_pct']:.1f}%"
                     for t in summary["top5"])
    st.info(f"Favoritos actualizados: {top}")
    ov = summary["overlay"]
    if any(ov.values()):
        st.caption(f"Overlay manual: {ov['updated']} aplicados, "
                   f"{ov['inserted']} insertados, {ov['obsolete']} ya en el CSV.")

"""Carga cacheada de modelos y datos para el dashboard."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import joblib
import pandas as pd
import streamlit as st

from config.settings import DB_PATH, PROCESSED_DIR, WC2026_FIXTURES
from src.models.elo_model import EloPredictor
from src.models.poisson_model import GlobalPoissonModel


@st.cache_resource
def load_models() -> tuple[EloPredictor, GlobalPoissonModel]:
    draw_model = joblib.load(PROCESSED_DIR / "draw_model.joblib")
    poisson_model = joblib.load(PROCESSED_DIR / "poisson_model.joblib")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ratings = dict(con.execute("SELECT team, elo FROM elo_current").fetchall())
    con.close()
    return EloPredictor(ratings, draw_model), poisson_model


@st.cache_resource
def load_best_predictor():
    """Mejor predictor disponible: calibrador → ensemble → blend Elo+Poisson."""
    for name in ("calibrator.joblib", "ensemble.joblib"):
        path = PROCESSED_DIR / name
        if path.exists():
            return joblib.load(path)
    elo_pred, _ = load_models()
    return elo_pred


@st.cache_data(ttl=900)
def load_upcoming_wc_matches(limit: int = 20) -> pd.DataFrame:
    """Próximos partidos del Mundial sin resultado, ordenados por fecha."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute("""
            SELECT match_id, date, home_team, away_team, city, country
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND home_score IS NULL
              AND date >= current_date
            ORDER BY date
            LIMIT ?
        """, [limit]).df()
    finally:
        con.close()


@st.cache_data(ttl=3600)
def load_elo_table() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT team, elo, n_matches, last_match_date
        FROM elo_current ORDER BY elo DESC
    """).df()
    con.close()
    return df


@st.cache_data
def load_wc2026() -> dict:
    return json.loads(WC2026_FIXTURES.read_text(encoding="utf-8"))


def wc_teams() -> list[str]:
    data = load_wc2026()
    return sorted({t for g in data["groups"].values() for t in g})

"""Storage remoto (Supabase Postgres) para el estado MUTABLE de la app.

En el deploy free-tier el filesystem de Render es efímero: DuckDB viaja
horneado en la imagen como almacén de solo lectura (histórico, Elo, modelos)
y todo lo que el usuario escribe vive en Supabase:

  - bet_log          (apuestas)
  - manual_results   (resultados cargados a mano)
  - app_config       (bankroll inicial, etc.)

Activación: variable de entorno SUPABASE_DB_URL (connection string Postgres,
p.ej. postgresql://postgres:...@db.xxx.supabase.co:5432/postgres).
Sin ella, los módulos siguen usando DuckDB local — el dev local no cambia.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd


def db_url() -> str:
    return os.environ.get("SUPABASE_DB_URL", "")


def is_remote() -> bool:
    return bool(db_url())


def get_conn():
    import psycopg2
    return psycopg2.connect(db_url())


def ensure_schema() -> None:
    """Crea las tablas mutables si no existen (idempotente, al boot)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS bet_log (
        bet_id        SERIAL PRIMARY KEY,
        placed_at     TIMESTAMPTZ DEFAULT now(),
        home_team     TEXT,
        away_team     TEXT,
        market        TEXT,
        stake         DOUBLE PRECISION,
        odds          DOUBLE PRECISION,
        model_prob    DOUBLE PRECISION,
        ev            DOUBLE PRECISION,
        kelly_fraction DOUBLE PRECISION,
        paper         BOOLEAN DEFAULT TRUE,
        result        TEXT DEFAULT 'pending',
        profit        DOUBLE PRECISION,
        closing_odds  DOUBLE PRECISION
    );
    CREATE TABLE IF NOT EXISTS manual_results (
        date        DATE NOT NULL,
        home_team   TEXT NOT NULL,
        away_team   TEXT NOT NULL,
        home_score  INTEGER NOT NULL,
        away_score  INTEGER NOT NULL,
        winner      TEXT,
        tournament  TEXT DEFAULT 'FIFA World Cup',
        neutral     BOOLEAN DEFAULT TRUE,
        entered_at  TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (date, home_team, away_team)
    );
    CREATE TABLE IF NOT EXISTS app_config (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """
    con = get_conn()
    try:
        with con, con.cursor() as cur:
            cur.execute(ddl)
    finally:
        con.close()


def _df(query: str, params=None) -> pd.DataFrame:
    con = get_conn()
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


def _exec(query: str, params=None, fetchone: bool = False):
    con = get_conn()
    try:
        with con, con.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone() if fetchone else None
    finally:
        con.close()


# ---- bet_log ----------------------------------------------------------------

def add_bet(home, away, market, stake, odds, model_prob, ev,
            kelly_fraction, paper=True) -> int:
    row = _exec("""
        INSERT INTO bet_log (placed_at, home_team, away_team, market, stake,
                             odds, model_prob, ev, kelly_fraction, paper)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING bet_id
    """, [datetime.now(), home, away, market, stake, odds, model_prob,
          ev, kelly_fraction, paper], fetchone=True)
    return int(row[0])


def settle_bet(bet_id: int, won: bool) -> None:
    row = _exec("SELECT stake, odds FROM bet_log WHERE bet_id = %s",
                [bet_id], fetchone=True)
    if row is None:
        raise ValueError(f"bet_id {bet_id} no existe")
    stake, odds = float(row[0]), float(row[1])
    profit = stake * (odds - 1) if won else -stake
    _exec("UPDATE bet_log SET result = %s, profit = %s WHERE bet_id = %s",
          ["win" if won else "lose", profit, bet_id])


def void_bet(bet_id: int) -> None:
    _exec("UPDATE bet_log SET result = 'void', profit = 0 WHERE bet_id = %s",
          [bet_id])


def delete_bet(bet_id: int) -> None:
    _exec("DELETE FROM bet_log WHERE bet_id = %s", [bet_id])


def set_closing_odds(bet_id: int, closing_odds: float) -> None:
    _exec("UPDATE bet_log SET closing_odds = %s WHERE bet_id = %s",
          [closing_odds, bet_id])


def get_bets(paper: bool | None = None, pending_only: bool = False) -> pd.DataFrame:
    q = "SELECT * FROM bet_log WHERE TRUE"
    params = []
    if paper is not None:
        q += " AND paper = %s"
        params.append(paper)
    if pending_only:
        q += " AND result = 'pending'"
    q += " ORDER BY placed_at DESC"
    return _df(q, params or None)


def settled_profit_sum(paper: bool = True) -> float:
    row = _exec("""
        SELECT COALESCE(SUM(profit), 0) FROM bet_log
        WHERE paper = %s AND result IN ('win', 'lose', 'void')
    """, [paper], fetchone=True)
    return float(row[0])


# ---- app_config ----------------------------------------------------------------

def get_config(key: str) -> str | None:
    row = _exec("SELECT value FROM app_config WHERE key = %s", [key],
                fetchone=True)
    return row[0] if row else None


def set_config(key: str, value: str) -> None:
    _exec("""
        INSERT INTO app_config (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, [key, value])


# ---- manual_results --------------------------------------------------------------

def save_manual_result(date, home, away, home_score, away_score,
                       winner=None, tournament="FIFA World Cup",
                       neutral=True) -> None:
    _exec("""
        INSERT INTO manual_results (date, home_team, away_team, home_score,
                                    away_score, winner, tournament, neutral, entered_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, home_team, away_team) DO UPDATE SET
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            winner     = EXCLUDED.winner,
            entered_at = EXCLUDED.entered_at
    """, [date, home, away, home_score, away_score, winner,
          tournament, neutral, datetime.now()])


def delete_manual_result(date, home, away) -> None:
    _exec("DELETE FROM manual_results WHERE date = %s AND home_team = %s "
          "AND away_team = %s", [date, home, away])


def get_manual_results() -> pd.DataFrame:
    return _df("SELECT * FROM manual_results ORDER BY date DESC")

"""CRUD del registro de apuestas (tabla bet_log en DuckDB).

Protocolo paper trading: toda apuesta nace con paper=TRUE. El flag se
decide al registrarla; el gate para dinero real está documentado en el
plan (Brier < 0.20, sin EV+ sistemático en longshots, 1/8 Kelly).
"""
from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pandas as pd

from config.settings import DB_PATH, PROCESSED_DIR
from src.db import remote

BANKROLL_CONFIG = PROCESSED_DIR / "bankroll_config.json"
DEFAULT_INITIAL_BANKROLL = 1000.0


# ---- Configuración de bankroll -----------------------------------------

def get_initial_bankroll() -> float:
    if remote.is_remote():
        v = remote.get_config("initial_bankroll")
        return float(v) if v else DEFAULT_INITIAL_BANKROLL
    if BANKROLL_CONFIG.exists():
        return float(json.loads(BANKROLL_CONFIG.read_text())["initial"])
    return DEFAULT_INITIAL_BANKROLL


def set_initial_bankroll(amount: float) -> None:
    if remote.is_remote():
        remote.set_config("initial_bankroll", str(amount))
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BANKROLL_CONFIG.write_text(json.dumps({"initial": amount}))


# ---- Operaciones sobre bet_log ------------------------------------------

def add_bet(home_team: str, away_team: str, market: str,
            stake: float, odds: float, model_prob: float,
            ev: float, kelly_fraction: float, paper: bool = True) -> int:
    """Registra una apuesta pendiente. Devuelve el bet_id asignado."""
    if remote.is_remote():
        return remote.add_bet(home_team, away_team, market, stake, odds,
                              model_prob, ev, kelly_fraction, paper)
    con = duckdb.connect(str(DB_PATH))
    try:
        next_id = con.execute(
            "SELECT COALESCE(MAX(bet_id), 0) + 1 FROM bet_log").fetchone()[0]
        con.execute("""
            INSERT INTO bet_log (bet_id, placed_at, home_team, away_team,
                                 market, stake, odds, model_prob, ev,
                                 kelly_fraction, paper, result, profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL)
        """, [next_id, datetime.now(), home_team, away_team, market,
              stake, odds, model_prob, ev, kelly_fraction, paper])
    finally:
        con.close()
    return int(next_id)


def settle_bet(bet_id: int, won: bool) -> None:
    """Liquida una apuesta: result = win/lose, profit calculado."""
    if remote.is_remote():
        return remote.settle_bet(bet_id, won)
    con = duckdb.connect(str(DB_PATH))
    try:
        row = con.execute(
            "SELECT stake, odds FROM bet_log WHERE bet_id = ?",
            [bet_id]).fetchone()
        if row is None:
            raise ValueError(f"bet_id {bet_id} no existe")
        stake, odds = row
        profit = stake * (odds - 1) if won else -stake
        con.execute("""
            UPDATE bet_log SET result = ?, profit = ?
            WHERE bet_id = ?
        """, ["win" if won else "lose", profit, bet_id])
    finally:
        con.close()


def set_closing_odds(bet_id: int, closing_odds: float) -> None:
    """Registra la cuota de cierre del mercado para calcular CLV.

    CLV = cuota_tomada / cuota_cierre - 1. Positivo = le ganaste al cierre
    (edge real); negativo = el mercado se movió en tu contra.
    """
    if closing_odds <= 1.0:
        raise ValueError("La cuota de cierre debe ser > 1.0")
    if remote.is_remote():
        return remote.set_closing_odds(bet_id, closing_odds)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("UPDATE bet_log SET closing_odds = ? WHERE bet_id = ?",
                    [closing_odds, bet_id])
    finally:
        con.close()


def clv_summary(df_bets: pd.DataFrame) -> dict | None:
    """Métricas de Closing Line Value sobre apuestas con cuota de cierre.

    El CLV converge mucho antes que el ROI: con ~20 apuestas ya es indicativo
    de si hay edge real, aunque el resultado económico sea ruido.
    """
    df = df_bets.dropna(subset=["closing_odds"])
    df = df[df["closing_odds"] > 1.0]
    if df.empty:
        return None
    clv = df["odds"] / df["closing_odds"] - 1.0
    return {
        "n_with_clv": len(df),
        "avg_clv": float(clv.mean()),
        "median_clv": float(clv.median()),
        "pct_beat_close": float((clv > 0).mean()),
        "clv_series": clv,
    }


def void_bet(bet_id: int) -> None:
    """Anula una apuesta (cuota anulada / push): profit 0."""
    if remote.is_remote():
        return remote.void_bet(bet_id)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("UPDATE bet_log SET result = 'void', profit = 0 "
                    "WHERE bet_id = ?", [bet_id])
    finally:
        con.close()


def delete_bet(bet_id: int) -> None:
    if remote.is_remote():
        return remote.delete_bet(bet_id)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("DELETE FROM bet_log WHERE bet_id = ?", [bet_id])
    finally:
        con.close()


def get_bets(paper: bool | None = None,
             pending_only: bool = False) -> pd.DataFrame:
    """Lee el bet_log. `paper=None` devuelve todo."""
    if remote.is_remote():
        return remote.get_bets(paper, pending_only)
    query = "SELECT * FROM bet_log WHERE 1=1"
    params = []
    if paper is not None:
        query += " AND paper = ?"
        params.append(paper)
    if pending_only:
        query += " AND result = 'pending'"
    query += " ORDER BY placed_at DESC"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(query, params).df()
    finally:
        con.close()


def current_bankroll(paper: bool = True) -> float:
    """Bankroll actual = inicial + suma de profits liquidados."""
    if remote.is_remote():
        return get_initial_bankroll() + remote.settled_profit_sum(paper)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        total = con.execute("""
            SELECT COALESCE(SUM(profit), 0) FROM bet_log
            WHERE paper = ? AND result IN ('win', 'lose', 'void')
        """, [paper]).fetchone()[0]
    finally:
        con.close()
    return get_initial_bankroll() + float(total)

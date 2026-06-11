"""Actualización de datos durante el torneo y recálculo completo.

Flujo:
  1. Ingresar resultados — por descarga del dataset martj42 (GitHub) o
     entrada manual (tabla manual_results, sobrevive a rebuilds).
  2. refresh_all(): ingesta + Elo → re-entrenamiento completo →
     simulación Monte Carlo condicionada al estado real del torneo.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from config.settings import DB_PATH, PROCESSED_DIR, RESULTS_CSV, SHOOTOUTS_CSV, ROOT

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
SHOOTOUTS_URL = ("https://raw.githubusercontent.com/martj42/"
                 "international_results/master/shootouts.csv")
LAST_REFRESH = PROCESSED_DIR / "last_refresh.json"


# ---- Descarga del dataset ------------------------------------------------

def download_latest_results(timeout: int = 60) -> dict:
    """Descarga results.csv y shootouts.csv de martj42 (GitHub).

    Hace backup del CSV anterior y solo lo reemplaza si el nuevo tiene al
    menos tantas filas. Devuelve resumen con filas nuevas detectadas.
    """
    old_rows = 0
    if RESULTS_CSV.exists():
        old_rows = sum(1 for _ in RESULTS_CSV.open(encoding="utf-8")) - 1

    resp = requests.get(RESULTS_URL, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    new_rows = content.count(b"\n") - 1

    if new_rows < old_rows:
        raise ValueError(
            f"El CSV descargado tiene menos filas ({new_rows:,}) que el local "
            f"({old_rows:,}) — no se reemplaza. ¿Problema con la fuente?")

    if RESULTS_CSV.exists():
        shutil.copy2(RESULTS_CSV, RESULTS_CSV.with_name("results_backup.csv"))
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.write_bytes(content)

    # shootouts.csv: necesario para ganadores por penales en eliminatorias
    try:
        so = requests.get(SHOOTOUTS_URL, timeout=timeout)
        so.raise_for_status()
        SHOOTOUTS_CSV.write_bytes(so.content)
        shootouts_ok = True
    except Exception:
        shootouts_ok = False

    # Últimos resultados añadidos (informativo)
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    played = df.dropna(subset=["home_score"])
    recent = played.nlargest(10, "date")[
        ["date", "home_team", "away_team", "home_score", "away_score"]]

    return {
        "rows_before": old_rows,
        "rows_after": new_rows,
        "rows_new": new_rows - old_rows,
        "shootouts_downloaded": shootouts_ok,
        "latest_results": recent,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


# ---- Entrada manual de resultados -----------------------------------------

def save_manual_result(date: str, home: str, away: str,
                       home_score: int, away_score: int,
                       winner: str | None = None,
                       tournament: str = "FIFA World Cup",
                       neutral: bool = True) -> None:
    """Guarda un resultado manual (upsert en manual_results) y lo aplica
    de inmediato a la tabla matches si la fila existe.

    `winner`: solo para eliminatorias empatadas (ganador por penales).
    """
    if home_score == away_score and winner is not None and winner not in (home, away):
        raise ValueError(f"winner debe ser {home!r} o {away!r}")

    from src.db import remote
    if remote.is_remote():
        remote.save_manual_result(date, home, away, home_score, away_score,
                                  winner=winner, tournament=tournament,
                                  neutral=neutral)
        return

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("""
            INSERT OR REPLACE INTO manual_results
                (date, home_team, away_team, home_score, away_score,
                 winner, tournament, neutral, entered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [date, home, away, home_score, away_score, winner,
              tournament, neutral, datetime.now()])
        # efecto inmediato (sin esperar al rebuild) si el match ya existe
        con.execute("""
            UPDATE matches SET home_score = ?, away_score = ?
            WHERE date = ? AND home_team = ? AND away_team = ?
              AND home_score IS NULL
        """, [home_score, away_score, date, home, away])
    finally:
        con.close()


def delete_manual_result(date: str, home: str, away: str) -> None:
    """Elimina un resultado manual (y revierte el score en matches si el
    CSV no lo trae — se restaura en el próximo rebuild)."""
    from src.db import remote
    if remote.is_remote():
        return remote.delete_manual_result(date, home, away)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("DELETE FROM manual_results WHERE date = ? "
                    "AND home_team = ? AND away_team = ?", [date, home, away])
    finally:
        con.close()


def get_manual_results() -> pd.DataFrame:
    from src.db import remote
    if remote.is_remote():
        return remote.get_manual_results()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        if "manual_results" not in tables:
            return pd.DataFrame()
        return con.execute(
            "SELECT * FROM manual_results ORDER BY date DESC").df()
    finally:
        con.close()


# ---- Recálculo completo ----------------------------------------------------

def refresh_all(n_sims: int = 10_000, seed: int = 42,
                progress_cb=None) -> dict:
    """Recálculo completo del sistema:
    1. Ingesta (CSV + overlay manual) + Elo forward pass
    2. Re-entrenamiento de todos los modelos (Poisson, DC, ensemble, calibración)
    3. Simulación Monte Carlo condicionada al estado real del torneo

    `progress_cb(fase: str)` se llama al inicio de cada fase.
    """
    sys.path.insert(0, str(ROOT))   # para importar scripts.run_training

    def notify(stage: str):
        if progress_cb:
            progress_cb(stage)

    # --- 1. Ingesta + Elo ---
    notify("ingesta")
    from src.ingestion.pipeline import run_ingestion
    ingestion = run_ingestion(verbose=False)

    # --- 2. Re-entrenamiento completo ---
    notify("entrenamiento")
    from scripts.run_training import train_all
    training = train_all(verbose=False)

    # --- 3. Simulación condicionada ---
    notify("simulacion")
    import joblib

    from src.simulation.monte_carlo import load_tournament_state, run_monte_carlo
    predictor = joblib.load(PROCESSED_DIR / "calibrator.joblib")
    state = load_tournament_state()
    sim = run_monte_carlo(predictor, n=n_sims, seed=seed, state=state)
    sim.to_parquet(PROCESSED_DIR / "sim_results.parquet", index=False)

    summary = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "matches_total": ingestion["matches"],
        "matches_played": ingestion["played"],
        "overlay": ingestion["overlay"],
        "brier_holdout": training["best_brier"],
        "gate_passed": training["gate_passed"],
        "n_group_played": state["n_group_played"],
        "n_ko_played": state["n_ko_played"],
        "n_sims": n_sims,
        "top5": sim.head(5)[["team", "champion_pct"]].to_dict("records"),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LAST_REFRESH.write_text(json.dumps(summary, indent=1, default=str),
                            encoding="utf-8")
    return summary


def get_last_refresh() -> dict | None:
    if LAST_REFRESH.exists():
        return json.loads(LAST_REFRESH.read_text(encoding="utf-8"))
    return None

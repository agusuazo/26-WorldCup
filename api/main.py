"""Backend FastAPI para el frontend WC26 Quant (Lovable).

Expone el contrato definido en docs/PROMPT_LOVABLE.md (claves camelCase).
Arrancar:  uvicorn api.main:app --port 8000   (desde la raíz del proyecto)
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import (BRIER_GATE, DB_PATH, EV_THRESHOLD_MED,
                             KELLY_CAP, KELLY_FRACTION, PROCESSED_DIR,
                             WC2026_FIXTURES)
from src.betting.bankroll import fractional_kelly
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import overround, remove_vig_multiplicative
from src.simulation.monte_carlo import (load_tournament_state, run_monte_carlo,
                                        simulate_tournament_detail)

app = FastAPI(title="WC26 Quant API")

# ---- Configuración de despliegue ------------------------------------------
# API_ACCESS_KEY: si está definida, todas las rutas /api/* exigen el header
#                 X-API-Key (clave compartida con los usuarios autorizados).
# SERVER_LIGHT=1: modo servidor free-tier — el refresh NO re-entrena modelos
#                 (eso se hace en el PC del admin + git push); solo re-simula
#                 condicionado a los resultados nuevos.
API_ACCESS_KEY = os.environ.get("API_ACCESS_KEY", "")
SERVER_LIGHT = os.environ.get("SERVER_LIGHT", "") == "1"


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if (API_ACCESS_KEY and request.method != "OPTIONS"
            and request.url.path.startswith("/api")
            and request.headers.get("x-api-key") != API_ACCESS_KEY):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


# CORS se añade DESPUÉS del auth middleware para quedar más externo:
# así los 401 también llevan headers CORS y el navegador puede leerlos.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/")
def health():
    """Health check (sin auth) para Render."""
    return {"ok": True, "service": "wc26-quant-api"}

# ---- Modelos cargados una vez (se recargan tras un refresh) ---------------
_MODELS: dict = {}
_CACHE: dict = {}          # caché en memoria: simulation / bracket
_REFRESH = {"stage": "idle", "done": True, "error": None}


def _load_models():
    from src.models.elo_model import EloPredictor
    calibrator = joblib.load(PROCESSED_DIR / "calibrator.joblib")
    poisson = joblib.load(PROCESSED_DIR / "poisson_model.joblib")
    draw = joblib.load(PROCESSED_DIR / "draw_model.joblib")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        ratings = dict(con.execute("SELECT team, elo FROM elo_current").fetchall())
    finally:
        con.close()
    _MODELS.update({
        "calibrator": calibrator,
        "poisson": poisson,
        "elo": EloPredictor(ratings, draw),
        "ratings": ratings,
    })
    _CACHE.clear()


_load_models()

# En modo remoto (Supabase), garantizar el esquema de tablas mutables al boot.
# No debe bloquear el arranque del server (Render mide el bind del puerto):
# si Supabase está inalcanzable, seguimos arrancando y que falle en la request.
from src.db import remote as _remote
if _remote.is_remote():
    try:
        _remote.ensure_schema()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] ensure_schema() failed, continuing without it: {exc}",
              file=sys.stderr)


def _groups() -> dict:
    return json.loads(WC2026_FIXTURES.read_text(encoding="utf-8"))["groups"]


# ---- Status ----------------------------------------------------------------

@app.get("/api/status")
def status():
    from src.ingestion.updater import get_last_refresh
    last = get_last_refresh()
    state = load_tournament_state()
    brier = last["brier_holdout"] if last else None
    if brier is None:
        bt = PROCESSED_DIR / "backtest_results.parquet"
        brier = float(pd.read_parquet(bt)["brier_score"].iloc[0]) if bt.exists() else 0.2222
    return {
        "lastRefresh": last["refreshed_at"] if last else None,
        "brierHoldout": float(brier),
        "gatePassed": float(brier) < BRIER_GATE,
        "groupMatchesPlayed": state["n_group_played"],
        "koMatchesPlayed": state["n_ko_played"],
    }


# ---- Teams -------------------------------------------------------------------

@app.get("/api/teams")
def teams():
    groups = _groups()
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    ratings = _MODELS["ratings"]
    rows = [{"name": t, "group": g, "elo": round(ratings.get(t, 1500.0))}
            for t, g in team_to_group.items()]
    return sorted(rows, key=lambda r: -r["elo"])


# ---- Upcoming matches ----------------------------------------------------------

@app.get("/api/matches/upcoming")
def upcoming():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT match_id, date, home_team, away_team FROM matches
            WHERE tournament = 'FIFA World Cup' AND home_score IS NULL
              AND date >= current_date
            ORDER BY date LIMIT 20
        """).df()
    finally:
        con.close()

    odds_df = None
    try:
        from src.ingestion.odds_api import best_odds_per_match, load_latest_odds
        raw = load_latest_odds()
        if raw is not None and not raw.empty:
            odds_df = best_odds_per_match(raw)
    except Exception:
        pass

    cal = _MODELS["calibrator"]
    out = []
    for m in df.itertuples():
        try:
            ph, pd_, pa = cal.predict_proba(m.home_team, m.away_team, True)
        except Exception:
            continue
        row = {
            "matchId": int(m.match_id),
            "date": pd.Timestamp(m.date).date().isoformat(),
            "home": m.home_team, "away": m.away_team,
            "probs": {"home": ph, "draw": pd_, "away": pa},
            "bestEv": None, "bestBet": None, "signal": None,
        }
        if odds_df is not None:
            mo = odds_df[(odds_df["home_team"] == m.home_team)
                         & (odds_df["away_team"] == m.away_team)]
            if not mo.empty:
                mo = mo.iloc[0]
                best_ev, best_label = -1.0, None
                for p, o, lbl in [(ph, mo["home_odds"], m.home_team),
                                  (pd_, mo["draw_odds"], "Empate"),
                                  (pa, mo["away_odds"], m.away_team)]:
                    if o and o > 1:
                        ev = expected_value(p, float(o))
                        if ev > best_ev:
                            best_ev, best_label = ev, f"{lbl} @{float(o):.2f}"
                row.update({"bestEv": best_ev, "bestBet": best_label,
                            "signal": classify_ev(best_ev)})
        out.append(row)
    return out


# ---- Predict / EV ---------------------------------------------------------------

class PredictBody(BaseModel):
    home: str
    away: str
    neutral: bool = True


@app.post("/api/predict")
def predict(body: PredictBody):
    cal, elo, pois = _MODELS["calibrator"], _MODELS["elo"], _MODELS["poisson"]
    ratings = _MODELS["ratings"]
    if body.home == body.away:
        raise HTTPException(422, "Equipos iguales")
    probs = cal.predict_proba(body.home, body.away, body.neutral)
    lh, la = cal.predict_lambdas(body.home, body.away, body.neutral)
    rh = ratings.get(body.home, 1500.0)
    ra = ratings.get(body.away, 1500.0)
    matrix = pois.score_matrix(lh, la, max_goals=6)
    p_elo = elo.predict_proba(body.home, body.away, neutral=body.neutral)
    p_pois = pois.predict_proba(rh, ra, neutral=body.neutral)
    return {
        "probs": {"home": probs[0], "draw": probs[1], "away": probs[2]},
        "lambdas": {"home": lh, "away": la},
        "elo": {"home": round(rh), "away": round(ra)},
        "scoreMatrix": np.asarray(matrix).tolist(),
        "byModel": [
            {"name": "Elo + draw model", "home": p_elo[0], "draw": p_elo[1], "away": p_elo[2]},
            {"name": "Poisson global", "home": p_pois[0], "draw": p_pois[1], "away": p_pois[2]},
            {"name": "Ensemble calibrado", "home": probs[0], "draw": probs[1], "away": probs[2]},
        ],
    }


class EvBody(BaseModel):
    home: str
    away: str
    neutral: bool = True
    odds: dict
    bankroll: float = 1000.0


@app.post("/api/ev")
def ev_analysis(body: EvBody):
    cal = _MODELS["calibrator"]
    probs = cal.predict_proba(body.home, body.away, body.neutral)
    odds = [float(body.odds["home"]), float(body.odds["draw"]), float(body.odds["away"])]
    if any(o <= 1.0 for o in odds):
        raise HTTPException(422, "Cuotas deben ser > 1.0")
    implied = remove_vig_multiplicative(odds)
    labels = ["Local", "Empate", "Visitante"]
    rows = []
    for p, o, imp, lbl in zip(probs, odds, implied, labels):
        ev = expected_value(p, o)
        kf = fractional_kelly(p, o, KELLY_FRACTION, KELLY_CAP)
        rows.append({"outcome": lbl, "modelProb": p, "impliedProb": imp,
                     "edge": p - imp, "ev": ev, "signal": classify_ev(ev),
                     "kellyStake": body.bankroll * kf if ev > 0 else 0.0,
                     "kellyPct": kf})
    best = max(rows, key=lambda r: r["ev"])
    if best["ev"] >= EV_THRESHOLD_MED:
        action, reason = "BET", (f"EV {best['ev']:+.1%} supera el umbral del 5%. "
                                 f"Stake Kelly 1/8 con cap 5%.")
    elif best["ev"] > 0:
        action, reason = "MARGINAL", ("EV positivo pero por debajo del 5% — con edges "
                                      "tan finos el error del modelo puede comerse el valor.")
    else:
        action, reason = "PASS", ("Ninguna cuota ofrece valor. Pasar también es "
                                  "una decisión correcta.")
    best_i = rows.index(best)
    return {
        "rows": [{k: v for k, v in r.items() if k != "kellyPct"} for r in rows],
        "recommendation": {
            "action": action,
            "outcome": labels[best_i] if action != "PASS" else None,
            "odds": odds[best_i] if action != "PASS" else None,
            "ev": best["ev"] if action != "PASS" else None,
            "stake": best["kellyStake"] if action != "PASS" else None,
            "kellyPct": best["kellyPct"] if action != "PASS" else None,
            "reason": reason,
        },
        "overround": overround(odds) - 1.0,
    }


# ---- Simulation / Bracket ---------------------------------------------------------

@app.get("/api/simulation")
def simulation(conditioned: bool = True):
    key = ("sim", conditioned)
    if key in _CACHE:
        return _CACHE[key]
    parquet = PROCESSED_DIR / "sim_results.parquet"
    if conditioned and parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        state = load_tournament_state() if conditioned else None
        df = run_monte_carlo(_MODELS["calibrator"], n=5000, seed=42, state=state)
    groups = _groups()
    team_to_group = {t: g for g, ts in groups.items() for t in ts}
    out = [{
        "rank": int(r.rank), "team": r.team,
        "group": team_to_group.get(r.team, "?"),
        "championPct": float(r.champion_pct), "finalistPct": float(r.finalist_pct),
        "sfPct": float(r.sf_pct), "qfPct": float(r.qf_pct),
        "r16Pct": float(r.r16_pct), "r32Pct": float(r.r32_pct),
        "groupExitPct": float(r.group_exit_pct),
    } for r in df.itertuples()]
    _CACHE[key] = out
    return out


@app.get("/api/bracket")
def bracket(mode: str = "expected", seed: int = 42, conditioned: bool = True):
    key = ("bracket", mode, seed, conditioned)
    if key in _CACHE:
        return _CACHE[key]
    state = load_tournament_state() if conditioned else None
    d = simulate_tournament_detail(_MODELS["calibrator"], seed=seed,
                                   mode=mode, state=state)
    played = d.get("played_pairs", set())
    out = {
        "mode": d["mode"], "seed": seed, "champion": d["champion"],
        "groups": {
            g: {
                "standings": [{"team": t, "pts": round(float(p), 2),
                               "gd": round(float(gd), 2), "gf": round(float(gf), 2)}
                              for t, p, gd, gf in info["standings"]],
                "qualified": info["qualified"],
            } for g, info in d["groups"].items()
        },
        "rounds": {
            rname: [{"a": a, "b": b, "winner": w,
                     "isReal": frozenset((a, b)) in played}
                    for a, b, w in matches]
            for rname, matches in d["rounds"].items()
        },
    }
    _CACHE[key] = out
    return out


# ---- Backtest ----------------------------------------------------------------------

@app.get("/api/backtest")
def backtest():
    summary_p = PROCESSED_DIR / "backtest_results.parquet"
    if not summary_p.exists():
        raise HTTPException(404, "Sin backtest. Ejecuta scripts/run_backtest.py")
    s = pd.read_parquet(summary_p).iloc[0]
    by_year = pd.read_parquet(PROCESSED_DIR / "backtest_by_year.parquet")
    cal = pd.read_parquet(PROCESSED_DIR / "calibration_curve.parquet")
    return {
        "brier": float(s["brier_score"]), "logLoss": float(s["log_loss"]),
        "nMatches": int(s["n_matches"]),
        "byYear": [{"year": int(r.year), "brier": float(r.brier), "n": int(r.n)}
                   for r in by_year.itertuples()],
        "calibration": [{"meanPred": float(r.mean_pred),
                         "actualFreq": float(r.actual_freq), "count": int(r.count)}
                        for r in cal.itertuples()],
    }


# ---- Timeline / resultados / refresh --------------------------------------------------

@app.get("/api/timeline")
def timeline():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT match_id, date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup' AND date >= '2026-06-01'
            ORDER BY date
        """).df()
    finally:
        con.close()

    def stage_of(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d)
        if d <= pd.Timestamp("2026-06-27"):
            return "group"
        if d <= pd.Timestamp("2026-07-03"):
            return "r32"
        if d <= pd.Timestamp("2026-07-07"):
            return "r16"
        if d <= pd.Timestamp("2026-07-11"):
            return "qf"
        if d <= pd.Timestamp("2026-07-15"):
            return "sf"
        return "final"

    return [{
        "matchId": int(r.match_id),
        "date": pd.Timestamp(r.date).date().isoformat(),
        "home": r.home_team, "away": r.away_team,
        "homeScore": None if pd.isna(r.home_score) else int(r.home_score),
        "awayScore": None if pd.isna(r.away_score) else int(r.away_score),
        "stage": stage_of(r.date),
        "played": not pd.isna(r.home_score),
    } for r in df.itertuples()]


class ResultBody(BaseModel):
    date: str
    home: str
    away: str
    homeScore: int
    awayScore: int
    winner: str | None = None


@app.post("/api/results")
def post_result(body: ResultBody):
    from src.ingestion.updater import save_manual_result
    save_manual_result(body.date, body.home, body.away,
                       body.homeScore, body.awayScore, winner=body.winner)
    _CACHE.clear()
    return {"ok": True}


@app.post("/api/data/download")
def download():
    if SERVER_LIGHT:
        raise HTTPException(
            400, "En el servidor free-tier la descarga del dataset y el "
                 "re-entrenamiento se hacen desde el PC del admin "
                 "(scripts/publish.ps1). Usa 'Recalcular' para re-simular "
                 "con los resultados manuales nuevos.")
    from src.ingestion.updater import download_latest_results
    info = download_latest_results()
    return {"rowsNew": info["rows_new"]}


def _refresh_worker():
    try:
        if SERVER_LIGHT:
            # Free tier: sin re-entrenamiento (CPU/RAM insuficientes).
            # Re-simula condicionado a los resultados nuevos de Supabase.
            _REFRESH["stage"] = "simulacion"
            state = load_tournament_state()
            sim = run_monte_carlo(_MODELS["calibrator"], n=5000, seed=42,
                                  state=state)
            sim.to_parquet(PROCESSED_DIR / "sim_results.parquet", index=False)
            _CACHE.clear()
        else:
            from src.ingestion.updater import refresh_all

            def cb(stage):
                _REFRESH["stage"] = stage

            refresh_all(n_sims=10_000, progress_cb=cb)
            _load_models()
        _REFRESH.update({"stage": "done", "done": True, "error": None})
    except Exception as e:
        _REFRESH.update({"stage": "error", "done": True, "error": str(e)})


@app.post("/api/refresh")
def refresh():
    if not _REFRESH["done"]:
        return {"jobId": "running"}
    _REFRESH.update({"stage": "ingesta", "done": False, "error": None})
    threading.Thread(target=_refresh_worker, daemon=True).start()
    return {"jobId": "refresh-1"}


@app.get("/api/refresh/status")
def refresh_status():
    return {"stage": _REFRESH["stage"], "done": _REFRESH["done"],
            "error": _REFRESH["error"]}

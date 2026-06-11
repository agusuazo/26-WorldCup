"""Entrena todos los modelos y valida en hold-out.

Orden: Elo (ya en DB) → draw model → Poisson → Dixon-Coles → Ensemble.
Gate de calidad: Brier multiclase < BRIER_GATE sobre hold-out 2023-2026.

Uso: python scripts/run_training.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import duckdb
import joblib
import numpy as np
from sklearn.metrics import log_loss

from config.settings import (BRIER_GATE, DB_PATH, ELO_WARMUP_END,
                             HOLDOUT_START, PROCESSED_DIR, TRAIN_END)
from src.models.calibration import MultinomialCalibrator
from src.models.dixon_coles import DixonColesModel
from src.models.elo_model import EloPredictor, fit_draw_model
from src.models.ensemble import EnsemblePredictor
from src.models.poisson_model import GlobalPoissonModel


def load_played_matches():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT m.match_id, m.date, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.tournament,
                   m.neutral, m.k_weight,
                   f.home_elo AS home_elo_pre, f.away_elo AS away_elo_pre
            FROM matches m JOIN match_features f USING (match_id)
            WHERE m.home_score IS NOT NULL AND m.date >= ?
            ORDER BY m.date
        """, [ELO_WARMUP_END]).df()
    finally:
        con.close()
    return df


def outcome_onehot(df):
    gd = df["home_score"] - df["away_score"]
    y = np.zeros((len(df), 3))
    y[gd > 0, 0] = 1
    y[gd == 0, 1] = 1
    y[gd < 0, 2] = 1
    return y


def brier_mc(probs, y_oh):
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)) / 3.0)


def train_all(verbose: bool = True) -> dict:
    """Entrena todos los modelos, evalúa en hold-out y persiste los joblib.

    Importable desde el updater (refresh_all). Devuelve dict con
    brier/log_loss por modelo y el resultado del gate.
    """
    t0 = time.time()
    df = load_played_matches()
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] < HOLDOUT_START)]
    hold = df[df["date"] >= HOLDOUT_START]
    print(f"Train   : {len(train):,}  ({train.date.min().date()} -> {train.date.max().date()})")
    print(f"Val     : {len(val):,}   ({val.date.min().date()} -> {val.date.max().date()})")
    print(f"Hold-out: {len(hold):,}  ({hold.date.min().date()} -> {hold.date.max().date()})\n")

    # ---- Elo + draw model ----
    print("Ajustando draw model (Elo)...")
    draw_model = fit_draw_model(train)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ratings = dict(con.execute("SELECT team, elo FROM elo_current").fetchall())
    con.close()
    elo_pred = EloPredictor(ratings, draw_model)

    # ---- Poisson global ----
    print("Ajustando Poisson global...")
    poisson_model = GlobalPoissonModel().fit(train)

    # ---- Dixon-Coles ----
    print("Ajustando Dixon-Coles (puede tardar 1-3 min)...")
    t_dc = time.time()
    dc_model = DixonColesModel(xi=0.0065, min_date="2010-01-01", reg=0.02)
    dc_model.fit(train)
    print(f"  Dixon-Coles listo en {time.time()-t_dc:.1f}s  rho={dc_model.rho_:.4f}")

    # ---- Ensemble ----
    print("Optimizando pesos del ensemble...")
    ensemble = EnsemblePredictor(elo_pred, poisson_model, dc_model)
    if len(val) >= 100:
        ensemble.fit_weights(val)
    print(f"  Pesos: Elo={ensemble.weights_[0]:.2f}  "
          f"Poisson={ensemble.weights_[1]:.2f}  DC={ensemble.weights_[2]:.2f}")

    # ---- Calibración ----
    print("Calibrando ensemble (multinomial)...")
    calibrator = MultinomialCalibrator()
    calibrator.fit(ensemble, val if len(val) >= 100 else train.tail(1000))

    # ---- Evaluación en hold-out ----
    y_oh = outcome_onehot(hold)
    y_lbl = y_oh.argmax(axis=1)
    rows_eval = []

    for name, predictor, use_team_names in [
        ("Blend 50/50 (Sprint0)", None, False),
        ("Ensemble (DC+weights)", ensemble, True),
        ("Ensemble calibrado", calibrator, True),
    ]:
        if name == "Blend 50/50 (Sprint0)":
            pe = elo_pred.predict_proba_batch(
                hold["home_elo_pre"].to_numpy(), hold["away_elo_pre"].to_numpy(),
                hold["neutral"].to_numpy(), hold["k_weight"].to_numpy())
            pp = np.array([poisson_model.predict_proba(r.home_elo_pre, r.away_elo_pre,
                                                        r.neutral)
                           for r in hold.itertuples()])
            probs = 0.5 * pe + 0.5 * pp
        else:
            probs = np.array([predictor.predict_proba(r.home_team, r.away_team,
                                                       bool(r.neutral))
                              for r in hold.itertuples()])
        b = brier_mc(probs, y_oh)
        ll = log_loss(y_lbl, probs, labels=[0, 1, 2])
        rows_eval.append((name, b, ll))

    print(f"\n{'Modelo':<30} {'Brier':>7} {'LogLoss':>9}")
    print("-" * 50)
    for name, b, ll in rows_eval:
        print(f"{name:<30} {b:>7.4f} {ll:>9.4f}")

    best_brier = min(r[1] for r in rows_eval[1:])
    gate = "PASA" if best_brier < BRIER_GATE else "NO PASA"
    print(f"\nGate Brier < {BRIER_GATE}: {gate}  (mejor: {best_brier:.4f})")
    print(f"Tiempo total: {time.time()-t0:.0f}s\n")

    # ---- Persistir ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(draw_model,    PROCESSED_DIR / "draw_model.joblib")
    joblib.dump(poisson_model, PROCESSED_DIR / "poisson_model.joblib")
    joblib.dump(dc_model,      PROCESSED_DIR / "dc_model.joblib")
    joblib.dump(ensemble,      PROCESSED_DIR / "ensemble.joblib")
    joblib.dump(calibrator,    PROCESSED_DIR / "calibrator.joblib")
    print(f"Modelos guardados en {PROCESSED_DIR}")

    return {
        "models": {name: {"brier": b, "log_loss": ll} for name, b, ll in rows_eval},
        "best_brier": best_brier,
        "gate_passed": best_brier < BRIER_GATE,
        "n_train": len(train),
        "n_holdout": len(hold),
        "elapsed_s": time.time() - t0,
    }


def main():
    train_all()


if __name__ == "__main__":
    main()

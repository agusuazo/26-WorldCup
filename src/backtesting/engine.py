"""Motor de backtesting walk-forward temporal.

Principio: no usar datos futuros para predecir el pasado.
El predictor se re-entrena en ventanas deslizantes; las predicciones
solo se hacen sobre datos más recientes que el corte de entrenamiento.

Para el backtest exprés de Sprint 1 (sin cuotas históricas para WC)
se usa la función `backtest_probability_quality()` que solo necesita
predicciones vs resultados reales — sin cuotas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.metrics import (brier_multiclass, calibration_bins,
                                     profit_factor, roi, sharpe_ratio,
                                     summarize)


# ---- Backtest de calidad probabilística (no requiere cuotas) ----------

def backtest_probability_quality(predictor,
                                 df: pd.DataFrame,
                                 start: str | None = None,
                                 end: str | None = None,
                                 batch_size: int = 1000) -> dict:
    """Evalúa la calidad probabilística del predictor sobre un periodo.

    Requiere columnas: date, home_team, away_team, neutral, home_score,
    away_score, tournament.
    Devuelve Brier score, log loss, curva de calibración y resumen por año.
    """
    df = df[df["home_score"].notna()].copy()
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    if df.empty:
        return {"error": "sin datos en ese periodo"}

    df = df.sort_values("date")
    probs_list = []
    outcomes = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        for row in batch.itertuples():
            try:
                p = predictor.predict_proba(row.home_team, row.away_team,
                                            bool(row.neutral))
                probs_list.append(p)
            except Exception:
                probs_list.append((1/3, 1/3, 1/3))
            gd = row.home_score - row.away_score
            outcomes.append(0 if gd > 0 else (1 if gd == 0 else 2))

    probs = np.array(probs_list)
    outcomes = np.array(outcomes)

    brier = brier_multiclass(probs, outcomes)
    from sklearn.metrics import log_loss
    ll = float(log_loss(outcomes, probs, labels=[0, 1, 2]))
    cal = calibration_bins(probs[:, 0], (outcomes == 0).astype(float))

    # Brier por año
    df["year"] = pd.to_datetime(df["date"]).dt.year
    yearly = []
    idx = 0
    for yr, grp in df.groupby("year"):
        n = len(grp)
        y_probs = probs[idx : idx + n]
        y_out = outcomes[idx : idx + n]
        yearly.append({"year": int(yr), "n": n, "brier": brier_multiclass(y_probs, y_out)})
        idx += n

    return {
        "n_matches": len(df),
        "brier_score": brier,
        "log_loss": ll,
        "calibration": cal,
        "by_year": pd.DataFrame(yearly),
        "probs": probs,
        "outcomes": outcomes,
    }


# ---- Walk-forward temporal ---------------------------------------------

def walk_forward_backtest(df: pd.DataFrame,
                          start: str = "2018-01-01",
                          end: str = "2025-12-31",
                          step_months: int = 3,
                          weights: tuple[float, float, float] = (0.33, 0.34, 0.33),
                          verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Backtest walk-forward: re-entrena los modelos cada `step_months` meses
    usando solo datos anteriores al fold, y predice el trimestre siguiente.

    Sin leakage por construcción:
    - Elo: usa home_elo_pre/away_elo_pre (snapshot pre-partido del feature store)
    - Draw model y Poisson: fit con partidos date < fold_start
    - Dixon-Coles: fit con date < fold_start y time-decay anclado en fold_start

    `df` requiere: date, home_team, away_team, home_score, away_score,
    neutral, k_weight, home_elo_pre, away_elo_pre.
    Devuelve (df_folds, dict_global con brier/log_loss agregados).
    """
    from sklearn.metrics import log_loss

    from src.models.dixon_coles import DixonColesModel
    from src.models.elo_model import fit_draw_model
    from src.models.poisson_model import GlobalPoissonModel

    df = df[df["home_score"].notna()].sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    fold_starts = pd.date_range(start=start, end=end,
                                freq=pd.DateOffset(months=step_months))
    all_probs, all_outcomes, fold_rows = [], [], []

    for fold_start in fold_starts:
        fold_end = fold_start + pd.DateOffset(months=step_months)
        train = df[df["date"] < fold_start]
        test = df[(df["date"] >= fold_start) & (df["date"] < fold_end)]
        if len(test) == 0 or len(train) < 2000:
            continue

        draw_model = fit_draw_model(train)
        poisson = GlobalPoissonModel().fit(train)
        dc = DixonColesModel(reg=0.02).fit(train, ref_date=fold_start)

        from src.models.elo_model import EloPredictor
        elo_stub = EloPredictor({}, draw_model)

        probs = []
        outcomes = []
        for row in test.itertuples():
            pe = np.array(elo_stub.predict_from_elo(
                row.home_elo_pre, row.away_elo_pre,
                bool(row.neutral), row.k_weight))
            pp = np.array(poisson.predict_proba(
                row.home_elo_pre, row.away_elo_pre, bool(row.neutral)))
            pdc_raw = dc.predict_proba(row.home_team, row.away_team,
                                       bool(row.neutral))
            pdc = np.array(pdc_raw) if pdc_raw is not None else pe
            blend = weights[0]*pe + weights[1]*pp + weights[2]*pdc
            blend = blend / blend.sum()
            probs.append(blend)
            gd = row.home_score - row.away_score
            outcomes.append(0 if gd > 0 else (1 if gd == 0 else 2))

        probs = np.array(probs)
        outcomes = np.array(outcomes)
        fold_brier = brier_multiclass(probs, outcomes)
        fold_rows.append({
            "fold_start": fold_start.date(),
            "fold_end": fold_end.date(),
            "n_train": len(train),
            "n_test": len(test),
            "brier": fold_brier,
        })
        all_probs.append(probs)
        all_outcomes.append(outcomes)
        if verbose:
            print(f"  {fold_start.date()} -> {fold_end.date()}: "
                  f"brier={fold_brier:.4f}  (n={len(test)})")

    if not fold_rows:
        return pd.DataFrame(), {}

    probs_total = np.vstack(all_probs)
    outcomes_total = np.concatenate(all_outcomes)
    summary = {
        "n_folds": len(fold_rows),
        "n_matches": len(outcomes_total),
        "brier_score": brier_multiclass(probs_total, outcomes_total),
        "log_loss": float(log_loss(outcomes_total, probs_total, labels=[0, 1, 2])),
    }
    return pd.DataFrame(fold_rows), summary


# ---- Backtest con cuotas (simulación de apuestas) ---------------------

def backtest_bets(predictor,
                  df: pd.DataFrame,
                  ev_threshold: float = 0.0,
                  kelly_fraction: float = 0.125,
                  kelly_cap: float = 0.05,
                  initial_bankroll: float = 1000.0) -> tuple[pd.DataFrame, dict]:
    """Simula apuestas siguiendo la estrategia EV+ + Fractional Kelly.

    df requiere: date, home_team, away_team, neutral, home_score, away_score,
    home_odds, draw_odds, away_odds, tournament.
    Devuelve (bet_log_df, summary_dict).
    """
    from src.betting.ev_calculator import expected_value
    from src.betting.odds_parser import remove_vig_multiplicative

    df = df[df["home_score"].notna() & df["home_odds"].notna()].copy()
    df = df.sort_values("date")

    bankroll = initial_bankroll
    bet_rows = []

    for row in df.itertuples():
        try:
            p_home, p_draw, p_away = predictor.predict_proba(
                row.home_team, row.away_team, bool(row.neutral))
        except Exception:
            continue

        odds_list = [row.home_odds, row.draw_odds, row.away_odds]
        if any(o <= 1.0 for o in odds_list):
            continue

        for outcome_idx, (model_p, odds, label) in enumerate(
                zip([p_home, p_draw, p_away], odds_list, ["home", "draw", "away"])):
            ev = expected_value(model_p, odds)
            if ev <= ev_threshold:
                continue

            from src.betting.bankroll import fractional_kelly
            frac = fractional_kelly(model_p, odds, kelly_fraction, kelly_cap)
            stake = bankroll * frac
            if stake < 0.01:
                continue

            gd = row.home_score - row.away_score
            actual = 0 if gd > 0 else (1 if gd == 0 else 2)
            won = actual == outcome_idx
            profit = stake * (odds - 1) if won else -stake
            bankroll += profit

            bet_rows.append({
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "bet": label,
                "odds": odds,
                "model_prob": model_p,
                "ev": ev,
                "stake": stake,
                "result": "win" if won else "lose",
                "profit": profit,
                "bankroll_after": bankroll,
            })

    if not bet_rows:
        return pd.DataFrame(), {}

    bet_df = pd.DataFrame(bet_rows)
    summary = summarize(bet_df, initial_bankroll)
    equity = np.concatenate([[initial_bankroll],
                              initial_bankroll + np.cumsum(bet_df["profit"].to_numpy())])
    summary["equity_curve"] = equity
    return bet_df, summary

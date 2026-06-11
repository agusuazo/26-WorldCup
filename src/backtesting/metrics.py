"""Métricas de rendimiento para sistemas de apuestas deportivas."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- Métricas de rentabilidad -----------------------------------------

def roi(stakes: np.ndarray, profits: np.ndarray) -> float:
    """ROI = ganancias_netas / stake_total. Devuelve float (ej. 0.05 = 5%)."""
    total_stake = stakes.sum()
    if total_stake == 0:
        return 0.0
    return float(profits.sum() / total_stake)


def yield_rate(profits: np.ndarray) -> float:
    """Yield = media de profit por apuesta (normalizado por cantidad de bets)."""
    if len(profits) == 0:
        return 0.0
    return float(profits.mean())


def profit_factor(profits: np.ndarray) -> float:
    """Profit Factor = suma_ganancias / abs(suma_pérdidas). >1 es positivo."""
    gains = profits[profits > 0].sum()
    losses = abs(profits[profits < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


# ---- Métricas de riesgo -----------------------------------------------

def max_drawdown(equity_curve: np.ndarray) -> float:
    """Peor caída absoluta desde un máximo anterior en la curva de equity."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peak
    return float(drawdown.min())


def max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Max drawdown como porcentaje del pico."""
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = np.where(peak > 0, (equity_curve - peak) / peak, 0.0)
    return float(dd_pct.min())


def sharpe_ratio(profits: np.ndarray, risk_free: float = 0.0) -> float:
    """Sharpe ratio sobre la serie de profits por apuesta."""
    if len(profits) < 2:
        return 0.0
    excess = profits - risk_free
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(len(profits)))


# ---- Métricas probabilísticas ----------------------------------------

def brier_multiclass(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier Score multiclase (3 resultados). Baseline uniforme = 0.2222.

    probs: (N, 3) probabilidades predichas [home, draw, away]
    outcomes: (N,) entero {0=home, 1=draw, 2=away}
    """
    n = len(outcomes)
    y_oh = np.zeros((n, 3))
    y_oh[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)) / 3.0)


def calibration_bins(probs: np.ndarray, outcomes: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibración: agrupa predicciones por decil y compara con frecuencia real."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": (lo + hi) / 2,
            "mean_pred": probs[mask].mean(),
            "actual_freq": outcomes[mask].mean(),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ---- Resumen completo -------------------------------------------------

def summarize(df_bets: pd.DataFrame, initial_bankroll: float = 1000.0) -> dict:
    """Resumen completo de una serie de apuestas.

    df_bets requiere: stake, odds, result ('win'/'lose'), profit, model_prob.
    """
    if df_bets.empty:
        return {}

    stakes = df_bets["stake"].to_numpy()
    profits = df_bets["profit"].to_numpy()
    equity = np.concatenate([[initial_bankroll], initial_bankroll + np.cumsum(profits)])

    return {
        "n_bets": len(df_bets),
        "n_won": int((df_bets["result"] == "win").sum()),
        "win_rate": float((df_bets["result"] == "win").mean()),
        "total_staked": float(stakes.sum()),
        "net_profit": float(profits.sum()),
        "roi": roi(stakes, profits),
        "yield": yield_rate(profits),
        "profit_factor": profit_factor(profits),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "sharpe": sharpe_ratio(profits),
        "avg_odds": float(df_bets["odds"].mean()),
        "avg_ev": float((df_bets["model_prob"] * df_bets["odds"] - 1).mean()),
    }

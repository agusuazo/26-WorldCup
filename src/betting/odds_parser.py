"""Conversión de cuotas decimales a probabilidades implícitas, con remoción de vig.

- Multiplicativa: aceptable para 1X2 (overround 4-8%).
- Potencia y Shin: obligatorias para mercados outright (overround 20-40%,
  sesgo longshot — la multiplicativa infla los longshots y genera falsos EV+).
"""
import numpy as np
from scipy.optimize import brentq


def implied_prob_raw(odds: np.ndarray | list) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> float:
    """Suma de probabilidades crudas. >1 = margen de la casa."""
    return float(implied_prob_raw(odds).sum())


def remove_vig_multiplicative(odds) -> np.ndarray:
    p = implied_prob_raw(odds)
    return p / p.sum()


def remove_vig_power(odds) -> np.ndarray:
    """p_i = (1/odds_i)^k con k tal que sumen 1. Corrige parcialmente longshot bias."""
    p = implied_prob_raw(odds)

    def excess(k):
        return np.power(p, k).sum() - 1.0

    k = brentq(excess, 0.05, 20.0)
    return np.power(p, k)


def remove_vig_shin(odds, max_z: float = 0.5) -> np.ndarray:
    """Método de Shin (1992): modela la proporción z de apostadores informados.

    El estándar académico para mercados con muchos resultados y longshot bias
    (outright de campeón). Resuelve z tal que las probabilidades sumen 1.
    Requiere overround (suma de implícitas > 1); si no lo hay, normaliza.
    """
    p = implied_prob_raw(odds)
    total = p.sum()
    if total <= 1.0:  # sin margen que remover (mercado sintético/incompleto)
        return p / total

    def shin_probs(z):
        return (np.sqrt(z**2 + 4.0 * (1.0 - z) * p**2 / total) - z) / (2.0 * (1.0 - z))

    def excess(z):
        return shin_probs(z).sum() - 1.0

    if excess(max_z) > 0.0:  # overround extremo fuera de rango: degradar
        return p / total
    z = brentq(excess, 1e-12, max_z)
    out = shin_probs(z)
    return out / out.sum()

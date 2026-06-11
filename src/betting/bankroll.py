"""Gestión de banca: Flat, Kelly completo y Kelly fraccional con cap."""
from config.settings import KELLY_CAP, KELLY_FRACTION


def flat_stake(bankroll: float, pct: float = 0.01) -> float:
    """Apuesta plana: siempre el mismo % del bankroll inicial."""
    return bankroll * pct


def full_kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """f* = (b·p - q) / b. Nunca negativo (no apostar si no hay edge)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * model_prob - (1.0 - model_prob)) / b
    return max(f, 0.0)


def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = KELLY_FRACTION,
                     cap: float = KELLY_CAP) -> float:
    """Kelly fraccional con hard cap. Mitiga la miscalibración del modelo:
    Kelly completo es catastrófico si las probabilidades están infladas."""
    return min(fraction * full_kelly_fraction(model_prob, decimal_odds), cap)


def kelly_stake(bankroll: float, model_prob: float, decimal_odds: float,
                fraction: float = KELLY_FRACTION, cap: float = KELLY_CAP) -> float:
    return bankroll * fractional_kelly(model_prob, decimal_odds, fraction, cap)

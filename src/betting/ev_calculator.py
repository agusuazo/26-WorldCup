"""Detección de valor esperado positivo (EV+)."""
from config.settings import (EV_THRESHOLD_HIGH, EV_THRESHOLD_LOW,
                             EV_THRESHOLD_MED, EV_THRESHOLD_OUTRIGHT)


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV por unidad apostada: EV = p·cuota - 1."""
    return model_prob * decimal_odds - 1.0


def classify_ev(ev: float, outright: bool = False) -> str:
    """Clasifica la oportunidad. Los mercados outright exigen umbral mayor
    porque el error del modelo se compone a través de ~7 rondas simuladas."""
    if outright:
        return "HIGH" if ev > EV_THRESHOLD_OUTRIGHT else "NONE"
    if ev > EV_THRESHOLD_HIGH:
        return "HIGH"
    if ev > EV_THRESHOLD_MED:
        return "MEDIUM"
    if ev > EV_THRESHOLD_LOW:
        return "LOW"
    return "NONE"

"""Modelo Elo propio: forward pass cronológico + conversión a probabilidades 1X2.

Metodología World Football Elo (eloratings.net):
- K-factor según importancia del torneo (config.settings.k_factor)
- Multiplicador por diferencia de goles
- Ventaja de localía: +ELO_HOME_ADV puntos solo si la cancha no es neutral

Conversión a 1X2: el expected score E de Elo cumple E = P(home) + 0.5·P(draw).
P(draw) se modela con regresión logística sobre |diff ajustado|, neutral y K.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.settings import ELO_INITIAL, ELO_HOME_ADV, k_factor


def expected_score(elo_home: float, elo_away: float, neutral: bool,
                   home_adv: float = ELO_HOME_ADV) -> float:
    """Expected score Elo del equipo local: E = P(win) + 0.5·P(draw)."""
    diff = elo_home - elo_away + (0.0 if neutral else home_adv)
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def goal_diff_multiplier(gd: int) -> float:
    """Multiplicador del K por margen de victoria (World Football Elo)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def run_elo_forward(matches: pd.DataFrame,
                    initial: float = ELO_INITIAL,
                    home_adv: float = ELO_HOME_ADV):
    """Forward pass cronológico sobre todos los partidos.

    `matches` debe venir ordenado por fecha y contener:
    home_team, away_team, home_score, away_score, tournament, neutral.
    Los partidos sin resultado (score NaN) reciben Elo pre-partido pero no
    actualizan ratings.

    Returns:
        (df con columnas home_elo_pre/away_elo_pre añadidas,
         dict {team: elo_actual},
         dict {team: n_partidos_jugados})
    """
    ratings: dict[str, float] = {}
    n_played: dict[str, int] = {}
    home_pre = np.empty(len(matches))
    away_pre = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        rh = ratings.get(row.home_team, initial)
        ra = ratings.get(row.away_team, initial)
        home_pre[i] = rh
        away_pre[i] = ra

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        gd = int(row.home_score) - int(row.away_score)
        result = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        e = expected_score(rh, ra, bool(row.neutral), home_adv)
        k = k_factor(row.tournament) * goal_diff_multiplier(gd)
        delta = k * (result - e)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        n_played[row.home_team] = n_played.get(row.home_team, 0) + 1
        n_played[row.away_team] = n_played.get(row.away_team, 0) + 1

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    return out, ratings, n_played


def _draw_features(adj_diff: np.ndarray, k_weight: np.ndarray) -> np.ndarray:
    """Features del modelo de empate: |diff ajustado| escalado e importancia."""
    return np.column_stack([np.abs(adj_diff) / 100.0, k_weight / 60.0])


def fit_draw_model(df: pd.DataFrame) -> LogisticRegression:
    """Ajusta P(draw) ~ |diff Elo ajustado| + importancia del torneo.

    `df` requiere: home_elo_pre, away_elo_pre, neutral, k_weight y scores.
    """
    adj_diff = (df["home_elo_pre"] - df["away_elo_pre"]
                + np.where(df["neutral"], 0.0, ELO_HOME_ADV))
    X = _draw_features(adj_diff.to_numpy(), df["k_weight"].to_numpy())
    y = (df["home_score"] == df["away_score"]).astype(int).to_numpy()
    model = LogisticRegression()
    model.fit(X, y)
    return model


class EloPredictor:
    """Convierte ratings Elo en probabilidades 1X2 calibradas con el draw model."""

    def __init__(self, ratings: dict, draw_model: LogisticRegression,
                 home_adv: float = ELO_HOME_ADV):
        self.ratings = ratings
        self.draw_model = draw_model
        self.home_adv = home_adv

    def predict_proba(self, home_team: str, away_team: str,
                      neutral: bool = True,
                      tournament: str = "FIFA World Cup") -> tuple[float, float, float]:
        rh = self.ratings.get(home_team)
        ra = self.ratings.get(away_team)
        if rh is None or ra is None:
            missing = home_team if rh is None else away_team
            raise KeyError(f"Equipo sin rating Elo: {missing}")
        return self.predict_from_elo(rh, ra, neutral, k_factor(tournament))

    def predict_from_elo(self, elo_home: float, elo_away: float,
                         neutral: bool, k_weight: float) -> tuple[float, float, float]:
        adj_diff = elo_home - elo_away + (0.0 if neutral else self.home_adv)
        e = 1.0 / (1.0 + 10.0 ** (-adj_diff / 400.0))
        X = _draw_features(np.array([adj_diff]), np.array([k_weight]))
        p_draw = float(self.draw_model.predict_proba(X)[0, 1])
        # E = P(home) + 0.5·P(draw)  =>  despeje y clip para evitar negativos
        p_home = max(e - 0.5 * p_draw, 0.005)
        p_away = max(1.0 - e - 0.5 * p_draw, 0.005)
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def predict_proba_batch(self, elo_home: np.ndarray, elo_away: np.ndarray,
                            neutral: np.ndarray, k_weight: np.ndarray) -> np.ndarray:
        """Versión vectorizada para validación/backtesting. Devuelve (n, 3)."""
        adj_diff = elo_home - elo_away + np.where(neutral, 0.0, self.home_adv)
        e = 1.0 / (1.0 + 10.0 ** (-adj_diff / 400.0))
        X = _draw_features(adj_diff, k_weight)
        p_draw = self.draw_model.predict_proba(X)[:, 1]
        p_home = np.maximum(e - 0.5 * p_draw, 0.005)
        p_away = np.maximum(1.0 - e - 0.5 * p_draw, 0.005)
        probs = np.column_stack([p_home, p_draw, p_away])
        return probs / probs.sum(axis=1, keepdims=True)

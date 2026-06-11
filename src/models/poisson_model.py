"""Poisson global: lambda = exp(a + b·elo_diff_ajustado).

Un solo modelo para todas las selecciones (sin parámetros por equipo):
cada partido aporta dos filas (perspectiva local y visitante) y el rating
Elo resume la fuerza. Robusto con pocos datos por selección.
Habilita matriz de marcadores -> mercados 1X2, O/U, marcador exacto,
y es el insumo del simulador Monte Carlo (samplear marcadores).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from config.settings import ELO_HOME_ADV

MAX_GOALS = 10


class GlobalPoissonModel:
    def __init__(self, home_adv: float = ELO_HOME_ADV):
        self.home_adv = home_adv
        self.reg = PoissonRegressor(alpha=1e-8, max_iter=1000)

    def fit(self, df: pd.DataFrame) -> "GlobalPoissonModel":
        """`df` requiere: home_elo_pre, away_elo_pre, neutral, home_score, away_score."""
        adv = np.where(df["neutral"], 0.0, self.home_adv)
        diff_home = (df["home_elo_pre"] - df["away_elo_pre"] + adv).to_numpy()
        # perspectiva local y visitante apiladas: x = diff/400, y = goles anotados
        X = np.concatenate([diff_home, -diff_home]).reshape(-1, 1) / 400.0
        y = np.concatenate([df["home_score"].to_numpy(),
                            df["away_score"].to_numpy()]).astype(float)
        self.reg.fit(X, y)
        return self

    def predict_lambdas(self, elo_home: float, elo_away: float,
                        neutral: bool = True) -> tuple[float, float]:
        diff = elo_home - elo_away + (0.0 if neutral else self.home_adv)
        lams = self.reg.predict(np.array([[diff / 400.0], [-diff / 400.0]]))
        return float(lams[0]), float(lams[1])

    def score_matrix(self, lam_home: float, lam_away: float,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        """P(i goles local, j goles visitante), normalizada."""
        gh = poisson.pmf(np.arange(max_goals + 1), lam_home)
        ga = poisson.pmf(np.arange(max_goals + 1), lam_away)
        m = np.outer(gh, ga)
        return m / m.sum()

    def predict_proba(self, elo_home: float, elo_away: float,
                      neutral: bool = True) -> tuple[float, float, float]:
        m = self.score_matrix(*self.predict_lambdas(elo_home, elo_away, neutral))
        p_home = float(np.tril(m, -1).sum())   # i > j
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())    # j > i
        return p_home, p_draw, p_away

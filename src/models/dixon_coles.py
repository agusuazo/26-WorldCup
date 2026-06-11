"""Dixon-Coles via doble regresión Poisson con dummies de equipo + time-decay.

Alternativa más rápida al MLE con gradientes numéricos: usa sklearn
PoissonRegressor (L-BFGS interno optimizado) con matrices sparse.
La corrección rho (baja-puntuación) se estima por separado con scalar MLE.

Identifiabilidad: L2 regularización empuja ataque/defensa hacia cero
(equivalente al equipo promedio), sin necesitar equipo de referencia explícito.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

from config.settings import ELO_HOME_ADV

MAX_GOALS = 8


class DixonColesModel:
    """Modelo Dixon-Coles para goles esperados de selecciones nacionales."""

    # Cap razonable para goles esperados: ningún equipo promedia >4.5 en partidos
    # internacionales; la regularización solo no siempre es suficiente para
    # equipos con parámetros extrapolados fuera de su rango histórico.
    LAMBDA_MAX: float = 4.5
    LAMBDA_MIN: float = 0.20

    def __init__(self, xi: float = 0.0065, min_date: str = "2010-01-01",
                 reg: float = 0.02, max_goals: int = MAX_GOALS):
        self.xi = xi
        self.min_date = min_date
        self.reg = reg
        self.max_goals = max_goals

    # ---- Entrenamiento ------------------------------------------------

    def fit(self, df: pd.DataFrame, ref_date=None) -> "DixonColesModel":
        """Ajusta el modelo sobre partidos jugados (home_score no-NaN).

        `df` requiere: date, home_team, away_team, home_score, away_score, neutral.
        """
        df = df[(df["date"] >= self.min_date) & df["home_score"].notna()].copy()
        ref = pd.to_datetime(ref_date or df["date"].max())
        w = self._time_weights(df["date"], ref)

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.enc_ = OneHotEncoder(sparse_output=True, handle_unknown="ignore",
                                  categories=[all_teams])
        self.enc_.fit(np.array(all_teams).reshape(-1, 1))

        ht = self.enc_.transform(df["home_team"].values.reshape(-1, 1))
        at = self.enc_.transform(df["away_team"].values.reshape(-1, 1))
        ha = csr_matrix((1.0 - df["neutral"].astype(float).values).reshape(-1, 1))

        # home goals: intercept + att_home + effect_away_defense + home_adv*(1-neutral)
        Xh = hstack([ht, at, ha])
        self.h_model_ = PoissonRegressor(alpha=self.reg, max_iter=400, verbose=0)
        self.h_model_.fit(Xh, df["home_score"].values.astype(float), sample_weight=w)

        # away goals: intercept + att_away + effect_home_defense
        Xa = hstack([at, ht])
        self.a_model_ = PoissonRegressor(alpha=self.reg, max_iter=400, verbose=0)
        self.a_model_.fit(Xa, df["away_score"].values.astype(float), sample_weight=w)

        # rho: corrección de baja puntuación (Dixon-Coles 1997)
        lh = self.h_model_.predict(Xh)
        ma = self.a_model_.predict(Xa)
        self.rho_ = self._fit_rho(
            df["home_score"].values.astype(int),
            df["away_score"].values.astype(int),
            lh, ma, w)

        return self

    def _time_weights(self, dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
        days = (ref - pd.to_datetime(dates)).dt.days.clip(lower=0).values
        return np.exp(-self.xi * days)

    def _fit_rho(self, hs: np.ndarray, aw: np.ndarray,
                 lh: np.ndarray, ma: np.ndarray, w: np.ndarray) -> float:
        def neg_ll(rho: float) -> float:
            tau = np.ones(len(hs), dtype=np.float64)
            tau[(hs == 0) & (aw == 0)] = np.maximum(
                1.0 - lh[(hs == 0) & (aw == 0)] * ma[(hs == 0) & (aw == 0)] * rho, 1e-9)
            tau[(hs == 1) & (aw == 0)] = np.maximum(
                1.0 + ma[(hs == 1) & (aw == 0)] * rho, 1e-9)
            tau[(hs == 0) & (aw == 1)] = np.maximum(
                1.0 + lh[(hs == 0) & (aw == 1)] * rho, 1e-9)
            tau[(hs == 1) & (aw == 1)] = np.maximum(1.0 - rho, 1e-9)
            return -float(np.dot(w, np.log(tau)))

        res = minimize_scalar(neg_ll, bounds=(-0.5, 0.3), method="bounded")
        return float(res.x)

    # ---- Predicción ---------------------------------------------------

    def predict_lambdas(self, home: str, away: str,
                        neutral: bool = True) -> tuple[float, float] | tuple[None, None]:
        """Devuelve (lambda_home, lambda_away) o (None, None) si equipo desconocido."""
        try:
            h = self.enc_.transform([[home]])
            a = self.enc_.transform([[away]])
            ha_val = csr_matrix([[0.0 if neutral else 1.0]])
            lh = float(self.h_model_.predict(hstack([h, a, ha_val]))[0])
            ma = float(self.a_model_.predict(hstack([a, h]))[0])
            lh = float(np.clip(lh, self.LAMBDA_MIN, self.LAMBDA_MAX))
            ma = float(np.clip(ma, self.LAMBDA_MIN, self.LAMBDA_MAX))
            return lh, ma
        except Exception:
            return None, None

    def score_matrix(self, lh: float, ma: float) -> np.ndarray:
        """Matriz de probabilidades de marcadores P(i,j), con corrección rho."""
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, ma))
        rho = self.rho_
        m[0, 0] *= max(1.0 - lh * ma * rho, 1e-9)
        m[1, 0] *= max(1.0 + ma * rho, 1e-9)
        m[0, 1] *= max(1.0 + lh * rho, 1e-9)
        m[1, 1] *= max(1.0 - rho, 1e-9)
        return m / m.sum()

    def predict_proba(self, home: str, away: str,
                      neutral: bool = True) -> tuple[float, float, float] | None:
        lh, ma = self.predict_lambdas(home, away, neutral)
        if lh is None:
            return None
        m = self.score_matrix(lh, ma)
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())

    def predict_lambdas_bulk(self, teams: list[str]) -> np.ndarray:
        """Precomputa lambdas para todos los pares en `teams` (n×n×2).

        [i, j, 0] = lambda esperados del equipo i cuando juega como local vs j.
        [i, j, 1] = lambda esperados del equipo j cuando juega como visitante vs i.
        Todos los partidos asumidos neutral=True.
        """
        n = len(teams)
        lams = np.zeros((n, n, 2), dtype=np.float32)
        for i, ti in enumerate(teams):
            for j, tj in enumerate(teams):
                if i == j:
                    continue
                lh, ma = self.predict_lambdas(ti, tj, neutral=True)
                if lh is not None:
                    lams[i, j, 0] = lh
                    lams[i, j, 1] = ma
        return lams

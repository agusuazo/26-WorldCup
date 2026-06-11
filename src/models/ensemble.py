"""Ensemble que blendea Elo, Poisson global y Dixon-Coles.

Pesos optimizados en validation set minimizando log-loss multiclase.
Si Dixon-Coles no está disponible para un equipo, degrada al blend Elo+Poisson.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.models.dixon_coles import DixonColesModel
from src.models.elo_model import EloPredictor
from src.models.poisson_model import GlobalPoissonModel


class EnsemblePredictor:
    """Blend ponderado de hasta 3 modelos base."""

    DEFAULT_WEIGHTS = np.array([0.33, 0.34, 0.33])  # Elo, Poisson, DC

    def __init__(self, elo: EloPredictor, poisson: GlobalPoissonModel,
                 dc: DixonColesModel | None = None):
        self.elo = elo
        self.poisson = poisson
        self.dc = dc
        self.weights_ = self.DEFAULT_WEIGHTS.copy()

    # ---- Optimización de pesos ----------------------------------------

    def fit_weights(self, df_val: "pd.DataFrame") -> "EnsemblePredictor":
        """Ajusta pesos sobre validation set. `df_val` requiere columnas del
        feature store (home_elo_pre, away_elo_pre, neutral, k_weight) y scores."""
        import pandas as pd

        probs_elo, probs_pois, probs_dc = [], [], []
        labels = []

        for row in df_val.itertuples():
            pe = self.elo.predict_from_elo(
                row.home_elo_pre, row.away_elo_pre, bool(row.neutral), row.k_weight)
            pp = self.poisson.predict_proba(row.home_elo_pre, row.away_elo_pre,
                                            bool(row.neutral))
            gd = row.home_score - row.away_score
            label = 0 if gd > 0 else (1 if gd == 0 else 2)
            probs_elo.append(pe)
            probs_pois.append(pp)

            if self.dc is not None:
                pd_ = self.dc.predict_proba(row.home_team, row.away_team, bool(row.neutral))
                probs_dc.append(pd_ if pd_ is not None else pe)
            else:
                probs_dc.append(pe)
            labels.append(label)

        A = np.array(probs_elo)
        B = np.array(probs_pois)
        C = np.array(probs_dc)
        y = np.array(labels)

        def loss(w):
            blend = w[0]*A + w[1]*B + w[2]*C
            blend = blend / blend.sum(axis=1, keepdims=True)
            return log_loss(y, blend, labels=[0, 1, 2])

        # Optimizar sobre el simplex (pesos >= 0, suman 1)
        res = minimize(loss, x0=[0.33, 0.34, 0.33],
                       method="SLSQP",
                       bounds=[(0.05, 0.9)] * 3,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        if res.success:
            self.weights_ = res.x
        return self

    # ---- Predicción ---------------------------------------------------

    def predict_proba(self, home: str, away: str,
                      neutral: bool = True,
                      tournament: str = "FIFA World Cup") -> tuple[float, float, float]:
        from config.settings import k_factor
        rh = self.elo.ratings.get(home, 1500.0)
        ra = self.elo.ratings.get(away, 1500.0)

        pe = np.array(self.elo.predict_from_elo(rh, ra, neutral, k_factor(tournament)))
        pp = np.array(self.poisson.predict_proba(rh, ra, neutral))

        if self.dc is not None:
            pd_ = self.dc.predict_proba(home, away, neutral)
            pc = np.array(pd_) if pd_ is not None else pe
        else:
            pc = pe

        w = self.weights_
        blend = w[0]*pe + w[1]*pp + w[2]*pc
        blend /= blend.sum()
        return float(blend[0]), float(blend[1]), float(blend[2])

    def predict_lambdas(self, home: str, away: str,
                        neutral: bool = True) -> tuple[float, float]:
        """Devuelve goles esperados. Usa DC si está disponible, Poisson global de lo contrario."""
        if self.dc is not None:
            lh, ma = self.dc.predict_lambdas(home, away, neutral)
            if lh is not None:
                return lh, ma
        rh = self.elo.ratings.get(home, 1500.0)
        ra = self.elo.ratings.get(away, 1500.0)
        return self.poisson.predict_lambdas(rh, ra, neutral)

    def predict_lambdas_bulk(self, teams: list[str]) -> np.ndarray:
        """Precomputa lambdas neutrales para todos los pares (n×n×2)."""
        n = len(teams)
        lams = np.zeros((n, n, 2), dtype=np.float32)
        for i, ti in enumerate(teams):
            for j, tj in enumerate(teams):
                if i != j:
                    lh, ma = self.predict_lambdas(ti, tj, neutral=True)
                    lams[i, j, 0] = lh
                    lams[i, j, 1] = ma
        return lams

"""Calibración probabilística multinomial (Platt adaptado a 3 clases).

Ajusta una regresión logística multinomial sobre los log-odds del ensemble
para corregir miscalibración sistemática. Con 3 resultados (1X2), la
calibración one-vs-rest + renormalización es equivalente pero menos
consistente; aquí se usa regresión multinomial directa.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MultinomialCalibrator:
    """Wraps cualquier predictor con una capa de calibración multinomial."""

    def __init__(self):
        self.cal_: LogisticRegression | None = None

    def _proba_matrix(self, predictor, df) -> np.ndarray:
        return np.array([
            predictor.predict_proba(r.home_team, r.away_team, bool(r.neutral))
            for r in df.itertuples()])

    def fit(self, predictor, df_cal) -> "MultinomialCalibrator":
        """Ajusta sobre `df_cal` (precisa home_team, away_team, neutral,
        home_score, away_score)."""
        df_cal = df_cal.dropna(subset=["home_score", "away_score"])
        probs = self._proba_matrix(predictor, df_cal)
        eps = 1e-9
        # Features: log-odds de cada clase vs la media geométrica
        X = np.log(np.clip(probs, eps, 1.0))
        gd = (df_cal["home_score"] - df_cal["away_score"]).values
        y = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
        self.cal_ = LogisticRegression(C=5.0,
                                       max_iter=500, solver="lbfgs")
        self.cal_.fit(X, y)
        self._base_predictor = predictor
        return self

    def predict_proba(self, home: str, away: str,
                      neutral: bool = True,
                      tournament: str = "FIFA World Cup") -> tuple[float, float, float]:
        raw = np.array(self._base_predictor.predict_proba(home, away, neutral, tournament))
        eps = 1e-9
        X = np.log(np.clip(raw, eps, 1.0)).reshape(1, -1)
        p = self.cal_.predict_proba(X)[0]
        # reordenar por índice de clase (0=home, 1=draw, 2=away)
        idx = list(self.cal_.classes_)
        out = np.array([p[idx.index(i)] if i in idx else 0.0 for i in range(3)])
        out /= out.sum()
        return float(out[0]), float(out[1]), float(out[2])

    def predict_lambdas(self, home: str, away: str, neutral: bool = True):
        return self._base_predictor.predict_lambdas(home, away, neutral)

    def predict_lambdas_bulk(self, teams: list[str]) -> np.ndarray:
        return self._base_predictor.predict_lambdas_bulk(teams)

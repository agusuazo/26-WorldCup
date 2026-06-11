"""Carga y limpieza de fuentes de datos crudas."""
import pandas as pd

from config.settings import RESULTS_CSV, k_factor


def load_results() -> pd.DataFrame:
    """Carga el dataset martj42 limpio y ordenado cronológicamente.

    Mantiene partidos futuros (scores NaN): reciben Elo pre-partido y son
    el fixture del WC 2026.
    """
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["home_team", "away_team"])
    df["neutral"] = df["neutral"].astype(bool)
    df["k_weight"] = df["tournament"].map(k_factor)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", df.index)
    return df

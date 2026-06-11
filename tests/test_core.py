"""Tests del núcleo: Elo, Kelly, EV, vig. Ejecutar: pytest tests/ -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.betting.bankroll import (flat_stake, fractional_kelly,
                                  full_kelly_fraction, kelly_stake)
from src.betting.ev_calculator import classify_ev, expected_value
from src.betting.odds_parser import (overround, remove_vig_multiplicative,
                                     remove_vig_power, remove_vig_shin)
from src.models.elo_model import (expected_score, goal_diff_multiplier,
                                  run_elo_forward)


# ---------------- Elo ----------------
def test_expected_score_symmetric():
    assert expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_score_home_advantage():
    assert expected_score(1500, 1500, neutral=False) > 0.5


def test_expected_score_complementary():
    e1 = expected_score(1700, 1500, neutral=True)
    e2 = expected_score(1500, 1700, neutral=True)
    assert e1 + e2 == pytest.approx(1.0)


def test_goal_diff_multiplier_monotonic():
    vals = [goal_diff_multiplier(g) for g in range(1, 8)]
    assert vals == sorted(vals)
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5


def test_elo_forward_conserves_total():
    """La suma de Elo se conserva (juego de suma cero)."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "C", "C"],
        "home_score": [2, 0, 1], "away_score": [0, 1, 1],
        "tournament": ["Friendly"] * 3,
        "neutral": [True] * 3,
    })
    out, ratings, n = run_elo_forward(df, initial=1500.0)
    assert sum(ratings.values()) == pytest.approx(3 * 1500.0)
    assert ratings["A"] > 1500.0          # A ganó 2-0 y empató
    assert out["home_elo_pre"].iloc[0] == 1500.0  # pre-partido, no post


def test_elo_skips_future_matches():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2026-07-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [1, np.nan], "away_score": [0, np.nan],
        "tournament": ["Friendly"] * 2, "neutral": [True] * 2,
    })
    _, ratings, n = run_elo_forward(df)
    assert n["A"] == 1  # el partido futuro no actualiza


# ---------------- Kelly ----------------
def test_full_kelly_known_value():
    # p=0.5, cuota 3.0 -> b=2, f* = (2*0.5 - 0.5)/2 = 0.25
    assert full_kelly_fraction(0.5, 3.0) == pytest.approx(0.25)


def test_kelly_no_edge_is_zero():
    # p exactamente la implícita justa -> sin edge -> 0
    assert full_kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert full_kelly_fraction(0.3, 2.0) == 0.0  # edge negativo


def test_fractional_kelly_cap():
    # edge enorme: el cap del 5% debe activarse
    assert fractional_kelly(0.9, 5.0, fraction=0.5) == pytest.approx(0.05)


def test_kelly_stake_scales_with_bankroll():
    s1 = kelly_stake(1000, 0.5, 3.0)
    s2 = kelly_stake(2000, 0.5, 3.0)
    assert s2 == pytest.approx(2 * s1)


def test_flat_stake():
    assert flat_stake(1000, 0.02) == pytest.approx(20.0)


# ---------------- EV ----------------
def test_expected_value():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.6, 2.0) == pytest.approx(0.2)


def test_classify_ev_thresholds():
    assert classify_ev(0.12) == "HIGH"
    assert classify_ev(0.07) == "MEDIUM"
    assert classify_ev(0.02) == "LOW"
    assert classify_ev(-0.05) == "NONE"
    # outright exige más
    assert classify_ev(0.12, outright=True) == "NONE"
    assert classify_ev(0.20, outright=True) == "HIGH"


# ---------------- Vig removal ----------------
ODDS_1X2 = [2.10, 3.30, 3.60]


def test_overround_positive():
    assert overround(ODDS_1X2) > 1.0


@pytest.mark.parametrize("fn", [remove_vig_multiplicative, remove_vig_power,
                                remove_vig_shin])
def test_vig_removal_sums_to_one(fn):
    p = fn(ODDS_1X2)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all() and (p < 1).all()


def test_shin_shrinks_longshots_vs_multiplicative():
    """Shin debe asignar MENOS probabilidad al longshot que la multiplicativa.

    Mercado outright realista: las implícitas crudas suman ~1.37 (37% overround).
    """
    outright = [1.8, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]
    assert overround(outright) > 1.2
    p_mult = remove_vig_multiplicative(outright)
    p_shin = remove_vig_shin(outright)
    assert p_shin.sum() == pytest.approx(1.0)
    assert p_shin[-1] < p_mult[-1]   # longshot deflactado
    assert p_shin[0] > p_mult[0]     # favorito recibe lo recortado


def test_shin_no_overround_degrades_to_normalization():
    under = [3.0, 5.0, 8.0]  # suma de implícitas < 1
    p = remove_vig_shin(under)
    assert p.sum() == pytest.approx(1.0)

"""Tests del simulador Monte Carlo: bracket oficial, terceros, métricas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (brier_multiclass, max_drawdown,
                                     profit_factor, roi)
from src.simulation.monte_carlo import (R32_OFFICIAL, _apply_tiebreakers,
                                        _assign_thirds, _build_r32_bracket,
                                        _compute_standings,
                                        _select_best_thirds)


# ---------------- Bracket oficial R32 ----------------

def test_r32_official_has_16_matches():
    assert len(R32_OFFICIAL) == 16


def test_r32_official_eight_third_slots():
    thirds = [m for m, _a, (k, _) in R32_OFFICIAL if k == "T"]
    assert len(thirds) == 8


def test_r32_official_each_position_used_once():
    """Cada 1° y 2° de grupo aparece exactamente una vez en el bracket."""
    winners, runners = [], []
    for _mid, spec_a, spec_b in R32_OFFICIAL:
        for kind, arg in (spec_a, spec_b):
            if kind == "W":
                winners.append(arg)
            elif kind == "R":
                runners.append(arg)
    assert sorted(winners) == list("ABCDEFGHIJKL")
    assert sorted(runners) == list("ABCDEFGHIJKL")


def test_assign_thirds_valid_assignment():
    rng = np.random.default_rng(7)
    allowed = {mid: arg for mid, _a, (k, arg) in R32_OFFICIAL if k == "T"}
    for combo in ["ABCDEFGH", "EHIJKLDG", "CDFGHIJL", "ABDFIJKL"]:
        qualified = [(f"T{g}", g) for g in combo]
        a = _assign_thirds(qualified, rng)
        assert len(a) == 8
        assert len(set(a.values())) == 8
        for mid, team in a.items():
            assert team[-1] in allowed[mid]


def test_build_r32_bracket_complete():
    rng = np.random.default_rng(1)
    standing1 = {g: f"W{g}" for g in "ABCDEFGHIJKL"}
    standing2 = {g: f"R{g}" for g in "ABCDEFGHIJKL"}
    thirds = [(f"T{g}", g) for g in "ABCDEFGH"]
    bracket = _build_r32_bracket(standing1, standing2, thirds, rng)
    assert len(bracket) == 16
    teams = [t for pair in bracket for t in pair]
    assert len(set(teams)) == 32


# ---------------- Standings y desempates ----------------

def test_standings_points():
    results = [("A", "B", 2, 0), ("C", "D", 1, 1),
               ("A", "C", 0, 0), ("B", "D", 3, 1),
               ("A", "D", 1, 0), ("B", "C", 0, 2)]
    rows = _compute_standings(["A", "B", "C", "D"], results)
    assert rows["A"]["pts"] == 7   # W, D, W
    assert rows["C"]["pts"] == 5   # D, D, W
    assert rows["B"]["pts"] == 3   # L, W, L
    assert rows["D"]["pts"] == 1   # D, L, L


def test_tiebreaker_order():
    rng = np.random.default_rng(0)
    rows = {"X": {"pts": 6, "gd": 3, "gf": 5},
            "Y": {"pts": 6, "gd": 1, "gf": 4},
            "Z": {"pts": 4, "gd": 5, "gf": 9}}
    ordered = _apply_tiebreakers(rows, rng)
    assert ordered == ["X", "Y", "Z"]  # pts primero, luego gd


def test_select_best_thirds_returns_team_group():
    rng = np.random.default_rng(0)
    thirds = [{"team": f"T{i}", "group": g, "pts": i, "gd": 0, "gf": 0}
              for i, g in enumerate("ABCDEFGHIJKL")]
    best = _select_best_thirds(thirds, n=8, rng=rng)
    assert len(best) == 8
    assert all(isinstance(t, tuple) and len(t) == 2 for t in best)
    # Los de más puntos clasifican
    assert ("T11", "L") in best
    assert ("T0", "A") not in best


# ---------------- Métricas de backtesting ----------------

def test_brier_perfect_and_uniform():
    outcomes = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert brier_multiclass(perfect, outcomes) == pytest.approx(0.0)
    uniform = np.full((3, 3), 1 / 3)
    assert brier_multiclass(uniform, outcomes) == pytest.approx(0.2222, abs=1e-3)


def test_roi_and_profit_factor():
    stakes = np.array([10.0, 10.0, 10.0])
    profits = np.array([15.0, -10.0, -10.0])
    assert roi(stakes, profits) == pytest.approx(-5 / 30)
    assert profit_factor(profits) == pytest.approx(15 / 20)


def test_max_drawdown():
    equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    assert max_drawdown(equity) == pytest.approx(-40.0)  # 120 -> 80


# ---------------- Simulación detallada (vista de bracket) ----------------

class _StubPredictor:
    """Lambdas sintéticos desde un ranking de fuerza por orden alfabético:
    el equipo i marca más goles esperados cuanto mejor sea su rank relativo."""

    def predict_lambdas_bulk(self, teams):
        n = len(teams)
        strength = np.linspace(2.2, 0.6, n)  # primero = más fuerte
        lams = np.zeros((n, n, 2), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    lams[i, j, 0] = strength[i] * 0.8 + 0.3
                    lams[i, j, 1] = strength[j] * 0.8 + 0.3
        return lams


@pytest.fixture(scope="module")
def detail_expected():
    from src.simulation.monte_carlo import simulate_tournament_detail
    return simulate_tournament_detail(_StubPredictor(), seed=1, mode="expected")


@pytest.fixture(scope="module")
def detail_sample():
    from src.simulation.monte_carlo import simulate_tournament_detail
    return simulate_tournament_detail(_StubPredictor(), seed=1, mode="sample")


@pytest.mark.parametrize("which", ["expected", "sample"])
def test_detail_structure(which, detail_expected, detail_sample):
    d = detail_expected if which == "expected" else detail_sample
    r32_teams = [t for a, b, _w in d["rounds"]["r32"] for t in (a, b)]
    assert len(r32_teams) == 32
    assert len(set(r32_teams)) == 32
    assert len(d["rounds"]["r16"]) == 8
    assert len(d["rounds"]["qf"]) == 4
    assert len(d["rounds"]["sf"]) == 2
    assert len(d["rounds"]["final"]) == 1
    fa, fb, w = d["rounds"]["final"][0]
    assert d["champion"] == w and w in (fa, fb)
    assert len(d["best_thirds"]) == 8
    assert len(d["groups"]) == 12


def test_detail_winners_advance(detail_sample):
    """Cada ganador de una ronda aparece en la siguiente."""
    d = detail_sample
    order = ["r32", "r16", "qf", "sf", "final"]
    for prev, nxt in zip(order, order[1:]):
        winners = {w for _a, _b, w in d["rounds"][prev]}
        next_teams = {t for a, b, _w in d["rounds"][nxt] for t in (a, b)}
        assert next_teams <= winners


def test_detail_qualified_consistent(detail_expected):
    """1° y 2° de cada grupo son los dos primeros del standing; el tercero
    clasificado (si lo hay) es exactamente el 3° del standing."""
    d = detail_expected
    n_thirds = 0
    for g in d["groups"].values():
        teams_ordered = [t for t, *_ in g["standings"]]
        q = g["qualified"]
        assert q["first"] == teams_ordered[0]
        assert q["second"] == teams_ordered[1]
        if q["third"] is not None:
            assert q["third"] == teams_ordered[2]
            n_thirds += 1
    assert n_thirds == 8


def test_detail_expected_reproducible():
    from src.simulation.monte_carlo import simulate_tournament_detail
    d1 = simulate_tournament_detail(_StubPredictor(), seed=5, mode="expected")
    d2 = simulate_tournament_detail(_StubPredictor(), seed=5, mode="expected")
    assert d1["rounds"] == d2["rounds"]
    assert d1["champion"] == d2["champion"]


def test_detail_invalid_mode():
    from src.simulation.monte_carlo import simulate_tournament_detail
    with pytest.raises(ValueError):
        simulate_tournament_detail(_StubPredictor(), mode="foo")


# ---------------- CLV (Closing Line Value) ----------------

def test_clv_summary_basic():
    from src.betting.bet_log import clv_summary
    df = pd.DataFrame({
        "odds":         [2.20, 3.00, 1.80, 2.50],
        "closing_odds": [2.00, 3.30, 1.80, None],   # +10%, -9.1%, 0%, sin cierre
    })
    s = clv_summary(df)
    assert s["n_with_clv"] == 3                      # el None se excluye
    assert s["avg_clv"] == pytest.approx((0.10 - 1/11 + 0.0) / 3, abs=1e-6)
    assert s["pct_beat_close"] == pytest.approx(1 / 3)


def test_clv_summary_empty():
    from src.betting.bet_log import clv_summary
    df = pd.DataFrame({"odds": [2.0], "closing_odds": [None]})
    assert clv_summary(df) is None


# ---------------- Conditioning a resultados reales ----------------

def _stub_group_setup():
    teams = ["P", "Q", "R", "S"]
    tidx = {t: i for i, t in enumerate(teams)}
    lams = np.full((4, 4, 2), 1.2, dtype=np.float32)
    return teams, tidx, lams


def test_group_matches_respect_played():
    from src.simulation.monte_carlo import _simulate_group_matches
    teams, tidx, lams = _stub_group_setup()
    rng = np.random.default_rng(0)
    played = [("P", "Q", 5, 0), ("R", "S", 2, 2)]
    for _ in range(20):
        results = _simulate_group_matches(teams, tidx, lams, rng, played)
        assert len(results) == 6
        assert ("P", "Q", 5, 0) in results
        assert ("R", "S", 2, 2) in results


def test_expected_group_real_points():
    from src.simulation.monte_carlo import _expected_group
    teams, tidx, lams = _stub_group_setup()
    # P ganó 1-0 a Q (real): P arranca con 3 pts reales + esperados de 2 partidos
    played = [("P", "Q", 1, 0)]
    _results, rows = _expected_group(teams, tidx, lams, played)
    _r2, rows_no = _expected_group(teams, tidx, lams, None)
    # Con lambdas idénticos para todos, lo esperado de los 2 partidos restantes
    # es igual para P y Q; la diferencia entre ambos debe ser los 3 pts reales.
    assert rows["P"]["pts"] - rows["Q"]["pts"] == pytest.approx(3.0)
    # Y P debe tener más puntos que en el caso sin condicionar (ganó un partido
    # que en expectativa daba ~1.x pts)
    assert rows["P"]["pts"] > rows_no["P"]["pts"]


def test_ko_winner_override_both_modes():
    from src.simulation.monte_carlo import simulate_tournament_detail
    base = simulate_tournament_detail(_StubPredictor(), seed=3, mode="expected")
    # Forzar que en la final real gane el subcampeón simulado
    fa, fb, w = base["rounds"]["final"][0]
    loser = fb if w == fa else fa
    state = {"group_results": {}, "ko_winners": {frozenset((fa, fb)): loser}}
    for mode in ("expected", "sample"):
        d = simulate_tournament_detail(_StubPredictor(), seed=3, mode=mode,
                                       state=state)
        # mismo seed + mismos grupos -> misma final; el override decide
        if set(d["rounds"]["final"][0][:2]) == {fa, fb}:
            assert d["champion"] == loser


def test_monte_carlo_conditioned_group_decided():
    """Grupo totalmente decidido: posiciones reales se respetan el 100%."""
    from src.simulation.monte_carlo import run_monte_carlo
    import json
    from pathlib import Path
    fixtures = Path(__file__).resolve().parents[1] / "config" / "wc2026_fixtures.json"
    groups = json.loads(fixtures.read_text(encoding="utf-8"))["groups"]
    g_name = "A"
    t1, t2, t3, t4 = groups[g_name]
    # t1 gana todo, t2 segundo, t3 tercero, t4 pierde todo
    played = [(t1, t2, 2, 0), (t1, t3, 2, 0), (t1, t4, 2, 0),
              (t2, t3, 2, 0), (t2, t4, 2, 0), (t3, t4, 2, 0)]
    state = {"group_results": {g_name: played}, "ko_winners": {}}
    df = run_monte_carlo(_StubPredictor(), n=150, seed=1, state=state)
    row4 = df[df["team"] == t4].iloc[0]
    assert row4["group_exit_pct"] == pytest.approx(100.0)   # 4° real: fuera siempre
    row1 = df[df["team"] == t1].iloc[0]
    assert row1["r32_pct"] == pytest.approx(100.0)          # 1° real: clasifica siempre


def test_monte_carlo_percentage_invariants():
    """Invariantes de las probabilidades acumuladas (vista del simulador)."""
    from src.simulation.monte_carlo import run_monte_carlo
    df = run_monte_carlo(_StubPredictor(), n=200, seed=2)
    assert df["champion_pct"].sum() == pytest.approx(100.0, abs=0.5)
    assert df["finalist_pct"].sum() == pytest.approx(200.0, abs=0.5)
    assert df["sf_pct"].sum() == pytest.approx(400.0, abs=0.5)
    assert df["qf_pct"].sum() == pytest.approx(800.0, abs=0.5)
    assert df["r16_pct"].sum() == pytest.approx(1600.0, abs=0.5)
    assert df["r32_pct"].sum() == pytest.approx(3200.0, abs=0.5)
    # Complemento por fila
    assert ((df["r32_pct"] + df["group_exit_pct"]) - 100.0).abs().max() < 0.1
    # Acumuladas monotonas por fila
    for a, b in [("champion_pct", "finalist_pct"), ("finalist_pct", "sf_pct"),
                 ("sf_pct", "qf_pct"), ("qf_pct", "r16_pct"),
                 ("r16_pct", "r32_pct")]:
        assert (df[b] >= df[a] - 1e-9).all()

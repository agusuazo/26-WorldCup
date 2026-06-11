"""Monte Carlo para el Mundial FIFA 2026.

Pipeline:
  1. Fase de grupos (12 grupos × 6 partidos) — samplea marcadores Poisson
  2. Selección de 8 mejores terceros — criterios FIFA
  3. Ronda de 32 con bracket predefinido
  4. Rondas de eliminación hasta campeón

Optimización de rendimiento: las lambdas se precomputan para todos los pares
de las 48 selecciones mundialistas (2256 pares). En el loop de simulación solo
se llama a numpy.random.poisson (vectorizado).

Con N=10_000 sims: ≈ 8-15 s en CPU estándar.
Con N=50_000 sims: ≈ 40-70 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ---- Bracket R32 oficial WC 2026 (FIFA, partidos 73-88) ------------------
# Fuente: calendario oficial FIFA / Wikipedia "2026 FIFA World Cup knockout
# stage". Cada cruce se especifica como ("W", grupo) = 1° del grupo,
# ("R", grupo) = 2° del grupo, ("T", grupos) = mejor 3° proveniente de uno
# de esos grupos (la asignación exacta depende de qué 8 grupos aportan
# terceros; se resuelve por matching con backtracking).
#
# El orden de la lista está dispuesto para que el pareo consecutivo de
# ganadores reproduzca el árbol oficial:
#   R16:  M89=(74,77) M90=(73,75) M93=(83,84) M94=(81,82)
#         M91=(76,78) M92=(79,80) M95=(86,88) M96=(85,87)
#   QF:   M97=(89,90) M98=(93,94) M99=(91,92) M100=(95,96)
#   SF:   M101=(97,98) M102=(99,100) → Final M104

R32_OFFICIAL: list[tuple[int, tuple[str, str], tuple[str, str]]] = [
    (74, ("W", "E"), ("T", "ABCDF")),
    (77, ("W", "I"), ("T", "CDFGH")),
    (73, ("R", "A"), ("R", "B")),
    (75, ("W", "F"), ("R", "C")),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (81, ("W", "D"), ("T", "BEFIJ")),
    (82, ("W", "G"), ("T", "AEHIJ")),
    (76, ("W", "C"), ("R", "F")),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("T", "CEFHI")),
    (80, ("W", "L"), ("T", "EHIJK")),
    (86, ("W", "J"), ("R", "H")),
    (88, ("R", "D"), ("R", "G")),
    (85, ("W", "B"), ("T", "EFGIJ")),
    (87, ("W", "K"), ("T", "DEIJL")),
]


def _assign_thirds(qualified: list[tuple[str, str]],
                   rng: np.random.Generator) -> dict[int, str]:
    """Asigna los 8 mejores terceros a los 8 slots del bracket oficial.

    `qualified`: lista de (team, group) de los 8 terceros clasificados.
    Devuelve {match_id: team}. La tabla oficial FIFA de 495 combinaciones
    garantiza que siempre existe una asignación factible; se encuentra por
    backtracking (slots con menos candidatos primero).
    """
    by_group = {g: t for t, g in qualified}
    slots = [(mid, b_arg) for mid, _a, (b_kind, b_arg) in R32_OFFICIAL
             if b_kind == "T"]
    # ordenar por nº de candidatos disponibles (heurística MRV)
    slots.sort(key=lambda s: sum(1 for g in s[1] if g in by_group))

    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        mid, allowed = slots[i]
        cands = [g for g in allowed if g in by_group and g not in used]
        rng.shuffle(cands)
        for g in cands:
            assignment[mid] = by_group[g]
            used.add(g)
            if backtrack(i + 1):
                return True
            del assignment[mid]
            used.discard(g)
        return False

    if not backtrack(0):
        # No debería ocurrir (Hall garantizado por la tabla FIFA); fallback:
        # asignar los terceros restantes en orden arbitrario.
        remaining = [t for t, g in qualified if g not in used]
        for mid, _ in slots:
            if mid not in assignment and remaining:
                assignment[mid] = remaining.pop()
    return assignment


def _build_r32_bracket(standing1: dict, standing2: dict,
                       best_thirds: list[tuple[str, str]],
                       rng: np.random.Generator) -> list[tuple[str, str]]:
    """Construye los 16 cruces de R32 según el bracket oficial FIFA."""
    third_by_match = _assign_thirds(best_thirds, rng)

    def resolve(spec: tuple[str, str], match_id: int) -> str | None:
        kind, arg = spec
        if kind == "W":
            return standing1.get(arg)
        if kind == "R":
            return standing2.get(arg)
        return third_by_match.get(match_id)

    bracket: list[tuple[str, str]] = []
    for match_id, spec_a, spec_b in R32_OFFICIAL:
        a = resolve(spec_a, match_id)
        b = resolve(spec_b, match_id)
        if a and b:
            bracket.append((a, b))
    return bracket


# ---- Utilidades de grupo -----------------------------------------------

def _simulate_group_matches(teams: list[str], tidx: dict,
                             lams: np.ndarray, rng: np.random.Generator,
                             played: list[tuple] | None = None
                             ) -> list[tuple[str, str, int, int]]:
    """Simula los 6 partidos de un grupo y devuelve (home, away, hs, as).

    `played`: resultados reales ya jugados [(home, away, hs, as), ...] —
    se respetan tal cual; solo se samplean los partidos restantes.
    """
    played_by_pair = {frozenset((h, a)): (h, a, hs, aws)
                      for h, a, hs, aws in (played or [])}
    results = []
    for i, ti in enumerate(teams):
        for j, tj in enumerate(teams):
            if j <= i:
                continue
            real = played_by_pair.get(frozenset((ti, tj)))
            if real is not None:
                results.append(real)
                continue
            lh = float(lams[tidx[ti], tidx[tj], 0])
            ma = float(lams[tidx[ti], tidx[tj], 1])
            hs = int(rng.poisson(max(lh, 0.01)))
            aws = int(rng.poisson(max(ma, 0.01)))
            results.append((ti, tj, hs, aws))
    return results


def _compute_standings(teams: list[str],
                       results: list[tuple[str, str, int, int]]
                       ) -> dict[str, dict]:
    """Tabla de posiciones de un grupo como dict plano (hot path del MC:
    se evita pandas — 12 grupos × n sims construcciones de DataFrame)."""
    rows = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for h, a, hs, aws in results:
        gd = hs - aws
        if gd > 0:
            rows[h]["pts"] += 3
        elif gd == 0:
            rows[h]["pts"] += 1
            rows[a]["pts"] += 1
        else:
            rows[a]["pts"] += 3
        rows[h]["gd"] += gd;   rows[h]["gf"] += hs
        rows[a]["gd"] -= gd;   rows[a]["gf"] += aws
    return rows


def _apply_tiebreakers(rows: dict[str, dict],
                       rng: np.random.Generator) -> list[str]:
    """Ordena la tabla aplicando desempates FIFA. Devuelve lista ordenada.

    Para desempates H2H se necesita el contexto del grupo completo;
    usamos pts → gd → gf → ruido aleatorio como proxy del sorteo final.
    """
    return sorted(rows, key=lambda t: (-rows[t]["pts"], -rows[t]["gd"],
                                       -rows[t]["gf"], rng.random()))


# ---- Simulador de partido de eliminatoria ------------------------------

def _simulate_ko_match(home: str, away: str, tidx: dict,
                       lams: np.ndarray, rng: np.random.Generator) -> str:
    """Simula 90min + prórroga + penales. Devuelve el ganador."""
    if home not in tidx or away not in tidx:
        # Fallback: 50/50
        return home if rng.random() < 0.5 else away

    lh = float(lams[tidx[home], tidx[away], 0])
    ma = float(lams[tidx[home], tidx[away], 1])
    lh, ma = max(lh, 0.01), max(ma, 0.01)

    hs = int(rng.poisson(lh))
    aws = int(rng.poisson(ma))

    if hs != aws:
        return home if hs > aws else away

    # Prórroga: tasas reducidas (fatiga ≈ 30% menos goles)
    hs_et = int(rng.poisson(lh * 0.50))
    aws_et = int(rng.poisson(ma * 0.50))
    if hs_et != aws_et:
        return home if hs_et > aws_et else away

    # Penales: 50/50 con leve ventaja al equipo con Elo mayor
    # (ya está implícito en las lambdas; aquí usamos ratio de lambdas)
    p_home_pen = np.clip(lh / (lh + ma), 0.35, 0.65)
    return home if rng.random() < p_home_pen else away


# ---- Selección de mejores terceros --------------------------------------

def _select_best_thirds(thirds: list[dict], n: int = 8,
                         rng: np.random.Generator = None) -> list[tuple[str, str]]:
    """Selecciona los `n` mejores terceros según criterios FIFA:
    puntos → dif. goles → goles a favor → aleatorio.
    Devuelve lista de (team, group) — el grupo se necesita para la
    asignación oficial de slots en el bracket R32."""
    thirds_sorted = sorted(
        thirds,
        key=lambda r: (-r["pts"], -r["gd"], -r["gf"],
                       rng.random() if rng else 0.0))
    return [(r["team"], r["group"]) for r in thirds_sorted[:n]]


# ---- Estado real del torneo (conditioning) -------------------------------

# La fase de grupos termina el 27-jun-2026; R32 empieza el 28-jun. Se usa la
# fecha (no la pertenencia a grupo) para clasificar: desde cuartos en adelante
# dos equipos del mismo grupo SÍ pueden cruzarse.
WC2026_START = "2026-06-01"
GROUP_STAGE_END = "2026-06-27"


def load_tournament_state(db_path=None,
                          wc_fixtures_path: Path | None = None) -> dict:
    """Lee de la BD los partidos WC 2026 ya jugados y los estructura para
    condicionar la simulación.

    Devuelve {"group_results": {grupo: [(h, a, hs, as), ...]},
              "ko_winners": {frozenset({X, Y}): ganador},
              "n_group_played": int, "n_ko_played": int}.

    Empates KO: el ganador se busca en manual_results.winner y luego en
    shootouts.csv; si no se puede determinar, el partido se omite (se simula).
    """
    import duckdb

    from config.settings import DB_PATH, SHOOTOUTS_CSV
    if db_path is None:
        db_path = DB_PATH
    if wc_fixtures_path is None:
        wc_fixtures_path = ROOT / "config" / "wc2026_fixtures.json"
    groups = json.loads(wc_fixtures_path.read_text(encoding="utf-8"))["groups"]
    team_to_group = {t: g for g, teams in groups.items() for t in teams}

    state = {"group_results": {}, "ko_winners": {},
             "n_group_played": 0, "n_ko_played": 0}
    if not Path(db_path).exists():
        return state

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        played = con.execute("""
            SELECT date, home_team, away_team, home_score, away_score
            FROM matches
            WHERE tournament = 'FIFA World Cup'
              AND date >= ? AND home_score IS NOT NULL
            ORDER BY date
        """, [WC2026_START]).df()
    finally:
        con.close()

    # Overlay de resultados manuales (local: tabla DuckDB; server: Supabase).
    # En el deploy la BD viaja horneada en la imagen y es de solo lectura,
    # así que los resultados manuales se mergean aquí en runtime.
    manual_winners: dict = {}
    try:
        from src.ingestion.updater import get_manual_results
        manual = get_manual_results()
    except Exception:
        manual = pd.DataFrame()
    if not manual.empty:
        manual = manual[pd.to_datetime(manual["date"]) >= pd.Timestamp(WC2026_START)]
        seen = {(pd.Timestamp(r.date).date(), r.home_team, r.away_team)
                for r in played.itertuples()}
        extra = []
        for r in manual.itertuples():
            if pd.notna(r.winner) and r.winner:
                manual_winners[frozenset((r.home_team, r.away_team))] = r.winner
            key = (pd.Timestamp(r.date).date(), r.home_team, r.away_team)
            if key not in seen:
                extra.append({"date": pd.Timestamp(r.date),
                              "home_team": r.home_team, "away_team": r.away_team,
                              "home_score": r.home_score, "away_score": r.away_score})
        if extra:
            played = pd.concat([played, pd.DataFrame(extra)],
                               ignore_index=True).sort_values("date")

    shootout_winners = {}
    if SHOOTOUTS_CSV.exists():
        so = pd.read_csv(SHOOTOUTS_CSV, parse_dates=["date"])
        so = so[so["date"] >= WC2026_START]
        for r in so.itertuples():
            shootout_winners[frozenset((r.home_team, r.away_team))] = r.winner

    group_end = pd.Timestamp(GROUP_STAGE_END)
    for r in played.itertuples():
        h, a = r.home_team, r.away_team
        hs, aws = int(r.home_score), int(r.away_score)
        if pd.Timestamp(r.date) <= group_end:
            g = team_to_group.get(h)
            if g is None or team_to_group.get(a) != g:
                continue   # partido no mapeable a un grupo
            state["group_results"].setdefault(g, []).append((h, a, hs, aws))
            state["n_group_played"] += 1
        else:
            pair = frozenset((h, a))
            if hs != aws:
                winner = h if hs > aws else a
            else:
                winner = manual_winners.get(pair) or shootout_winners.get(pair)
                if winner not in (h, a):
                    continue   # indeterminable: se simula
            state["ko_winners"][pair] = winner
            state["n_ko_played"] += 1
    return state


# ---- Loop principal Monte Carlo ----------------------------------------

def run_monte_carlo(ensemble,
                    wc_fixtures_path: Path | None = None,
                    n: int = 10_000,
                    seed: int = 42,
                    progress_cb=None,
                    state: dict | None = None) -> pd.DataFrame:
    """Simula el torneo WC 2026 `n` veces.

    `ensemble` debe implementar `predict_lambdas(home, away, neutral=True)`.
    `state` (de load_tournament_state): condiciona la simulación a los
    partidos ya jugados — sus resultados son fijos en todas las iteraciones.
    Devuelve DataFrame con probabilidades por selección y ronda.
    """
    if wc_fixtures_path is None:
        wc_fixtures_path = ROOT / "config" / "wc2026_fixtures.json"
    wc = json.loads(wc_fixtures_path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = wc["groups"]
    all_teams: list[str] = sorted({t for g in groups.values() for t in g})
    tidx: dict[str, int] = {t: i for i, t in enumerate(all_teams)}

    # Pre-computar lambdas para todos los pares (un solo forward pass)
    lams = ensemble.predict_lambdas_bulk(all_teams)

    group_played = (state or {}).get("group_results", {})
    ko_winners = (state or {}).get("ko_winners", {})

    rng = np.random.default_rng(seed)

    # exit_counts[team][round] = nº de sims donde ese equipo salió en esa ronda
    # Rondas posibles: "group", "r32", "r16", "qf", "sf", "ru", "champion"
    EXIT_ROUNDS = ["group", "r32", "r16", "qf", "sf", "ru", "champion"]
    exit_counts: dict[str, dict[str, int]] = {
        t: {r: 0 for r in EXIT_ROUNDS} for t in all_teams}

    def _ko_run(bracket_in: list[str], exit_round: str) -> list[str]:
        """Simula una ronda KO: devuelve ganadores, registra perdedores."""
        winners = []
        for i in range(0, len(bracket_in), 2):
            if i + 1 >= len(bracket_in):
                winners.append(bracket_in[i])
                continue
            h, a = bracket_in[i], bracket_in[i + 1]
            w = ko_winners.get(frozenset((h, a))) if ko_winners else None
            if w is None:
                w = _simulate_ko_match(h, a, tidx, lams, rng)
            loser = a if w == h else h
            exit_counts[loser][exit_round] += 1
            winners.append(w)
        return winners

    for sim in range(n):
        if progress_cb and sim % 500 == 0:
            progress_cb(sim / n)

        standing1: dict[str, str] = {}
        standing2: dict[str, str] = {}
        thirds: list[dict] = []

        # ---- Fase de grupos (cada equipo recibe su ronda de salida aquí) ----
        for g_name, g_teams in groups.items():
            results = _simulate_group_matches(g_teams, tidx, lams, rng,
                                              played=group_played.get(g_name))
            standings = _compute_standings(g_teams, results)
            ordered = _apply_tiebreakers(standings, rng)
            standing1[g_name] = ordered[0]
            standing2[g_name] = ordered[1]
            # El 4° sale definitivamente en grupos
            exit_counts[ordered[3]]["group"] += 1
            # El 3° puede salir en grupos o avanzar (se decide abajo)
            third = standings[ordered[2]]
            thirds.append({"team": ordered[2], "group": g_name,
                           "pts": third["pts"], "gd": third["gd"],
                           "gf": third["gf"]})

        # ---- Mejores 8 terceros ----
        best_8 = _select_best_thirds(thirds, n=8, rng=rng)
        best_8_set = {t for t, _g in best_8}
        for t_info in thirds:
            if t_info["team"] not in best_8_set:
                exit_counts[t_info["team"]]["group"] += 1   # 3° que no clasifica

        # ---- Construir bracket R32 (32 equipos) ----
        bracket = _build_r32_bracket(standing1, standing2, best_8, rng)
        flat = [t for pair in bracket for t in pair]   # lista plana de 32

        # ---- Rondas KO ----
        r16 = _ko_run(flat,  "r32")  # perdedores de R32
        qf  = _ko_run(r16,   "r16")  # perdedores de R16
        sf  = _ko_run(qf,    "qf")   # perdedores de QF
        fin = _ko_run(sf,    "sf")   # perdedores de SF

        if len(fin) >= 2:
            champ = ko_winners.get(frozenset((fin[0], fin[1]))) if ko_winners else None
            if champ is None:
                champ = _simulate_ko_match(fin[0], fin[1], tidx, lams, rng)
            ru = fin[1] if champ == fin[0] else fin[0]
            exit_counts[ru]["ru"] += 1
            exit_counts[champ]["champion"] += 1
        elif fin:
            exit_counts[fin[0]]["champion"] += 1

    # ---- Convertir a probabilidades acumuladas ----
    rows = []
    for team in all_teams:
        c = exit_counts[team]
        total = sum(c.values())
        # Probabilidad de llegar AL MENOS hasta cada ronda (acumulada desde arriba)
        champion_pct  = c["champion"] / n
        finalist_pct  = champion_pct  + c["ru"] / n
        sf_pct        = finalist_pct  + c["sf"] / n
        qf_pct        = sf_pct        + c["qf"] / n
        r16_pct       = qf_pct        + c["r16"] / n
        r32_pct       = r16_pct       + c["r32"] / n
        group_pct     = 1.0 - r32_pct   # porcentaje que sale en grupos
        rows.append({
            "team": team,
            "champion_pct": round(champion_pct * 100, 2),
            "finalist_pct": round(finalist_pct * 100, 2),
            "sf_pct":       round(sf_pct * 100, 2),
            "qf_pct":       round(qf_pct * 100, 2),
            "r16_pct":      round(r16_pct * 100, 2),
            "r32_pct":      round(r32_pct * 100, 2),
            "group_exit_pct": round(group_pct * 100, 2),
        })

    df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# ---- Simulación detallada de UN torneo (vista de bracket) ----------------

def _match_probs_from_lams(lh: float, la: float,
                           max_goals: int = 8) -> tuple[float, float, float]:
    """Probabilidades 1X2 a 90min desde lambdas Poisson independientes."""
    from scipy.stats import poisson
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, max(lh, 0.01)), poisson.pmf(g, max(la, 0.01)))
    m /= m.sum()
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def _expected_group(teams: list[str], tidx: dict, lams: np.ndarray,
                    played: list[tuple] | None = None) -> tuple[list, dict]:
    """Standings esperados de un grupo (sin sampleo): pts = Σ 3·P(win)+P(draw),
    gd/gf desde lambdas. Los 'resultados' son los goles esperados (floats).

    `played`: partidos reales — aportan puntos/goles reales (3/1/0) y se
    excluyen del cálculo esperado.
    """
    played_by_pair = {frozenset((h, a)): (h, a, hs, aws)
                      for h, a, hs, aws in (played or [])}
    rows = {t: {"pts": 0.0, "gd": 0.0, "gf": 0.0} for t in teams}
    results = []
    for i, ti in enumerate(teams):
        for j, tj in enumerate(teams):
            if j <= i:
                continue
            real = played_by_pair.get(frozenset((ti, tj)))
            if real is not None:
                h, a, hs, aws = real
                gd = hs - aws
                rows[h]["pts"] += 3 if gd > 0 else (1 if gd == 0 else 0)
                rows[a]["pts"] += 3 if gd < 0 else (1 if gd == 0 else 0)
                rows[h]["gd"] += gd;   rows[h]["gf"] += hs
                rows[a]["gd"] -= gd;   rows[a]["gf"] += aws
                results.append(real)
                continue
            lh = float(lams[tidx[ti], tidx[tj], 0])
            la = float(lams[tidx[ti], tidx[tj], 1])
            ph, pd_, pa = _match_probs_from_lams(lh, la)
            rows[ti]["pts"] += 3 * ph + pd_
            rows[tj]["pts"] += 3 * pa + pd_
            rows[ti]["gd"] += lh - la;  rows[ti]["gf"] += lh
            rows[tj]["gd"] += la - lh;  rows[tj]["gf"] += la
            results.append((ti, tj, lh, la))
    return results, rows


def _expected_ko_winner(home: str, away: str, tidx: dict,
                        lams: np.ndarray) -> str:
    """Ganador determinista de una eliminatoria: el de mayor probabilidad
    de avanzar (90min + aproximación de prórroga/penales por lambda ratio)."""
    lh = float(lams[tidx[home], tidx[away], 0])
    la = float(lams[tidx[home], tidx[away], 1])
    lh, la = max(lh, 0.01), max(la, 0.01)
    ph, pd_, _pa = _match_probs_from_lams(lh, la)
    p_pen = float(np.clip(lh / (lh + la), 0.35, 0.65))
    p_advance = ph + pd_ * p_pen
    return home if p_advance >= 0.5 else away


KO_ROUND_NAMES = ["r32", "r16", "qf", "sf", "final"]


def simulate_tournament_detail(predictor,
                               seed: int | None = None,
                               mode: str = "sample",
                               wc_fixtures_path: Path | None = None,
                               state: dict | None = None) -> dict:
    """Simula UN torneo completo registrando todos los detalles.

    mode="sample":   una iteración Monte Carlo (marcadores sampleados).
    mode="expected": escenario más probable — standings por puntos esperados
                     y eliminatorias ganadas por el favorito del cruce.
                     Reproducible con la misma seed (el rng solo interviene
                     en la asignación de slots de terceros).

    Devuelve dict con groups (standings/results/qualified), best_thirds,
    rounds (cruces con ganador por ronda) y champion.
    """
    if mode not in ("sample", "expected"):
        raise ValueError(f"mode inválido: {mode!r}")
    if wc_fixtures_path is None:
        wc_fixtures_path = ROOT / "config" / "wc2026_fixtures.json"
    wc = json.loads(wc_fixtures_path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = wc["groups"]
    all_teams = sorted({t for g in groups.values() for t in g})
    tidx = {t: i for i, t in enumerate(all_teams)}

    lams = predictor.predict_lambdas_bulk(all_teams)
    rng = np.random.default_rng(seed)

    group_played = (state or {}).get("group_results", {})
    ko_winners = (state or {}).get("ko_winners", {})
    played_pairs = {frozenset((h, a))
                    for results in group_played.values()
                    for h, a, _hs, _as in results} | set(ko_winners)

    # ---- Fase de grupos ----
    group_details: dict[str, dict] = {}
    standing1: dict[str, str] = {}
    standing2: dict[str, str] = {}
    thirds: list[dict] = []

    for g_name, g_teams in groups.items():
        played = group_played.get(g_name)
        if mode == "expected":
            results, standings = _expected_group(g_teams, tidx, lams, played)
        else:
            results = _simulate_group_matches(g_teams, tidx, lams, rng, played)
            standings = _compute_standings(g_teams, results)
        ordered = _apply_tiebreakers(standings, rng)
        standing1[g_name] = ordered[0]
        standing2[g_name] = ordered[1]
        third = standings[ordered[2]]
        thirds.append({"team": ordered[2], "group": g_name,
                       "pts": third["pts"], "gd": third["gd"],
                       "gf": third["gf"]})
        group_details[g_name] = {
            "standings": [(t, standings[t]["pts"], standings[t]["gd"],
                           standings[t]["gf"]) for t in ordered],
            "results": results,
            "qualified": {"first": ordered[0], "second": ordered[1],
                          "third": None},  # se completa tras seleccionar terceros
        }

    best_8 = _select_best_thirds(thirds, n=8, rng=rng)
    best_8_groups = {g for _t, g in best_8}
    for g_name in groups:
        if g_name in best_8_groups:
            detail = group_details[g_name]
            detail["qualified"]["third"] = detail["standings"][2][0]

    # ---- Eliminatorias ----
    bracket = _build_r32_bracket(standing1, standing2, best_8, rng)
    rounds: dict[str, list[tuple[str, str, str]]] = {}
    current: list[tuple[str, str]] = bracket

    for round_name in KO_ROUND_NAMES:
        matches: list[tuple[str, str, str]] = []
        winners: list[str] = []
        for a, b in current:
            w = ko_winners.get(frozenset((a, b))) if ko_winners else None
            if w is None:
                if mode == "expected":
                    w = _expected_ko_winner(a, b, tidx, lams)
                else:
                    w = _simulate_ko_match(a, b, tidx, lams, rng)
            matches.append((a, b, w))
            winners.append(w)
        rounds[round_name] = matches
        current = [(winners[i], winners[i + 1])
                   for i in range(0, len(winners) - 1, 2)]

    return {
        "groups": group_details,
        "best_thirds": best_8,
        "rounds": rounds,
        "champion": rounds["final"][0][2],
        "mode": mode,
        "seed": seed,
        "played_pairs": played_pairs,   # para marcar resultados reales en UI
    }

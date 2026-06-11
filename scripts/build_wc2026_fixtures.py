"""Genera config/wc2026_fixtures.json desde el fixture real del dataset martj42.

Los grupos se derivan de las cliques del grafo de cruces y se etiquetan con
las letras oficiales FIFA (sorteo dic-2025), ancladas por un equipo de cada grupo.
Los nombres de equipos quedan EXACTAMENTE como en el dataset (clave de join).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import RESULTS_CSV, WC2026_FIXTURES

# Ancla -> letra oficial (sorteo FIFA diciembre 2025)
GROUP_ANCHORS = {
    "Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D",
    "Germany": "E", "Netherlands": "F", "Belgium": "G", "Spain": "H",
    "France": "I", "Argentina": "J", "Portugal": "K", "England": "L",
}


def main():
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")].copy()
    assert len(wc) == 72, f"Se esperaban 72 partidos de grupos, hay {len(wc)}"

    teams = set(wc.home_team) | set(wc.away_team)
    adj = {t: set() for t in teams}
    for r in wc.itertuples():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)

    groups = {}
    for anchor, letter in GROUP_ANCHORS.items():
        assert anchor in teams, f"Ancla no encontrada en fixture: {anchor}"
        groups[letter] = sorted({anchor} | adj[anchor])
        assert len(groups[letter]) == 4

    fixture = [
        {"date": r.date.strftime("%Y-%m-%d"), "home": r.home_team,
         "away": r.away_team, "city": r.city, "country": r.country}
        for r in wc.sort_values("date").itertuples()
    ]

    out = {
        "tournament": "FIFA World Cup 2026",
        "format": {"teams": 48, "groups": 12, "advance_direct": 2, "best_thirds": 8},
        "groups": {k: groups[k] for k in sorted(groups)},
        "group_fixtures": fixture,
        # El mapeo oficial de terceros al bracket R32 se añade en Sprint 1
        "r32_bracket": None,
    }
    WC2026_FIXTURES.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(f"OK -> {WC2026_FIXTURES}")
    for k in sorted(groups):
        print(f"Grupo {k}: {', '.join(groups[k])}")


if __name__ == "__main__":
    main()

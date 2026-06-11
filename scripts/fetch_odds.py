"""Descarga cuotas en vivo del Mundial desde The Odds API.

Requiere la variable de entorno ODDS_API_KEY (gratis: the-odds-api.com).
Uso: python scripts/fetch_odds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.odds_api import best_odds_per_match, fetch_wc_odds


def main():
    df = fetch_wc_odds()
    if df.empty:
        print("Sin eventos disponibles (¿mercados cerrados?)")
        return
    best = best_odds_per_match(df)
    print(f"\nMejores cuotas por partido ({len(best)} partidos):")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()

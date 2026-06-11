"""Cliente de The Odds API (the-odds-api.com) para cuotas en vivo.

Tier gratuito: 500 requests/mes. Cada llamada a /odds consume
1 request × nº de mercados × nº de regiones. Con 1-2 pulls diarios
durante el torneo (~40 días) se mantiene dentro del límite.

Configuración: variable de entorno ODDS_API_KEY, o pasar api_key explícito.

Uso:
    from src.ingestion.odds_api import fetch_wc_odds, load_latest_odds
    df = fetch_wc_odds()          # llama a la API y persiste
    df = load_latest_odds()       # lee el último snapshot sin gastar requests
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config.settings import ROOT

ODDS_DIR = ROOT / "data" / "raw" / "odds"
LATEST_PARQUET = ODDS_DIR / "latest_odds.parquet"

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

# Mapeo de nombres The Odds API → nombres del dataset martj42
TEAM_NAME_MAP = {
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Republic of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland",
    "Iran": "Iran",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
}


def _normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_wc_odds(api_key: str | None = None,
                  regions: str = "eu",
                  markets: str = "h2h",
                  sport: str = SPORT_KEY) -> pd.DataFrame:
    """Descarga cuotas 1X2 del Mundial y las persiste en data/raw/odds/.

    Devuelve DataFrame con una fila por (partido, bookmaker):
    home_team, away_team, commence_time, bookmaker, home_odds, draw_odds,
    away_odds, fetched_at.
    """
    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Falta la API key. Define la variable de entorno ODDS_API_KEY "
            "(gratis en https://the-odds-api.com).")

    resp = requests.get(
        f"{BASE_URL}/sports/{sport}/odds",
        params={"apiKey": api_key, "regions": regions, "markets": markets,
                "oddsFormat": "decimal"},
        timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    events = resp.json()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in events:
        home = _normalize_team(ev["home_team"])
        away = _normalize_team(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                odds = {o["name"]: o["price"] for o in mk["outcomes"]}
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "commence_time": ev["commence_time"],
                    "bookmaker": bk["title"],
                    "home_odds": odds.get(ev["home_team"]),
                    "draw_odds": odds.get("Draw"),
                    "away_odds": odds.get(ev["away_team"]),
                    "fetched_at": fetched_at,
                })

    df = pd.DataFrame(rows)

    # Persistir: snapshot con timestamp + alias "latest"
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    raw_path = ODDS_DIR / f"odds_{stamp}.json"
    raw_path.write_text(json.dumps(events, indent=1), encoding="utf-8")
    if not df.empty:
        df.to_parquet(LATEST_PARQUET, index=False)

    print(f"Cuotas: {len(events)} eventos, {len(df)} filas. "
          f"Requests restantes este mes: {remaining}")
    return df


def load_latest_odds() -> pd.DataFrame | None:
    """Lee el último snapshot de cuotas sin gastar requests de la API."""
    if not LATEST_PARQUET.exists():
        return None
    return pd.read_parquet(LATEST_PARQUET)


def best_odds_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Mejor cuota disponible por resultado entre todos los bookmakers
    (line shopping: maximiza el EV alcanzable)."""
    return (df.groupby(["home_team", "away_team", "commence_time"])
              .agg(home_odds=("home_odds", "max"),
                   draw_odds=("draw_odds", "max"),
                   away_odds=("away_odds", "max"),
                   n_bookmakers=("bookmaker", "nunique"))
              .reset_index())

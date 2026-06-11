"""Pipeline de ingesta: CSV crudo -> overlay manual -> Elo forward pass -> DuckDB."""
import duckdb
import pandas as pd

from config.settings import DB_PATH, ROOT, k_factor
from src.ingestion.loaders import load_results
from src.models.elo_model import run_elo_forward

SCHEMA_SQL = ROOT / "src" / "db" / "schema.sql"


def apply_manual_overlay(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica los resultados de manual_results sobre el DataFrame del CSV.

    - Match existente sin score en el CSV → escribe el resultado manual.
    - Match inexistente (p.ej. eliminatoria aún no listada) → inserta fila.
    - Si el CSV ya trae score para esa clave, el CSV gana (overlay obsoleto).

    Devuelve (df actualizado con match_id regenerado, resumen).
    """
    from src.db import remote
    if remote.is_remote():
        # Los resultados manuales viven en Supabase (deploy compartido):
        # el rebuild local los incorpora desde ahí.
        overlay = remote.get_manual_results()
    elif DB_PATH.exists():
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
            if "manual_results" not in tables:
                return df, {"updated": 0, "inserted": 0, "obsolete": 0}
            overlay = con.execute("SELECT * FROM manual_results").df()
        finally:
            con.close()
    else:
        return df, {"updated": 0, "inserted": 0, "obsolete": 0}
    if overlay.empty:
        return df, {"updated": 0, "inserted": 0, "obsolete": 0}

    df = df.copy()
    df["_key"] = (df["date"].dt.strftime("%Y-%m-%d") + "|"
                  + df["home_team"] + "|" + df["away_team"])
    key_to_idx = dict(zip(df["_key"], df.index))

    updated = inserted = obsolete = 0
    new_rows = []
    for r in overlay.itertuples():
        key = f"{pd.Timestamp(r.date):%Y-%m-%d}|{r.home_team}|{r.away_team}"
        idx = key_to_idx.get(key)
        if idx is not None:
            if pd.notna(df.at[idx, "home_score"]):
                obsolete += 1          # el CSV ya lo trae: gana el CSV
                continue
            df.at[idx, "home_score"] = r.home_score
            df.at[idx, "away_score"] = r.away_score
            updated += 1
        else:
            new_rows.append({
                "date": pd.Timestamp(r.date),
                "home_team": r.home_team, "away_team": r.away_team,
                "home_score": r.home_score, "away_score": r.away_score,
                "tournament": r.tournament, "city": None, "country": None,
                "neutral": bool(r.neutral),
                "k_weight": k_factor(r.tournament),
            })
            inserted += 1

    df = df.drop(columns=["_key"])
    if new_rows:
        df = pd.concat([df.drop(columns=["match_id"]),
                        pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
        df.insert(0, "match_id", df.index)
    return df, {"updated": updated, "inserted": inserted, "obsolete": obsolete}


def run_ingestion(verbose: bool = True) -> dict:
    """Ejecuta la ingesta completa y devuelve un resumen."""
    df = load_results()
    df, overlay_summary = apply_manual_overlay(df)
    df_elo, ratings, n_played = run_elo_forward(df)

    last_date = (
        df_elo.dropna(subset=["home_score"])
        .melt(id_vars="date", value_vars=["home_team", "away_team"], value_name="team")
        .groupby("team")["date"].max()
    )
    elo_current = pd.DataFrame({
        "team": list(ratings),
        "elo": [ratings[t] for t in ratings],
        "n_matches": [n_played.get(t, 0) for t in ratings],
    })
    elo_current["last_match_date"] = elo_current["team"].map(last_date)

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(SCHEMA_SQL.read_text(encoding="utf-8"))

        matches = df_elo[["match_id", "date", "home_team", "away_team",
                          "home_score", "away_score", "tournament",
                          "city", "country", "neutral", "k_weight"]]
        features = df_elo[["match_id", "home_elo_pre", "away_elo_pre", "neutral"]].copy()
        features["elo_delta"] = features["home_elo_pre"] - features["away_elo_pre"]
        features = features.rename(columns={"home_elo_pre": "home_elo",
                                            "away_elo_pre": "away_elo"})
        features = features[["match_id", "home_elo", "away_elo", "elo_delta", "neutral"]]

        con.register("matches_df", matches)
        con.execute("DELETE FROM matches")
        con.execute("INSERT INTO matches SELECT * FROM matches_df")
        con.register("features_df", features)
        con.execute("DELETE FROM match_features")
        con.execute("INSERT INTO match_features SELECT * FROM features_df")
        con.register("elo_df", elo_current)
        con.execute("DELETE FROM elo_current")
        con.execute("INSERT INTO elo_current SELECT * FROM elo_df")

        summary = {
            "overlay": overlay_summary,
            "matches": con.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            "played": con.execute(
                "SELECT COUNT(*) FROM matches WHERE home_score IS NOT NULL").fetchone()[0],
            "future": con.execute(
                "SELECT COUNT(*) FROM matches WHERE home_score IS NULL").fetchone()[0],
            "teams": con.execute("SELECT COUNT(*) FROM elo_current").fetchone()[0],
            "date_range": con.execute(
                "SELECT MIN(date), MAX(date) FROM matches").fetchone(),
        }
    finally:
        con.close()

    if verbose:
        ov = summary["overlay"]
        if any(ov.values()):
            print(f"Overlay manual    : {ov['updated']} actualizados, "
                  f"{ov['inserted']} insertados, {ov['obsolete']} obsoletos (CSV gana)")
        print(f"Partidos cargados : {summary['matches']:,}")
        print(f"  jugados         : {summary['played']:,}")
        print(f"  futuros (fixture): {summary['future']:,}")
        print(f"Selecciones con Elo: {summary['teams']:,}")
        print(f"Rango de fechas   : {summary['date_range'][0]} -> {summary['date_range'][1]}")
    return summary

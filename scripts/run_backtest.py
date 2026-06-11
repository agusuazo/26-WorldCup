"""Backtest exprés de calidad probabilística.

Evalúa el calibrator sobre el hold-out (2023-2026) y genera un reporte.
No requiere cuotas históricas — solo calidad probabilística.

Uso: python scripts/run_backtest.py [--start 2023-01-01] [--end 2025-12-31]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import duckdb
import joblib
import pandas as pd

from config.settings import DB_PATH, ELO_WARMUP_END, HOLDOUT_START, PROCESSED_DIR
from src.backtesting.engine import backtest_probability_quality


def run_walk_forward(args):
    """Backtest walk-forward: re-entrena cada trimestre sin leakage."""
    from src.backtesting.engine import walk_forward_backtest

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT m.match_id, m.date, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.tournament,
                   m.neutral, m.k_weight,
                   f.home_elo AS home_elo_pre, f.away_elo AS away_elo_pre
            FROM matches m JOIN match_features f USING (match_id)
            WHERE m.home_score IS NOT NULL AND m.date >= ?
            ORDER BY m.date
        """, [ELO_WARMUP_END]).df()
    finally:
        con.close()

    print(f"Walk-forward: {args.start} -> {args.end} "
          f"(re-entrenamiento trimestral, {len(df):,} partidos disponibles)\n")
    folds, summary = walk_forward_backtest(df, start=args.start, end=args.end)
    if not summary:
        print("Sin folds evaluables.")
        return

    print(f"\n--- Walk-forward global ---")
    print(f"Folds:       {summary['n_folds']}")
    print(f"Partidos:    {summary['n_matches']:,}")
    print(f"Brier Score: {summary['brier_score']:.4f}")
    print(f"Log Loss:    {summary['log_loss']:.4f}")

    folds.to_parquet(PROCESSED_DIR / "walkforward_folds.parquet", index=False)
    print(f"\nFolds guardados en {PROCESSED_DIR / 'walkforward_folds.parquet'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=HOLDOUT_START)
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Re-entrena trimestralmente (lento pero riguroso)")
    args = parser.parse_args()

    if args.walk_forward:
        run_walk_forward(args)
        return

    print(f"Backtest probabilistico: {args.start} -> {args.end}")
    t0 = time.time()

    # Cargar datos
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT m.match_id, m.date, m.home_team, m.away_team,
                   m.home_score, m.away_score, m.tournament, m.neutral
            FROM matches m
            WHERE m.home_score IS NOT NULL AND m.date >= ?
            ORDER BY m.date
        """, [ELO_WARMUP_END]).df()
    finally:
        con.close()

    # Cargar predictor
    cal_path = PROCESSED_DIR / "calibrator.joblib"
    ens_path = PROCESSED_DIR / "ensemble.joblib"
    if cal_path.exists():
        predictor = joblib.load(cal_path)
        name = "Calibrador multinomial"
    elif ens_path.exists():
        predictor = joblib.load(ens_path)
        name = "Ensemble (sin calibrar)"
    else:
        print("No se encontraron modelos entrenados. Ejecuta run_training.py primero.")
        sys.exit(1)

    print(f"Predictor: {name}")

    results = backtest_probability_quality(predictor, df,
                                           start=args.start, end=args.end)
    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print(f"\n--- Resultados backtest probabilistico ({args.start} - {args.end}) ---")
    print(f"Partidos evaluados: {results['n_matches']:,}")
    print(f"Brier Score:        {results['brier_score']:.4f}  (baseline uniforme=0.2222)")
    print(f"Log Loss:           {results['log_loss']:.4f}")
    print(f"\nBrier por año:")
    for _, row in results["by_year"].iterrows():
        print(f"  {int(row['year'])}: {row['brier']:.4f}  (n={int(row['n'])})")

    # Guardar para la página Streamlit
    out_path = PROCESSED_DIR / "backtest_results.parquet"
    by_year_path = PROCESSED_DIR / "backtest_by_year.parquet"
    cal_path_out = PROCESSED_DIR / "calibration_curve.parquet"
    results["by_year"].to_parquet(by_year_path, index=False)
    results["calibration"].to_parquet(cal_path_out, index=False)

    summary_df = pd.DataFrame([{
        "start": args.start, "end": args.end,
        "n_matches": results["n_matches"],
        "brier_score": results["brier_score"],
        "log_loss": results["log_loss"],
        "predictor": name,
    }])
    summary_df.to_parquet(out_path, index=False)
    print(f"\nResultados guardados en {PROCESSED_DIR}")
    print(f"Tiempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

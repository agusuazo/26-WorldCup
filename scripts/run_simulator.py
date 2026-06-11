"""CLI: ejecuta el simulador Monte Carlo del WC 2026 y guarda resultados.

Uso:
  python scripts/run_simulator.py             # 10_000 simulaciones
  python scripts/run_simulator.py --n 50000   # más simulaciones

Los resultados se guardan en data/processed/sim_results.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from config.settings import PROCESSED_DIR
from src.simulation.monte_carlo import run_monte_carlo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unconditioned", action="store_true",
                    help="Ignora los resultados reales del torneo (simula desde cero)")
    args = ap.parse_args()

    print(f"Cargando ensemble...")
    predictor_path = PROCESSED_DIR / "calibrator.joblib"
    if not predictor_path.exists():
        predictor_path = PROCESSED_DIR / "ensemble.joblib"
    if not predictor_path.exists():
        # Fallback: blend Elo+Poisson del Sprint 0
        import duckdb
        from config.settings import DB_PATH
        from src.models.elo_model import EloPredictor
        from src.models.ensemble import EnsemblePredictor
        from src.models.poisson_model import GlobalPoissonModel
        draw = joblib.load(PROCESSED_DIR / "draw_model.joblib")
        pois = joblib.load(PROCESSED_DIR / "poisson_model.joblib")
        con = duckdb.connect(str(DB_PATH), read_only=True)
        ratings = dict(con.execute("SELECT team, elo FROM elo_current").fetchall())
        con.close()
        elo = EloPredictor(ratings, draw)
        predictor = EnsemblePredictor(elo, pois)
    else:
        predictor = joblib.load(predictor_path)

    state = None
    if not args.unconditioned:
        from src.simulation.monte_carlo import load_tournament_state
        state = load_tournament_state()
        n_cond = state["n_group_played"] + state["n_ko_played"]
        if n_cond:
            print(f"Condicionado a {n_cond} partidos reales "
                  f"({state['n_group_played']} de grupos, {state['n_ko_played']} KO)")
        else:
            print("Sin partidos reales del torneo en la BD (simulación desde cero)")

    print(f"Simulando {args.n:,} torneos...")

    def progress(p):
        print(f"  {p:.0%}", end="\r", flush=True)

    df = run_monte_carlo(predictor, n=args.n, seed=args.seed,
                         progress_cb=progress, state=state)
    print()

    out = PROCESSED_DIR / "sim_results.parquet"
    df.to_parquet(out, index=False)
    print(f"\nTop 10 favoritos al título:")
    print(df[["rank", "team", "champion_pct", "finalist_pct",
              "sf_pct", "qf_pct", "r32_pct"]].head(10).to_string(index=False))
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

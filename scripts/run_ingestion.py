"""CLI: ejecuta la ingesta completa (CSV -> Elo -> DuckDB).

Uso: python scripts/run_ingestion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion()

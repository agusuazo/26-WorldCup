"""Helper invocado por publish.ps1 — evita pelear con el quoting anidado de
PowerShell para un one-liner. No se usa directamente."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.updater import refresh_all

s = refresh_all(n_sims=10000)
gate = "PASA" if s["gate_passed"] else "NO PASA"
print(f"Brier: {s['brier_holdout']:.4f} | gate: {gate} | "
      f"condicionado a {s['n_group_played']}+{s['n_ko_played']} partidos")

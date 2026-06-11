"""Configuración central del sistema. Todas las constantes viven aquí."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "mundial.duckdb"
RESULTS_CSV = ROOT / "data" / "raw" / "results" / "results.csv"
SHOOTOUTS_CSV = ROOT / "data" / "raw" / "results" / "shootouts.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
WC2026_FIXTURES = ROOT / "config" / "wc2026_fixtures.json"

# ---------------- Elo ----------------
ELO_INITIAL = 1500.0
ELO_HOME_ADV = 80.0  # puntos Elo por localía real (0 en cancha neutral)

# K-factor por importancia del torneo (metodología World Football Elo)
K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0       # Euro, Copa América, AFCON, Asian Cup, Gold Cup
K_QUALIFIER = 40.0         # clasificatorias y Nations League
K_MINOR = 30.0             # torneos menores
K_FRIENDLY = 20.0

_CONTINENTAL_KEYWORDS = (
    "UEFA Euro", "Copa América", "African Cup of Nations",
    "Africa Cup of Nations", "AFC Asian Cup", "Gold Cup",
    "CONCACAF Championship", "Confederations Cup", "Copa America",
)


def k_factor(tournament: str) -> float:
    """K-factor de Elo según la importancia del torneo."""
    t = tournament or ""
    if "qualification" in t:
        return K_QUALIFIER
    if t == "FIFA World Cup":
        return K_WORLD_CUP
    if any(kw in t for kw in _CONTINENTAL_KEYWORDS):
        return K_CONTINENTAL
    if "Nations League" in t:
        return K_QUALIFIER
    if t == "Friendly":
        return K_FRIENDLY
    return K_MINOR


# ---------------- Entrenamiento / validación ----------------
TRAIN_END = "2022-12-31"      # draw model y Poisson se ajustan hasta aquí
HOLDOUT_START = "2023-01-01"  # hold-out para el gate de Brier
ELO_WARMUP_END = "1980-01-01" # partidos previos solo calibran Elo, no entrenan modelos

# Gate para usar dinero real (Brier multiclase promediado; uniforme = 0.2222)
BRIER_GATE = 0.2000

# ---------------- Apuestas ----------------
KELLY_FRACTION = 0.125   # 1/8 Kelly las primeras 2 semanas (plan: riesgo operacional)
KELLY_CAP = 0.05         # máximo 5% del bankroll por apuesta
EV_THRESHOLD_LOW = 0.00
EV_THRESHOLD_MED = 0.05
EV_THRESHOLD_HIGH = 0.10
EV_THRESHOLD_OUTRIGHT = 0.15  # umbral más exigente para mercado de campeón

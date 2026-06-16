# WC26 Quant — Sistema Predictivo Mundial FIFA 2026

Sistema cuantitativo de predicción y apuestas para el Mundial 2026: modelos Elo,
Poisson y Dixon-Coles ensamblados y calibrados, simulación Monte Carlo del bracket
oficial, detección de EV+ y gestión de banca Kelly fraccional.

## Arquitectura

```
Vercel (React/TanStack)  ──X-API-Key──►  Render (FastAPI, Docker)
                                           ├─ lectura : DuckDB + modelos (horneados)
                                           └─ escritura: Supabase Postgres (bets, resultados)

Tu PC: re-entrenamiento → scripts/publish.ps1 → git push → redeploy automático
```

**Repositorios:**
- Backend (este repo): modelos, API, Streamlit local
- Frontend: repo independiente de Lovable (`frontend/` — excluido de git)

## Estructura

```
api/            FastAPI — backend de producción (deployado en Render)
dashboard/      Streamlit — dashboard local de análisis
src/            lógica de negocio compartida
  backtesting/  backtesting de modelos y apuestas
  betting/      EV, Kelly, bet log, bankroll
  db/           DuckDB + capa Supabase para deploy
  ingestion/    pipeline CSV→DuckDB, descarga martj42, updater
  models/       Elo, Poisson, Dixon-Coles, ensemble, calibración
  simulation/   Monte Carlo con bracket oficial FIFA R32
config/         settings.py, wc2026_fixtures.json
data/
  db/           mundial.duckdb (horneado en Docker)
  processed/    modelos .joblib, sim_results.parquet, backtest
  raw/results/  results.csv + shootouts.csv (martj42)
scripts/        CLI: ingesta, entrenamiento, simulador, odds, publish
tests/          43 tests (pytest)
docs/           DEPLOY.md — runbook completo Supabase + Render + Vercel
frontend/       repo Lovable (excluido de este git — repo aparte)
Dockerfile      imagen Docker para Render
render.yaml     Blueprint de Render (infra como código)
```

## Uso local

```bash
pip install -r requirements.txt

# Ingesta + entrenamiento + simulación
python scripts/run_ingestion.py
python scripts/run_training.py
python scripts/run_simulator.py

# Dashboard Streamlit
streamlit run dashboard/main.py

# API FastAPI (para desarrollo)
uvicorn api.main:app --reload

# Tests
pytest tests/ -q
```

## Ciclo durante el torneo

```powershell
# Tras cada jornada: recalcula todo localmente y publica al deploy
.\scripts\publish.ps1
```

Hace: descarga resultados → rebuild Elo → re-entrena modelos → simula Monte Carlo
condicionado → commit artefactos → push → Render redespliega (~10 min).

## Deploy (free tier)

Ver [docs/DEPLOY.md](docs/DEPLOY.md) para el runbook completo.

| Capa | Servicio | Coste |
|---|---|---|
| Frontend | Vercel | gratis |
| Backend API | Render (Docker) | gratis |
| Estado mutable | Supabase Postgres | gratis |

## Modelos

| Modelo | Brier hold-out | Notas |
|---|---|---|
| Elo + Poisson + DC (ensemble) | 0.1652 | gate < 0.20 superado |
| Walk-forward trimestral | 0.1713 | sin leakage temporal |
| Base rate (uniforme) | 0.2222 | referencia |

## Protocolo de riesgo

1. Paper trading la primera semana (`paper=TRUE` en bet_log).
2. Dinero real solo con Brier < 0.20 ✅, 1/8 Kelly, cap 5% por apuesta.
3. Mercados outright: método Shin + umbral EV > 15%.

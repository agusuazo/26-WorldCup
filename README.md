# Sistema Predictivo — Mundial FIFA 2026 (EV+)

Sistema cuantitativo de apuestas para el Mundial 2026: estima probabilidades
reales de partidos internacionales, las compara contra cuotas y detecta valor
esperado positivo (EV+) con gestión de banca Kelly fraccional.

## Estado: Sprint 0 completado ✅

- **Datos:** 49.405 partidos internacionales (1872-2026, dataset martj42) + fixture real WC 2026 en DuckDB.
- **Modelos:** Elo propio (K por torneo, multiplicador por margen) + modelo logístico de empate + Poisson global.
- **Validación:** Brier 0.1696 en hold-out 2023-2026 (3.598 partidos) — gate de calidad superado (uniforme = 0.2222, base rates = 0.2120).
- **Motor de apuestas:** remoción de vig (multiplicativa / potencia / Shin), EV, Kelly 1/8 con cap 5%.
- **Dashboard:** Streamlit con ranking Elo, grupos oficiales y match predictor.

## Uso

```bash
pip install -r requirements.txt

python scripts/run_ingestion.py        # CSV -> Elo -> DuckDB
python scripts/run_training.py         # entrena + valida Brier + persiste modelos
python scripts/build_wc2026_fixtures.py  # regenera fixture/grupos WC 2026

streamlit run app/main.py              # dashboard
pytest tests/ -q                       # tests
```

Para actualizar resultados durante el torneo: re-descargar
`results.csv` desde github.com/martj42/international_results a
`data/raw/results/` y re-ejecutar ingesta + entrenamiento.

## Protocolo de riesgo (obligatorio)

1. **Paper trading** la primera semana (campo `paper=TRUE` en `bet_log`).
2. Dinero real solo con: Brier hold-out < 0.20 ✅, sin EV+ sistemático en longshots, stakes a 1/8 Kelly.
3. Mercados outright: vig con método de Shin y umbral EV > 15%.

## Próximo (Sprint 1, antes de knockouts ~28-jun)

- Dixon-Coles regularizado + calibración multinomial + ensemble.
- Simulador Monte Carlo del torneo completo (bracket R32 con mapeo oficial de terceros).
- Página Tournament Simulator + cuotas en vivo (The Odds API).
- Backtest exprés con cuotas históricas WC 2018/2022.

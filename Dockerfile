# Backend WC26 Quant — imagen para Render (free tier).
# Los artefactos (DuckDB, modelos joblib, parquets) viajan horneados:
# son de solo lectura en producción; el estado mutable vive en Supabase.
FROM python:3.12-slim

WORKDIR /app

# Solo las dependencias del API (sin streamlit/plotly/pytest: ahorra ~200MB)
RUN pip install --no-cache-dir \
    "duckdb>=1.0" "pandas>=2.0" "numpy>=1.26" "scipy>=1.11" \
    "scikit-learn>=1.4" "requests>=2.31" "fastapi>=0.115" \
    "uvicorn>=0.30" "psycopg2-binary>=2.9" "pyarrow>=15.0" "joblib>=1.3"

COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY data/db/mundial.duckdb data/db/
COPY data/processed/ data/processed/
COPY data/raw/results/ data/raw/results/

ENV SERVER_LIGHT=1
ENV PYTHONUNBUFFERED=1

# Render inyecta PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

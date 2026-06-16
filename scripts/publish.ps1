# Recalcula todo localmente y publica los artefactos frescos al deploy.
#
# Flujo free-tier: el re-entrenamiento corre en este PC (Render free no puede);
# los artefactos (DuckDB, modelos, simulación) se commitean y el push dispara
# el redeploy automático de Render con los modelos nuevos.
#
# Requiere: repo git inicializado con remote 'origin' (ver docs/DEPLOY.md).
# Si SUPABASE_DB_URL está en .env, el recálculo incorpora los resultados
# manuales que tus amigos cargaron en la web.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Cargar .env si existe (SUPABASE_DB_URL, ODDS_API_KEY...)
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match "^\s*[^#].*=" } | ForEach-Object {
        $k, $v = $_ -split "=", 2
        [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim())
    }
    Write-Host "Variables de .env cargadas" -ForegroundColor DarkGray
}

Write-Host "`n[1/3] Recalculo completo (ingesta + Elo + modelos + simulacion)..." -ForegroundColor Cyan
python scripts/_publish_run.py
if (-not $?) { throw "Recalculo fallido" }

Write-Host "`n[2/3] Commit de artefactos..." -ForegroundColor Cyan
git add data/db/mundial.duckdb data/processed data/raw/results/results.csv data/raw/results/shootouts.csv
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "Recalculo $stamp - modelos y simulacion actualizados" --allow-empty

Write-Host "`n[3/3] Push (dispara redeploy de Render)..." -ForegroundColor Cyan
git push origin main

Write-Host "`nListo. Render redespliega en ~5-10 min con los modelos frescos." -ForegroundColor Green

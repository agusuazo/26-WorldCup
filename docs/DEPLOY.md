# Deploy WC26 Quant — Vercel + Render + Supabase (free tiers)

## Arquitectura

```
Vercel (frontend)  ──X-API-Key──►  Render (FastAPI, Docker)
                                      │ lectura: DuckDB + modelos horneados en la imagen
                                      └ escritura: Supabase Postgres (bets, resultados, config)

Tu PC (admin): re-entrenamiento → scripts/publish.ps1 → git push → Render redespliega
```

**Qué puede hacer cada quien:**
- Tus amigos y tú (web): predicciones, EV, registrar apuestas, cargar resultados,
  re-simular ("Recalcular" = re-simulación condicionada, ~30-60 s).
- Solo tú (PC): re-entrenamiento completo de modelos + actualización del dataset
  → `scripts/publish.ps1` (1 comando, ~5 min + redeploy).

**Limitación free-tier asumida:** Render duerme tras 15 min sin uso → la primera
petición tarda ~1 min (cold start + carga de modelos). Después va fluido.

---

## Paso 1 — Supabase (estado mutable)

1. Crear cuenta en [supabase.com](https://supabase.com) → **New project** (región EU).
2. En **Project Settings → Database → Connection string → URI**, copiar la URL
   (formato `postgresql://postgres.[ref]:[PASSWORD]@...pooler.supabase.com:5432/postgres`).
   Usa la variante **Session pooler** (puerto 5432) — psycopg2 no soporta el transaction
   pooler para todo.
3. No hace falta crear tablas: el backend las crea al primer arranque
   (`ensure_schema()`).

## Paso 2 — Repo del backend en GitHub

El frontend ya tiene su repo (el de Lovable). El backend necesita el suyo:

```powershell
cd "c:\Users\UA\Desktop\Proyectos\Mundial FIFA"
git init -b main
git add .
git commit -m "WC26 Quant backend + artefactos"
# Crear repo en GitHub (privado) llamado p.ej. wc26-quant-api y luego:
git remote add origin https://github.com/TU_USUARIO/wc26-quant-api.git
git push -u origin main
```

> El `.gitignore` ya excluye `frontend/` (repo aparte), secretos y backups.
> Los artefactos (DuckDB 18 MB + modelos) SÍ van al repo — es el mecanismo de deploy.

## Paso 3 — Render (backend)

1. Cuenta en [render.com](https://render.com) → **New → Blueprint** → conectar el
   repo `wc26-quant-api`. Render lee `render.yaml` y crea el servicio Docker.
2. En **Environment**, definir los 2 secretos:
   - `SUPABASE_DB_URL` = la connection string del Paso 1
   - `API_ACCESS_KEY` = una clave compartida que tú inventes (ej. frase larga);
     es la que les darás a tus amigos
3. Deploy automático (~10 min la primera vez). Verificar:
   `https://wc26-quant-api.onrender.com/` debe responder `{"ok": true}`.
4. Probar auth: `https://.../api/status` sin header debe dar 401.

## Paso 4 — Vercel (frontend)

1. Cuenta en [vercel.com](https://vercel.com) → **Add New → Project** → importar
   el repo de Lovable (`wc26-quant-dashboard`).
2. En **Environment Variables** (build):
   - `VITE_API_BASE` = `https://wc26-quant-api.onrender.com` (la URL real de Render)
   - `NITRO_PRESET` = `vercel`  (el config de Lovable usa cloudflare por defecto)
3. Deploy. La URL de Vercel (`https://wc26-quant.vercel.app`) es la que compartes.
4. Antes: commitea y pushea los cambios del frontend (api.ts con X-API-Key,
   data.tsx con el stepper real) al repo de Lovable.

## Paso 5 — Tu PC (admin)

Crear `.env` en la raíz del proyecto (gitignored):

```
SUPABASE_DB_URL=postgresql://postgres...
ODDS_API_KEY=...        # opcional
```

Con `SUPABASE_DB_URL` en el entorno, **tu Streamlit y scripts locales comparten
el mismo bet log y resultados manuales que la web** — una sola fuente de verdad.

### Ciclo de actualización (tras cada jornada)

```powershell
.\scripts\publish.ps1
```

Hace: recálculo completo local (incorpora resultados de Supabase) → commit de
artefactos → push → Render redespliega con modelos frescos (~10 min).

---

## Uso para tus amigos

1. Les pasas la URL de Vercel + la `API_ACCESS_KEY`.
2. La primera vez, la app les pide la clave (se guarda en su navegador).
3. Posible primera carga lenta (~1 min) si Render estaba dormido.

## Solución de problemas

| Síntoma | Causa probable |
|---|---|
| Primera petición tarda 1 min | Cold start de Render free (normal) |
| 401 persistente | Clave mal escrita — borrar localStorage o re-ingresarla |
| "Recalcular" no re-entrena | Por diseño en free tier: re-entrena con publish.ps1 |
| Build de Vercel falla en nitro | Verificar `NITRO_PRESET=vercel` en env vars |
| Error Postgres al boot | `SUPABASE_DB_URL` mal copiada o pooler de transacción (usar puerto 5432) |

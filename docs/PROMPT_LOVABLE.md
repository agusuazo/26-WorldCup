# Prompt para Lovable — Frontend del Sistema Predictivo WC 2026

> Copiar y pegar todo lo que sigue en Lovable. El frontend se conectará después
> a un backend FastAPI local; por eso es crítico que TODA la capa de datos viva
> en un único cliente API con mocks intercambiables.

---

Construye una aplicación web de analítica de apuestas deportivas para el **Mundial FIFA 2026**, llamada **"WC26 Quant"**. Es un dashboard personal (un solo usuario, sin login) que muestra predicciones de un modelo estadístico propio, las compara contra cuotas de casas de apuestas para detectar valor esperado positivo (EV+), simula el torneo completo y gestiona un bankroll con criterio Kelly.

## Diseño

**Minimalista pero atractivo, estética tipo terminal financiera moderna (Linear/Vercel) aplicada al fútbol:**

- **Tema oscuro por defecto** (sin toggle de tema, solo oscuro).
- **Paleta**:
  - Fondo base: azul-carbón muy oscuro `#0A0F1A`
  - Superficies/cards: `#111827` con borde sutil `#1F2937`
  - Acento primario (verde césped/éxito/EV+): esmeralda `#10B981`
  - Acento secundario (campeón, destacados premium): dorado `#D4A017`
  - Negativo/EV-: rojo coral `#F87171`; advertencia: ámbar `#FBBF24`
  - Texto: `#F9FAFB` principal, `#9CA3AF` secundario
- **Tipografía**: Inter para UI; números tabulares (tabular-nums) en todas las tablas y métricas — esto es una app de números.
- **Layout**: sidebar izquierda fija y delgada (solo iconos + tooltip, expandible) con navegación; contenido con max-width amplio; cards con `border-radius` 12px, sin sombras pesadas — bordes finos y jerarquía por espaciado.
- **Microinteracciones sobrias**: transiciones de 150ms, hover sutil en filas de tabla, skeleton loaders mientras carga data. Nada de animaciones llamativas.
- **Todo el texto de la UI en español.**
- Responsive: usable en móvil (sidebar colapsa a bottom-nav), pero optimizado para desktop.
- Gráficos con **Recharts**, con los colores de la paleta y grids muy tenues.

## Arquitectura de datos (CRÍTICO para la integración posterior)

- Centraliza **toda** la obtención de datos en `src/lib/api.ts` con interfaces TypeScript explícitas para cada respuesta.
- `const API_BASE = import.meta.env.VITE_API_BASE ?? null` — si `API_BASE` es null, cada función devuelve **datos mock realistas** (defínelos en `src/lib/mocks.ts`); si está definido, hace `fetch` al endpoint real. Así la conexión al backend será solo definir una variable de entorno.
- Usa TanStack Query (react-query) para fetching, caché e invalidación.

### Contrato de API (el backend FastAPI expondrá exactamente esto)

```typescript
// GET /api/status
interface Status {
  lastRefresh: string | null;        // ISO datetime
  brierHoldout: number;              // ej. 0.1652
  gatePassed: boolean;
  groupMatchesPlayed: number;        // partidos reales que condicionan la simulación
  koMatchesPlayed: number;
  bankroll: number;
  paperMode: boolean;
}

// GET /api/teams
interface Team { name: string; group: string; elo: number; }

// GET /api/matches/upcoming
interface UpcomingMatch {
  matchId: number; date: string;
  home: string; away: string;
  probs: { home: number; draw: number; away: number };  // 0-1
  bestEv: number | null;             // mejor EV si hay cuotas, ej. 0.07
  bestBet: string | null;            // ej. "Mexico @ 2.10"
  signal: "HIGH" | "MEDIUM" | "LOW" | "NONE" | null;
}

// POST /api/predict  body: { home, away, neutral }
interface Prediction {
  probs: { home: number; draw: number; away: number };
  lambdas: { home: number; away: number };              // goles esperados
  elo: { home: number; away: number };
  scoreMatrix: number[][];                              // 7x7, probabilidades 0-1
  byModel: { name: string; home: number; draw: number; away: number }[];
}

// POST /api/ev  body: { home, away, neutral, odds: {home,draw,away}, bankroll }
interface EvAnalysis {
  rows: { outcome: string; modelProb: number; impliedProb: number;
          edge: number; ev: number; signal: string; kellyStake: number }[];
  recommendation: {
    action: "BET" | "MARGINAL" | "PASS";
    outcome: string | null; odds: number | null;
    ev: number | null; stake: number | null; kellyPct: number | null;
    reason: string;
  };
  overround: number;
}

// GET /api/simulation?conditioned=true
interface SimRow {
  rank: number; team: string; group: string;
  championPct: number; finalistPct: number; sfPct: number;
  qfPct: number; r16Pct: number; r32Pct: number; groupExitPct: number;  // acumuladas, 0-100
}

// GET /api/bracket?mode=expected|sample&seed=42&conditioned=true
interface Bracket {
  mode: "expected" | "sample"; seed: number; champion: string;
  groups: Record<string, {
    standings: { team: string; pts: number; gd: number; gf: number }[];
    qualified: { first: string; second: string; third: string | null };
  }>;
  rounds: Record<"r32"|"r16"|"qf"|"sf"|"final",
                 { a: string; b: string; winner: string; isReal: boolean }[]>;
}

// GET /api/bets · POST /api/bets · PATCH /api/bets/:id · DELETE /api/bets/:id
interface Bet {
  betId: number; placedAt: string;
  homeTeam: string; awayTeam: string;
  market: "1X2-home" | "1X2-draw" | "1X2-away" | "outright";
  stake: number; odds: number; closingOdds: number | null;
  modelProb: number; ev: number; kellyFraction: number;
  paper: boolean; result: "pending" | "win" | "lose" | "void";
  profit: number | null;
  clv: number | null;                // odds/closingOdds - 1
}
// PATCH body: { action: "win" | "lose" | "void" } o { closingOdds: number }

// GET /api/bankroll
interface BankrollSummary {
  current: number; initial: number; netProfit: number;
  roi: number; winRate: number; profitFactor: number; maxDrawdown: number;
  equityCurve: number[];
  clv: { avg: number; beatClosePct: number; n: number } | null;
}

// GET /api/backtest
interface Backtest {
  brier: number; logLoss: number; nMatches: number;
  byYear: { year: number; brier: number; n: number }[];
  calibration: { meanPred: number; actualFreq: number; count: number }[];
}

// POST /api/results  body: { date, home, away, homeScore, awayScore, winner? }
// POST /api/data/download   → { rowsNew: number }
// POST /api/refresh          → { jobId }  ·  GET /api/refresh/status → { stage, done }
```

## Páginas (rutas)

### 1. `/` — Dashboard
- Fila de 4 KPIs grandes: Bankroll (con delta vs inicial), Brier del modelo (con badge verde "GATE OK" si `gatePassed`), Partidos condicionando la simulación, Último recálculo (tiempo relativo, ej. "hace 2 h").
- Tabla "Próximos partidos" con probabilidades como **barras horizontales apiladas tricolor** (verde local / gris empate / coral visitante) dentro de la celda, y columna de señal EV con badge (HIGH dorado, MEDIUM esmeralda, LOW ámbar tenue).
- Card "Favoritos al título": top-8 del simulador como barras horizontales, el #1 con acento dorado.
- Card compacta de curva de equity del bankroll.

### 2. `/predictor` — Match Predictor
- Dos selectores de equipo grandes lado a lado con "VS" en el centro (con bandera-emoji si es viable), toggle "cancha neutral".
- Resultado: 3 cards de probabilidad (local/empate/visita) + goles esperados.
- Heatmap del marcador (matriz 7×7) en escala de esmeralda.
- Sección "Cuotas": 3 inputs numéricos → tabla de EV por resultado (prob modelo, prob implícita, edge, EV, stake Kelly).
- **Card de recomendación** prominente según `recommendation.action`: BET (borde esmeralda, "APOSTAR: X @ 2.10 — EV +8.2%, stake 24.50"), MARGINAL (ámbar), PASS (gris, "No apostar — pasar también es una decisión correcta"). Botón "Registrar apuesta (paper)" cuando action ≠ PASS.

### 3. `/simulator` — Simulador del Torneo
- Tabla de 48 equipos con toggle de vista: "Acumulada (llega al menos)" / "Exclusiva (ronda de salida)" — en exclusiva cada fila suma 100% y se muestra como **barra apilada de 7 segmentos** además de los números.
- Toggle "Condicionar a resultados reales" con badge "📌 N partidos reales".
- Tab de cuotas outright: textarea para pegar "Equipo: cuota" por línea → tabla EV con método Shin, umbral 15%.

### 4. `/bracket` — Llaves
- Vista horizontal del bracket completo: R32 → Octavos → Cuartos → Semis → Final → Campeón (scroll horizontal suave en pantallas chicas).
- Cada cruce es una mini-card: ganador en esmeralda con ✓, partidos reales con chip "📌 real".
- Selector de modo ("Más probable" / "Aleatorio" con botón re-simular 🎲) y selector de equipo a resaltar — su camino se ilumina en dorado a través de todo el árbol.
- Arriba, grid colapsable 4×3 con las tablas finales de los 12 grupos (✅ clasifica, 3️⃣ mejor tercero, ❌ eliminado).
- Card final del campeón con fondo dorado sutil.

### 5. `/bankroll` — Bankroll & Apuestas
- KPIs: bankroll, ROI, win rate, profit factor, max drawdown.
- **Bloque CLV destacado** (es la métrica estrella): CLV medio, % de veces que le gana al cierre, n; con mensaje interpretativo (positivo sostenido = edge real → verde; negativo = no pasar a dinero real → rojo).
- Curva de equity (área esmeralda con línea de bankroll inicial punteada).
- Tabla de apuestas con acciones por fila: liquidar ganada/perdida, anular, registrar cuota de cierre (popover con input), borrar. Columna CLV con color por signo.
- Formulario "Nueva apuesta" en un sheet/drawer lateral.
- Switch global "Paper / Real" con advertencia al cambiar a Real.

### 6. `/data` — Actualizar Datos
- Timeline vertical del torneo: partidos jugados (marcador) y pendientes.
- Botón "Descargar dataset (GitHub)" con resultado ("+N filas nuevas").
- Formulario de resultado manual: partido pendiente → marcador → si empate en eliminatorias, selector de ganador por penales.
- Botón primario grande "🔄 Recalcular todo" → muestra stepper de progreso con las 3 fases (Ingesta + Elo → Re-entrenamiento → Simulación condicionada) consultando `/api/refresh/status`, y al terminar un resumen (Brier nuevo, favoritos top-5).

### 7. `/model` — Calidad del Modelo
- Brier score con gauge/indicador vs baseline 0.2222 y umbral 0.20.
- Barras de Brier por año y curva de calibración (diagonal punteada de referencia, puntos con tamaño según nº de muestras).
- Card estática explicando el protocolo: paper trading 1 semana → gate Brier < 0.20 → CLV positivo → 1/8 Kelly cap 5%.

## Detalles finales
- Formato numérico es-ES: probabilidades "62,3 %", dinero "1.250,00 €", cuotas con 2 decimales.
- Estados vacíos cuidados en cada página ("Sin apuestas registradas todavía", etc.) con icono tenue y CTA.
- Toasts para confirmaciones (apuesta registrada, resultado guardado, recálculo completo).
- No implementes lógica estadística en el frontend: todo viene del API; los mocks deben ser estáticos pero realistas (usa equipos reales: Brazil 19.8% campeón, France 15.3%, Argentina 14.7%, Netherlands 11.6%, Spain 8.4%...).

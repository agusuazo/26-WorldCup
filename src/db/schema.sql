-- Esquema principal. Claves naturales por nombre de equipo (dataset martj42).

CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY,
    date        DATE NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER,            -- NULL = partido futuro
    away_score  INTEGER,
    tournament  TEXT,
    city        TEXT,
    country     TEXT,
    neutral     BOOLEAN DEFAULT FALSE,
    k_weight    DOUBLE              -- K-factor aplicado por importancia
);

-- Elo pre-partido materializado (feature store, evita leakage por construcción)
CREATE TABLE IF NOT EXISTS match_features (
    match_id     INTEGER PRIMARY KEY,
    home_elo     DOUBLE,
    away_elo     DOUBLE,
    elo_delta    DOUBLE,            -- home - away, sin ajuste de localía
    neutral      BOOLEAN
);

-- Snapshot de Elo vigente por selección (para inferencia)
CREATE TABLE IF NOT EXISTS elo_current (
    team            TEXT PRIMARY KEY,
    elo             DOUBLE,
    n_matches       INTEGER,
    last_match_date DATE
);

CREATE TABLE IF NOT EXISTS predictions (
    match_id    INTEGER,
    model_name  TEXT,
    prob_home   DOUBLE,
    prob_draw   DOUBLE,
    prob_away   DOUBLE,
    xg_home     DOUBLE,
    xg_away     DOUBLE,
    created_at  TIMESTAMP DEFAULT current_timestamp
);

-- Resultados ingresados a mano (overlay sobre el CSV de martj42).
-- Sobrevive al wipe & rebuild del pipeline; clave natural porque match_id
-- es posicional y cambia cuando el CSV crece. Si el CSV llega a traer el
-- resultado para la misma clave, el CSV gana y la fila queda obsoleta.
CREATE TABLE IF NOT EXISTS manual_results (
    date        DATE NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER NOT NULL,
    away_score  INTEGER NOT NULL,
    winner      TEXT,               -- solo eliminatorias con empate (penales)
    tournament  TEXT DEFAULT 'FIFA World Cup',
    neutral     BOOLEAN DEFAULT TRUE,
    entered_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS bet_log (
    bet_id      INTEGER,
    placed_at   TIMESTAMP DEFAULT current_timestamp,
    home_team   TEXT,
    away_team   TEXT,
    market      TEXT,               -- '1X2-home' | '1X2-draw' | '1X2-away' | 'outright'
    stake       DOUBLE,
    odds        DOUBLE,
    model_prob  DOUBLE,
    ev          DOUBLE,
    kelly_fraction DOUBLE,
    paper       BOOLEAN DEFAULT TRUE,  -- TRUE = paper trading
    result      TEXT DEFAULT 'pending',
    profit      DOUBLE,
    closing_odds DOUBLE               -- cuota de cierre del mercado (para CLV)
);

-- ═══════════════════════════════════════════════════════════════
-- init.sql — Schéma de la base de données du projet scoring crédit
-- Exécuté automatiquement par docker-compose au démarrage de Postgres
-- ═══════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────
-- TABLE clients : scores PRÉCALCULÉS (façon B)
-- Remplie par le batch. Lue par la route GET /predict/{sk_id_curr}
-- Objectif : renvoyer un score instantané pour un client connu
-- ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clients (
    sk_id_curr   BIGINT       PRIMARY KEY,   -- ID client, unique (1 ligne/client)
    features     JSONB        NOT NULL,      -- les 81 features (dict JSON), utiles pour SHAP + re-scoring
    proba        REAL         NOT NULL,      -- probabilité de défaut (sortie modèle)
    decision     SMALLINT     NOT NULL,      -- 0 = accordé, 1 = refusé (seuil 0,5)
    date_calcul  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP  -- date du précalcul batch
);


-- ───────────────────────────────────────────────────────────────
-- TABLE logs : journal des appels API en PRODUCTION (monitoring)
-- Écrite à chaque appel (REUSSI ou EN ERREUR). Lue par le drift + dashboard
-- Objectif : stocker inputs + outputs + temps (exigé par l'énoncé)
-- ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS logs (
    id                  BIGSERIAL  PRIMARY KEY,  -- ID auto-incrémenté
    sk_id_curr          BIGINT,                  -- ID client (NULL si nouveau/erreur)
    inputs              JSONB      NOT NULL,      -- features reçues (pour le drift)
    proba               REAL,                    -- proba renvoyée (NULL si erreur)
    decision            SMALLINT,                -- décision (NULL si erreur)
    temps_inference_ms  REAL,                    -- temps modèle en ms (NULL si erreur)
    latence_totale_ms   REAL       NOT NULL,     -- latence totale (toujours mesurée)
    statut              TEXT       NOT NULL DEFAULT 'ok',  -- 'ok' ou 'erreur'
    code_http           INTEGER,                 -- 200, 422, 500... (NULL si ok historique)
    message_erreur      TEXT,                    -- detail de l'erreur (NULL si ok)
    timestamp_appel     TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ───────────────────────────────────────────────────────────────
-- Index sur le timestamp des logs : accélère les analyses de drift
-- qui filtrent/trient par période (ex: "les appels de la semaine")
-- ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp_appel);
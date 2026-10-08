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
-- Écrite à chaque appel live. Lue par le script d'analyse de drift
-- Objectif : stocker inputs + outputs + temps (exigé par l'énoncé)
-- ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS logs (
    id                  BIGSERIAL  PRIMARY KEY,  -- ID auto-incrémenté (1 client peut être appelé N fois)
    sk_id_curr          BIGINT,                  -- ID client (NULL possible si nouveau client)
    inputs              JSONB      NOT NULL,      -- features reçues en entrée (pour le drift)
    proba               REAL       NOT NULL,      -- proba renvoyée (output)
    decision            SMALLINT   NOT NULL,      -- décision renvoyée (output)
    temps_inference_ms  REAL       NOT NULL,      -- temps d'inférence du modèle en ms (predict_proba seul)
    latence_totale_ms   REAL       NOT NULL,      -- latence totale de la requête API en ms (validation + inférence + réponse)
    timestamp_appel     TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP  -- horodatage de l'appel
);

-- ───────────────────────────────────────────────────────────────
-- Index sur le timestamp des logs : accélère les analyses de drift
-- qui filtrent/trient par période (ex: "les appels de la semaine")
-- ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp_appel);
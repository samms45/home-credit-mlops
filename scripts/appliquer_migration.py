# ═══════════════════════════════════════════════════════════════
# appliquer_migration.py — Met a jour la table logs (suivi des erreurs)
# A lancer UNE FOIS par base existante (locale, puis Render).
#
#   Local  :  uv run python scripts/appliquer_migration.py
#   Render :  $env:DATABASE_URL="<DATABASE_URL_EXTERNAL de ton .env>"
#             uv run python scripts/appliquer_migration.py
# ═══════════════════════════════════════════════════════════════

import os
import ssl
from urllib.parse import urlparse
import pg8000.native as pg8000
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://credit_user:credit_pass@localhost:5433/credit_scoring",  # defaut local
)

# Les changements a appliquer. Chaque instruction est idempotente
# (IF NOT EXISTS / DROP NOT NULL) : on peut la relancer sans risque.
MIGRATIONS = [
    "ALTER TABLE logs ALTER COLUMN proba DROP NOT NULL",
    "ALTER TABLE logs ALTER COLUMN decision DROP NOT NULL",
    "ALTER TABLE logs ALTER COLUMN temps_inference_ms DROP NOT NULL",
    "ALTER TABLE logs ADD COLUMN IF NOT EXISTS statut TEXT NOT NULL DEFAULT 'ok'",
    "ALTER TABLE logs ADD COLUMN IF NOT EXISTS code_http INTEGER",
    "ALTER TABLE logs ADD COLUMN IF NOT EXISTS message_erreur TEXT",
]

def params_connexion(url):
    u = urlparse(url)
    params = {
        "user": u.username, "password": u.password, "host": u.hostname,
        "port": u.port or 5432, "database": u.path.lstrip("/"),
    }
    hote = u.hostname or ""
    if hote.startswith("dpg-") or "render.com" in hote:
        params["ssl_context"] = ssl.create_default_context()
    return params

hote = urlparse(DATABASE_URL).hostname
print(f"Connexion a la base : {hote} ...")
conn = pg8000.Connection(**params_connexion(DATABASE_URL))
try:
    for sql in MIGRATIONS:
        conn.run(sql)
        print(f"  OK : {sql}")
finally:
    conn.close()
print("\nMigration terminee. La table logs est prete pour le suivi des erreurs.")
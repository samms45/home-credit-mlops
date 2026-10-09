# ═══════════════════════════════════════════════════════════════
# init_db_deployee.py — Crée les 2 tables dans la base DEPLOYEE (Render)
# Rejoue le fichier db/init.sql sur la base distante via l'External URL
# ═══════════════════════════════════════════════════════════════

import os                              # pour lire les variables d'environnement
from pathlib import Path               # pour lire le fichier init.sql
from urllib.parse import urlparse      # pour decouper l'URL en morceaux
import pg8000.native as pg8000         # driver PostgreSQL
from dotenv import load_dotenv         # pour charger le fichier .env

# ─── 1. CHARGER LE .env ────────────────────────────────────────
# load_dotenv lit le fichier .env et rend ses variables accessibles via os.environ
load_dotenv()

# Recuperer l'URL externe (celle pour se connecter depuis le PC)
DATABASE_URL = os.environ.get("DATABASE_URL_EXTERNAL")

# Securite : on s'arrete si l'URL n'est pas trouvee
if not DATABASE_URL:
    raise ValueError("DATABASE_URL_EXTERNAL introuvable dans le .env")

print("URL de connexion chargee depuis .env")

# ─── 2. DECOUPER L'URL POUR pg8000 ─────────────────────────────
# pg8000 ne prend pas une URL directement : on la decoupe en morceaux
url = urlparse(DATABASE_URL)           # postgresql://user:pass@host:port/dbname

# ─── 3. SE CONNECTER A LA BASE RENDER ──────────────────────────
print("Connexion a la base deployee (Render)...")
# ssl_context=True : Render EXIGE une connexion securisee (SSL) depuis l'exterieur
conn = pg8000.Connection(
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port or 5432,             # 5432 par defaut si non precise
    database=url.path.lstrip("/"),     # enleve le "/" devant le nom de la base
    ssl_context=True,                  # connexion chiffree (obligatoire sur Render)
)
print("Connexion reussie.")

# ─── 4. LIRE LE FICHIER init.sql ───────────────────────────────
print("Lecture de db/init.sql...")
sql = Path("db/init.sql").read_text(encoding="utf-8")

# ─── 5. EXECUTER LE SQL (creation des tables) ──────────────────
# conn.run execute les commandes SQL contenues dans le fichier
print("Creation des tables sur la base deployee...")
conn.run(sql)
print("Tables creees.")

# ─── 6. VERIFIER QUE LES 2 TABLES EXISTENT ─────────────────────
# On interroge le catalogue systeme de PostgreSQL pour lister les tables
resultat = conn.run(
    """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
)
tables = [ligne[0] for ligne in resultat]
print(f"\nTables presentes dans la base : {tables}")

conn.close()
print("Termine.")
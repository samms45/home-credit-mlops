# ============================================================
# Dockerfile - API de scoring credit
# Construit une image autonome contenant l'API et le modele
# ============================================================

# ------------------------------------------------------------
# 1. IMAGE DE BASE
# ------------------------------------------------------------
# On part d'une image Python 3.11 officielle et legere ("slim").
# 3.11 est OBLIGATOIRE : c'est la version qui a cree le model.pkl.
# "slim" = version allegee (moins d'outils inutiles = image plus petite)
FROM python:3.11-slim

# ------------------------------------------------------------
# 2. DOSSIER DE TRAVAIL dans la boite
# ------------------------------------------------------------
# Tout ce qui suit se passera dans /app a l'interieur du conteneur.
WORKDIR /app

# ------------------------------------------------------------
# 2bis. INSTALLER UNE BIBLIOTHEQUE SYSTEME requise par LightGBM
# ------------------------------------------------------------
# libgomp1 : bibliotheque de calcul parallele dont LightGBM a besoin.
# Elle n'est PAS incluse dans l'image "slim", il faut l'ajouter.
# apt-get update : rafraichit la liste des paquets systeme
# --no-install-recommends : evite d'installer des paquets superflus
# rm -rf ... : nettoie le cache apt pour garder l'image legere
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 3. INSTALLER LES DEPENDANCES
# ------------------------------------------------------------
# On copie D'ABORD le fichier des dependances, puis on installe.
# Pourquoi d'abord ? Docker garde en cache cette etape : tant que
# requirements.txt ne change pas, il ne reinstalle pas tout a chaque build.
COPY requirements-api.txt .

# pip install : installe les librairies listees
# --no-cache-dir : n'garde pas le cache pip (image plus petite)
# --trusted-host : fait confiance a ces serveurs malgre le certificat
#                  intercepte par le reseau de l'entreprise (proxy)
RUN pip install --no-cache-dir -r requirements-api.txt \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org

# ------------------------------------------------------------
# 4. COPIER LE CODE NECESSAIRE
# ------------------------------------------------------------
# On ne copie QUE ce dont l'API a besoin :
#   - le dossier app/   (le code de l'API)
#   - le dossier model/ (le modele + la signature + le seuil)
COPY app/ ./app/
COPY model/ ./model/

# ------------------------------------------------------------
# 5. EXPOSER LE PORT
# ------------------------------------------------------------
# Indique que l'API ecoutera sur le port 8000 (documentaire)
EXPOSE 8000

# ------------------------------------------------------------
# 6. COMMANDE DE DEMARRAGE
# ------------------------------------------------------------
# Lance uvicorn au demarrage du conteneur.
# --host 0.0.0.0 : OBLIGATOIRE dans Docker pour etre joignable de l'exterieur
#                  (127.0.0.1 ne serait accessible que DANS la boite)
# --port 8000 : le port d'ecoute
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Render fournit le port a utiliser via la variable d'environnement $PORT.
# Si $PORT n'existe pas (ex: en local), on retombe sur 8000 par defaut.
# La forme "shell" (sans crochets) permet d'utiliser la variable $PORT.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

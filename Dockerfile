# ---------------------------------------------------------------
# json2video — image unique avec Node.js, Python, FFmpeg.
# Le pipeline Node (server.js) appelle un script Python pour générer
# l'ASS, on a donc besoin des deux runtimes dans la même image.
# ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_VERSION=20.x \
    DEBIAN_FRONTEND=noninteractive

# Dépendances système : FFmpeg + libs OpenCV + fonts (DejaVu) + Node 20
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
      ffmpeg \
      fonts-dejavu \
      libgl1 libglib2.0-0 \
      tini \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Étape Python : déps en cache si requirements.txt ne bouge pas ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape Node : idem ---
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev || npm install --omit=dev

# --- Code applicatif ---
COPY . .

# Créer les dossiers runtime (montés en volume en compose)
RUN mkdir -p /app/uploads /app/outputs /app/credentials

# Utilisateur non-root
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 3000 5001

# tini comme PID 1 → propage SIGTERM correctement
ENTRYPOINT ["/usr/bin/tini", "--"]

# Par défaut on démarre le serveur Node (override en compose pour Flask)
CMD ["node", "server.js"]

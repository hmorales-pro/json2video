#!/usr/bin/env bash
# ----------------------------------------------------------------
# Purge les secrets de l'historique git du repo json2video.
#
# ⚠️  AVANT DE LANCER CE SCRIPT :
#   1) RÉVOQUER les clés exposées (elles le sont déjà publiquement) :
#      - OpenAI    : https://platform.openai.com/api-keys
#      - Google    : Cloud Console → IAM & Admin → Service Accounts → Keys
#      - OAuth     : Cloud Console → APIs & Services → Credentials
#   2) Faire passer le repo en PRIVÉ sur GitHub :
#      Settings → Danger Zone → Change visibility → Make private
#   3) S'assurer qu'on travaille sur une COPIE locale (clone séparé)
#      ou faire un backup : cp -R . ../json2video-backup
#
# Le script utilise git filter-repo (à installer si absent) :
#   pip install git-filter-repo
# ----------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Vérifications préalables
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "❌ git-filter-repo manquant. Installer avec : pip install git-filter-repo"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ Working tree non clean. Commit ou stash d'abord."
  exit 1
fi

read -p "As-tu révoqué les clés OpenAI / Google ? (yes/no) " ANSW
if [[ "$ANSW" != "yes" ]]; then
  echo "Abort. Révoque les clés d'abord."
  exit 1
fi

# 2. Purge des chemins sensibles de l'historique
git filter-repo --force \
  --invert-paths \
  --path .env \
  --path config.py \
  --path-glob 'client_secret*.json' \
  --path-glob 'json2video-*.json'

# 3. Ré-ajouter le remote (git filter-repo le supprime par sécurité)
echo
echo "✅ Historique nettoyé."
echo
echo "Étapes restantes :"
echo "  git remote add origin https://github.com/hmorales-pro/json2video.git"
echo "  git push --force --all"
echo "  git push --force --tags"
echo
echo "⚠️  Préviens tous les collaborateurs : ils doivent re-cloner."

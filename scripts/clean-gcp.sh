#!/usr/bin/env bash
# ---------------------------------------------------------------
# clean-gcp.sh — supprime toutes les traces locales de Google Cloud
# du projet json2video.
#
# Ce que ce script fait :
#   1) Détecte les fichiers de credentials JSON
#   2) Détecte les variables GCP dans .env (GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET)
#   3) Détecte le pipeline Flask legacy (app.py + config.py)
#   4) Affiche un résumé et demande confirmation
#   5) Supprime/nettoie ce qui est confirmé
#
# Ce que ce script NE FAIT PAS :
#   - Toucher à la console Google Cloud (à faire manuellement avant)
#   - Toucher à l'historique git (utiliser scripts/purge-secrets-from-git.sh)
#
# Usage :
#   bash scripts/clean-gcp.sh          # mode interactif
#   bash scripts/clean-gcp.sh --yes    # auto-confirme tout (CI / scripté)
# ---------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

YES=0
if [[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]]; then
  YES=1
fi

# ----- Helpers -----
red()    { printf "\033[31m%s\033[0m\n" "$1"; }
green()  { printf "\033[32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }
bold()   { printf "\033[1m%s\033[0m\n" "$1"; }

confirm() {
  local msg="$1"
  if [[ $YES -eq 1 ]]; then
    echo "  [auto-yes] $msg"
    return 0
  fi
  read -r -p "  → $msg [y/N] " ans
  [[ "$ans" =~ ^[yY]$ ]]
}

# sed in-place compatible BSD (macOS) et GNU (Linux)
sed_inplace() {
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

# ----- Détection -----
bold "==================== clean-gcp.sh ===================="
echo

# 1. Fichiers credentials
# (Bash 3.2 sur macOS n'a pas `mapfile` → on utilise une boucle portable.)
CRED_FILES=()
for pattern in 'json2video-*.json' 'client_secret*.json' '*service-account*.json' 'credentials/*.json'; do
  for f in $pattern; do
    [[ -f "$f" ]] && CRED_FILES+=("$f")
  done
done

# 2. Variables GCP dans .env
HAS_GAC=0
HAS_GCS=0
if [[ -f .env ]]; then
  grep -qE "^GOOGLE_APPLICATION_CREDENTIALS=" .env && HAS_GAC=1 || true
  grep -qE "^GCS_BUCKET=" .env && HAS_GCS=1 || true
fi

# 3. Flask legacy
HAS_APPPY=0
HAS_CONFIGPY=0
[[ -f app.py ]] && HAS_APPPY=1
[[ -f config.py ]] && HAS_CONFIGPY=1

# ----- Résumé -----
echo "Détecté dans $(pwd) :"
echo

if [[ ${#CRED_FILES[@]} -gt 0 ]]; then
  yellow "  Fichiers credentials JSON (${#CRED_FILES[@]}) :"
  for f in "${CRED_FILES[@]}"; do echo "    - $f"; done
else
  green "  ✓ Aucun fichier credentials JSON sur disque"
fi
echo

if [[ $HAS_GAC -eq 1 || $HAS_GCS -eq 1 ]]; then
  yellow "  Variables GCP dans .env :"
  [[ $HAS_GAC -eq 1 ]] && echo "    - GOOGLE_APPLICATION_CREDENTIALS"
  [[ $HAS_GCS -eq 1 ]] && echo "    - GCS_BUCKET"
else
  green "  ✓ Aucune variable GCP dans .env"
fi
echo

if [[ $HAS_APPPY -eq 1 || $HAS_CONFIGPY -eq 1 ]]; then
  yellow "  Pipeline Flask legacy :"
  [[ $HAS_APPPY -eq 1 ]] && echo "    - app.py (utilise google-cloud-speech)"
  [[ $HAS_CONFIGPY -eq 1 ]] && echo "    - config.py (chargé par app.py)"
else
  green "  ✓ Pipeline Flask legacy déjà supprimé"
fi
echo

# Rien à faire ?
if [[ ${#CRED_FILES[@]} -eq 0 && $HAS_GAC -eq 0 && $HAS_GCS -eq 0 && $HAS_APPPY -eq 0 && $HAS_CONFIGPY -eq 0 ]]; then
  green "Rien à nettoyer. Le projet est déjà GCP-free."
  exit 0
fi

bold "Actions proposées (chacune avec confirmation)"
echo

# ----- 1) Suppression fichiers credentials -----
if [[ ${#CRED_FILES[@]} -gt 0 ]]; then
  if confirm "Supprimer les ${#CRED_FILES[@]} fichier(s) credentials JSON ?"; then
    for f in "${CRED_FILES[@]}"; do
      rm -f "$f" && echo "    rm $f"
    done
    green "  ✓ Fichiers credentials supprimés"
  else
    yellow "  ⚠ Fichiers credentials conservés"
  fi
  echo
fi

# ----- 2) Nettoyer .env -----
if [[ $HAS_GAC -eq 1 || $HAS_GCS -eq 1 ]]; then
  if confirm "Retirer les variables GCP du .env ?"; then
    [[ $HAS_GAC -eq 1 ]] && sed_inplace '/^GOOGLE_APPLICATION_CREDENTIALS=/d' .env
    [[ $HAS_GCS -eq 1 ]] && sed_inplace '/^GCS_BUCKET=/d' .env
    green "  ✓ Variables GCP retirées du .env"
  else
    yellow "  ⚠ Variables GCP conservées dans .env"
  fi
  echo
fi

# ----- 3) Supprimer Flask legacy -----
if [[ $HAS_APPPY -eq 1 || $HAS_CONFIGPY -eq 1 ]]; then
  echo "  Le pipeline Flask legacy (app.py) n'est plus fonctionnel sans GCP."
  echo "  Le pipeline Node (server.js + Whisper) reste indépendant."
  if confirm "Supprimer app.py et config.py ?"; then
    [[ $HAS_APPPY -eq 1 ]] && rm -f app.py && echo "    rm app.py"
    [[ $HAS_CONFIGPY -eq 1 ]] && rm -f config.py && echo "    rm config.py"
    # Aussi vider le __pycache__ associé
    [[ -d __pycache__ ]] && rm -rf __pycache__ && echo "    rm -rf __pycache__"
    green "  ✓ Flask legacy supprimé"

    # Si on supprime Flask, retirer aussi les deps Python qui n'étaient que pour Flask
    if [[ -f requirements.txt ]]; then
      if confirm "Simplifier requirements.txt (ne garder que pysubs2) ?"; then
        cat > requirements.txt <<'REQ'
# json2video — dépendances Python pour le pipeline Node
# pysubs2 est utilisé par words_to_ass.py et convert_srt_to_ass.py.
pysubs2>=1.7.0
REQ
        green "  ✓ requirements.txt simplifié"
      fi
    fi

    # Retirer aussi les variables Flask du .env si présentes
    if [[ -f .env ]]; then
      for var in FLASK_PORT FLASK_HOST FLASK_DEBUG MONITOR_USERNAME MONITOR_PASSWORD; do
        if grep -qE "^${var}=" .env; then
          sed_inplace "/^${var}=/d" .env && echo "    retiré $var du .env"
        fi
      done
    fi
  else
    yellow "  ⚠ Flask legacy conservé"
  fi
  echo
fi

# ----- Résumé final -----
bold "==================== Résumé ===================="
echo
echo "État actuel du projet :"
[[ -f .env ]] && echo "  .env :" && sed 's/^/    /' .env || echo "  (pas de .env)"
echo
echo "Fichiers JSON restants à la racine :"
ls -1 *.json 2>/dev/null | sed 's/^/  /' || echo "  (aucun)"
echo
green "Nettoyage GCP terminé. ✓"
echo
echo "Pour aller plus loin :"
echo "  - Vérifier que rien ne casse : npm start (le preflight doit passer)"
echo "  - Si pas encore fait : révoque les credentials côté Google Cloud Console"
echo "  - Pour purger l'historique git : bash scripts/purge-secrets-from-git.sh"

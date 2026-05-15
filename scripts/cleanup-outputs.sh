#!/usr/bin/env bash
# ---------------------------------------------------------------
# Nettoie outputs/ et uploads/ des fichiers anciens.
#
# Usage :
#   bash scripts/cleanup-outputs.sh           # supprime > 24 h
#   AGE_HOURS=72 bash scripts/cleanup-outputs.sh
#
# À planifier en cron pour la prod :
#   0 4 * * * cd /chemin/json2video && bash scripts/cleanup-outputs.sh
# ---------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."
AGE_HOURS="${AGE_HOURS:-24}"
MINUTES=$((AGE_HOURS * 60))

echo "Suppression des fichiers > ${AGE_HOURS}h dans outputs/ et uploads/..."
before_out=$(du -sh outputs/ 2>/dev/null | cut -f1 || echo 0)
before_up=$(du -sh uploads/ 2>/dev/null | cut -f1 || echo 0)

find outputs/ -type f -name "*.mp4" -mmin +"${MINUTES}" -print -delete 2>/dev/null || true
find uploads/ -type f \( -name "*.mp4" -o -name "*.jpg" -o -name "*.png" -o -name "*.mp3" -o -name "*.wav" -o -name "*.m4a" -o -name "*.ass" -o -name "*.srt" -o -name "*.json" \) -mmin +"${MINUTES}" -print -delete 2>/dev/null || true

after_out=$(du -sh outputs/ 2>/dev/null | cut -f1 || echo 0)
after_up=$(du -sh uploads/ 2>/dev/null | cut -f1 || echo 0)
echo "outputs/ : ${before_out} → ${after_out}"
echo "uploads/ : ${before_up} → ${after_up}"

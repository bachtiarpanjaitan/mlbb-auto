#!/bin/bash
# MLBB Minimap Dataset Inspector
# Memeriksa data gambar dan label YOLO di trainings/hero_detector/
set -e

cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MLBB Minimap Dataset Inspector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Jalankan script inspector dengan meneruskan argumen (e.g. --val, --all)
python3 tools/inspect_dataset.py "$@"

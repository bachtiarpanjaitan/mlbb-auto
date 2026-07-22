#!/bin/bash
# Label Minimap — MLBB Hero & Jungle Detection Dataset
# B = blue_hero, R = red_hero, J = jungle
# N = next frame, S = save & next, Q = quit

set -e
cd "$(dirname "$0")/.."
python3 tools/label_minimap.py "$@"

# Auto-cleanup file .txt kosong jika ada
find trainings/hero_detector/labels -type f -name "*.txt" -size 0 -delete 2>/dev/null || true


#!/bin/bash
# Label Minimap Heroes — MLBB Hero Detection Dataset
# Klik kiri = blue_hero, klik kanan = red_hero
# N = next frame, S = save & next, Q = quit

set -e
cd "$(dirname "$0")/.."
python3 tools/label_minimap.py "$@"

#!/bin/bash
# Minimap Simulation from Parquet Data
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 tools/simulate_minimap.py "$@"

#!/bin/bash
# Region Editor — MLBB Minimap
# Saves region definitions as JSON, used by the minimap processor.

set -e
cd "$(dirname "$0")/.."
python3 tools/region_editor.py
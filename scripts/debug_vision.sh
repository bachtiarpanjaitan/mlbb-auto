#!/bin/bash
cd "$(dirname "$0")/.."

# Bersihkan .tmp dari crop sebelumnya
if [ -d ".tmp" ]; then
    rm -f .tmp/*
    echo "🧹 Cleaned .tmp/"
fi

source .venv/bin/activate
python3 tools/debug_vision.py "$@"

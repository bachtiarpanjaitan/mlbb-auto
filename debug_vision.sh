#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python3 tools/debug_vision.py "$@"

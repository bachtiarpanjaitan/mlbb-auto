#!/bin/bash
# YOLO Minimap Detection Inference
set -e
cd "$(dirname "$0")/.."
python3 tools/inference.py "$@"

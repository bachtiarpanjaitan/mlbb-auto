#!/bin/bash
# Train CD Number Digit Classifier — Baca angka cooldown dari skill icon
set -e

cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CD Number Digit Classifier Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Hero dataset: trainings/hero_skills/dataset/"
N=$(find trainings/hero_skills/dataset/ -path '*/cooldown/*.png' 2>/dev/null | wc -l)
echo "    Cooldown crops: $N"
N2=$(ls trainings/cd_number/real_dataset/*.png 2>/dev/null | wc -l)
echo "    CD number real: $N2"
echo ""

# Run training
python3 tools/train_cd_number.py "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: models/cd_number_classifier.onnx"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

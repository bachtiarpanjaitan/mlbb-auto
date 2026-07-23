#!/bin/bash
# Train Skill State CNN Classifier — MLBB Skill Icon Cooldown Detection
# Jalankan setelah mengumpulkan dataset via tools/crop_skills_dataset.py
set -e

cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Skill State CNN Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Dataset: trainings/hero_skills/dataset/"
echo "  Files : $(find trainings/hero_skills/dataset/ -name '*.png' 2>/dev/null | wc -l) images"
echo "  Videos:"
for v in trainings/hero_skills/dataset/*/; do
    if [ -d "$v" ]; then
        n=$(find "$v" -name '*.png' 2>/dev/null | wc -l)
        echo "           $(basename "$v"): $n samples"
    fi
done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run training
python3 tools/train_skill_classifier.py "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: models/skill_classifier.onnx"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

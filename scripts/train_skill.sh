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

# Auto-resume dari checkpoint terakhir (kalau ada)
CKPT="trainings/hero_skills/checkpoints/best.pt"
if [ -f "$CKPT" ]; then
    # Cek apakah user sudah provide --resume
    case "$*" in
        *--resume*) HAS_RESUME=1 ;;
        *) HAS_RESUME=0 ;;
    esac
    if [ "$HAS_RESUME" = "0" ]; then
        echo "  🔄 Auto-resume: $CKPT"
        EXTRA="--resume $CKPT"
    else
        EXTRA=""
    fi
else
    echo "  🆕 Training from scratch"
    EXTRA=""
fi

# Run training
python3 tools/train_skill_classifier.py $EXTRA "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: models/skill_classifier.onnx"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

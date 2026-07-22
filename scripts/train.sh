#!/bin/bash
# Train YOLOv11n — MLBB Minimap Hero Detection
# Jalankan setelah selesai labeling data di trainings/hero_detector/
set -e

cd "$(dirname "$0")/.."

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  YOLOv11n Training — MLBB Minimap"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Dataset: trainings/hero_detector/"
echo "  Images: $(ls trainings/hero_detector/images/train/*.png 2>/dev/null | wc -l) train"
echo "  Labels: $(find trainings/hero_detector/labels/train -name '*.txt' -not -empty | wc -l) non-empty"
echo "  Classes:"
cat trainings/hero_detector/labels/train/*.txt 2>/dev/null | grep -v '^$' | awk '{print $1}' | sort | uniq -c | sort -rn | while read count cls; do
    name="blue_hero"
    [ "$cls" = "1" ] && name="red_hero"
    echo "           $count $name"
done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Copy ke validation set (20% data)
mkdir -p trainings/hero_detector/images/val trainings/hero_detector/labels/val
rm -f trainings/hero_detector/images/val/* trainings/hero_detector/labels/val/*

ls trainings/hero_detector/labels/train/*.txt 2>/dev/null | shuf | head -$(( $(ls trainings/hero_detector/labels/train/*.txt 2>/dev/null | wc -l) / 5 )) | while read f; do
    base=$(basename "$f" .txt)
    cp "trainings/hero_detector/labels/train/${base}.txt" "trainings/hero_detector/labels/val/" 2>/dev/null || true
    cp "trainings/hero_detector/images/train/${base}.png" "trainings/hero_detector/images/val/" 2>/dev/null || true
done

echo "Val set: $(ls trainings/hero_detector/images/val/*.png 2>/dev/null | wc -l) images"
echo ""

# Train
python3 tools/train_yolo.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: trainings/hero_detector/yolo11n_minimap/weights/best.pt"
echo "  ONNX:  trainings/hero_detector/yolo11n_minimap/weights/best.onnx"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

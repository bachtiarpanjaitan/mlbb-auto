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
python3 -c "
import yaml
from pathlib import Path
from collections import Counter

data_cfg = yaml.safe_load(Path('trainings/hero_detector/data.yaml').read_text())
names = data_cfg.get('names', [])

counts = Counter()
for f in Path('trainings/hero_detector/labels/train').glob('*.txt'):
    for line in f.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1

for cls_id, count in counts.most_common():
    cname = names[cls_id] if cls_id < len(names) else f'cls_{cls_id}'
    print(f'           {count:5d} {cname}')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Split validation set (20% data, terpisah murni tanpa kebohongan leakage)
mkdir -p trainings/hero_detector/images/val trainings/hero_detector/labels/val

# Reset: kembalikan file val lama ke train terlebih dahulu jika ada
mv trainings/hero_detector/labels/val/*.txt trainings/hero_detector/labels/train/ 2>/dev/null || true
mv trainings/hero_detector/images/val/*.png trainings/hero_detector/images/train/ 2>/dev/null || true

# Pindahkan 20% data secara acak dari train ke val
ls trainings/hero_detector/labels/train/*.txt 2>/dev/null | shuf | head -$(( $(ls trainings/hero_detector/labels/train/*.txt 2>/dev/null | wc -l) / 5 )) | while read f; do
    base=$(basename "$f" .txt)
    mv "trainings/hero_detector/labels/train/${base}.txt" "trainings/hero_detector/labels/val/" 2>/dev/null || true
    mv "trainings/hero_detector/images/train/${base}.png" "trainings/hero_detector/images/val/" 2>/dev/null || true
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

# Copy model terbaru ke folder models untuk dipakai langsung
ONNX_SRC="trainings/hero_detector/yolo11n_minimap/weights/best.onnx"
ONNX_DST="models/hero_tracker.onnx"

if [ -f "$ONNX_SRC" ]; then
    mkdir -p models
    cp "$ONNX_SRC" "$ONNX_DST"
    echo ""
    echo "✅ Model copied → $ONNX_DST"
else
    echo ""
    echo "⚠️  ONNX model not found: $ONNX_SRC"
    echo "   Jalankan export manual jika diperlukan."
fi

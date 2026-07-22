"""
Train YOLOv11n on MLBB minimap hero detection dataset.
- Jika sudah pernah training: lanjut dari model terakhir (resume)
- Jika pertama kali: download pretrained yolo11n.pt
Usage: python tools/train_yolo.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_best_model() -> str | None:
    """Cari model terbaik dari training sebelumnya."""
    candidates = [
        "trainings/hero_detector/yolo11n_minimap/weights/best.pt",
        "runs/detect/trainings/hero_detector/yolo11n_minimap/weights/best.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def main():
    device = get_device()
    prev_model = get_best_model()

    if prev_model:
        print(f"📦 Load model sebelumnya: {prev_model}")
        print(f"   → Melanjutkan training (fine-tune)")
        model = YOLO(prev_model)
    else:
        print(f"📦 First time training — download yolo11n.pt dari pretrained COCO")
        model = YOLO("yolo11n.pt")

    print(f"⚙️  Device: {device}")
    print(f"📊 Dataset: trainings/hero_detector/data.yaml")
    print(f"   Images: {len(list(Path('trainings/hero_detector/images/train').glob('*.png')))} train, "
          f"{len(list(Path('trainings/hero_detector/images/val').glob('*.png')))} val")

    results = model.train(
        data="trainings/hero_detector/data.yaml",
        epochs=150,
        imgsz=320,
        batch=16,
        lr0=0.005,           # lower LR for fine-tuning
        augment=True,
        patience=30,
        device=device,
        project="trainings/hero_detector",
        name="yolo11n_minimap",
        exist_ok=True,
        workers=4,
        close_mosaic=10,
        resume=False,         # auto-resume not needed (we load weights manually)
    )

    # Export
    model.export(format="onnx", imgsz=320)
    model_path = "trainings/hero_detector/yolo11n_minimap/weights/best.pt"
    print(f"\n✅ Model: {model_path}")
    print(f"   ONNX:  trainings/hero_detector/yolo11n_minimap/weights/best.onnx")


if __name__ == "__main__":
    main()

"""
Train YOLOv11n on MLBB minimap hero & jungle detection dataset (11 classes).
Usage: python tools/train_yolo.py
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress MPS warnings SEBELUM import ultralytics
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


def setup_env():
    """Suppress warnings, configure environment."""
    import warnings
    warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
    warnings.filterwarnings("ignore", message=".*index_put_with_accumulate_mps.*")

    import os
    os.environ["PYTHONWARNINGS"] = "ignore"


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    setup_env()
    device = get_device()

    print(f"📦 Loading base model: yolo11n.pt (11 classes dataset)")
    model = YOLO("yolo11n.pt")

    print(f"⚙️  Device: {device}")
    print(f"📊 Dataset: trainings/hero_detector/data.yaml")
    print(f"   Images: {len(list(Path('trainings/hero_detector/images/train').glob('*.png')))} train, "
          f"{len(list(Path('trainings/hero_detector/images/val').glob('*.png')))} val")

    project_dir = Path("trainings/hero_detector").resolve()

    results = model.train(
        data="trainings/hero_detector/data.yaml",
        epochs=100,
        imgsz=352,             # Closer to original minimap crop size (350x340)
        batch=16,
        lr0=0.005,
        lrf=0.01,
        hsv_h=0.0,             # CRITICAL: Disable hue shift so blue vs red hero colors are not distorted
        hsv_s=0.1,             # Mild saturation variation
        hsv_v=0.1,             # Mild brightness variation
        mosaic=0.0,            # Disable mosaic to prevent tiny hero icons from being cropped/destroyed
        patience=25,
        device=device,
        project=str(project_dir),
        name="yolo11n_minimap",
        exist_ok=True,
        workers=4,
    )

    # Export ONNX
    model.export(format="onnx", imgsz=352)

    # Ensure target weights directory exists
    target_dir = Path("trainings/hero_detector/yolo11n_minimap/weights")
    target_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(results.save_dir) / "weights"
    if save_dir.exists() and save_dir.resolve() != target_dir.resolve():
        import shutil
        for weight_file in save_dir.glob("*"):
            if weight_file.is_file():
                shutil.copy(weight_file, target_dir / weight_file.name)

    model_path = target_dir / "best.pt"
    onnx_path = target_dir / "best.onnx"
    print(f"\n✅ Model: {model_path}")
    print(f"   ONNX:  {onnx_path}")


if __name__ == "__main__":
    main()

"""
YOLOv11n Detector — MLBB Minimap Hero Detection
Mendeteksi 2 class: blue_hero, red_hero
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("mlbb.vision.yolo_detector")


class YOLOMinimapDetector:
    """
    YOLOv11n wrapper untuk deteksi hero di minimap.

    Args:
        model_path: Path ke YOLO weights (.pt atau .onnx)
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        input_size: Input size untuk model
    """

    def __init__(
        self,
        model_path: str | Path = "trainings/hero_detector/yolo11n_minimap/weights/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        input_size: int = 320,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self._model = None
        self._last_result: list = []
        self._lock = False
        self._load_model()

    @property
    def last_result(self) -> list:
        return list(self._last_result)

    def detect_async(self, minimap_img: np.ndarray):
        import threading
        def _run():
            try:
                self._lock = True
                if minimap_img is not None and minimap_img.size > 0:
                    self._last_result = self.detect(minimap_img)
            finally:
                self._lock = False
        if not self._lock:
            t = threading.Thread(target=_run, daemon=True)
            t.start()

    def _get_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            # Move to best device
            device = self._get_device()
            if device != "cpu":
                self._model.to(device)
            logger.info("YOLO model loaded from %s (device: %s)", self.model_path, device)
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def detect(
        self, minimap_img: np.ndarray,
    ) -> list[tuple[str, int, int, int, float]]:
        """
        Detect hero dots in minimap.

        Args:
            minimap_img: Cropped minimap image (BGR, 350x340 typical)

        Returns:
            List of (team, cx, cy, radius, confidence)
            team: "blue" or "red"
        """
        if minimap_img is None or minimap_img.size == 0:
            return []

        # Run inference
        results = self._model(
            minimap_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return []

        detections = []
        h, w = minimap_img.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Center and radius
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = max((x2 - x1), (y2 - y1)) // 2

            # Clamp to image bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            team = "blue" if cls_id == 0 else "red"
            detections.append((team, cx, cy, r, conf))

        return detections


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vision.core.cropper import crop_region

    # Test with frame from video
    cap = cv2.VideoCapture("videos/alpha_1.mp4")
    for _ in range(200):
        cap.read()
    _, frame = cap.read()
    cap.release()

    mm = crop_region(frame, "map")
    if mm is not None:
        detector = YOLOMinimapDetector()
        dets = detector.detect(mm)
        print(f"Detected {len(dets)} heroes")
        for team, cx, cy, r, conf in dets:
            print(f"  {team}: ({cx},{cy}) r={r} conf={conf:.2f}")

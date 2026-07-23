"""
YOLOv11n Detector — MLBB Minimap Detection
Mendeteksi 11 class:
  0  = blue_hero
  1  = red_hero
  2  = lord            (1 lokasi)
  3  = turtle          (1 lokasi)
  4  = thunder_fenrir  (2 lokasi: blue buff di kedua sisi)
  5  = molten_fiend    (2 lokasi: red buff di kedua sisi)
  6  = lithowanderer   (1 lokasi)
  7  = crab            (2 lokasi)
  8  = lava_golem      (2 lokasi)
  9  = fire_beetle     (2 lokasi)
  10 = horned_lizard   (2 lokasi)

Backend:
  - .onnx → onnxruntime (ringan, TANPA PyTorch, support CoreML di macOS)
  - .pt   → ultralytics (fallback, butuh PyTorch)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("mlbb.vision.yolo_detector")


# ── Class ID mapping ────────────────────────────────────────────────────────
YOLO_CLASS_MAP: dict[int, tuple[str, str | None]] = {
    0:  ("blue",   None),
    1:  ("red",    None),
    2:  ("jungle", "lord"),
    3:  ("jungle", "turtle"),
    4:  ("jungle", "thunder_fenrir"),
    5:  ("jungle", "molten_fiend"),
    6:  ("jungle", "lithowanderer"),
    7:  ("jungle", "crab"),
    8:  ("jungle", "lava_golem"),
    9:  ("jungle", "fire_beetle"),
    10: ("jungle", "horned_lizard"),
}

NUM_CLASSES = len(YOLO_CLASS_MAP)


class YOLOMinimapDetector:
    """
    YOLOv11n wrapper untuk deteksi hero & jungle camp di minimap.

    Secara otomatis memilih backend:
      - .onnx → onnxruntime (ringan, TANPA PyTorch, support CoreML)
      - .pt   → ultralytics (fallback, membutuhkan PyTorch)

    Args:
        model_path    : Path ke YOLO weights (.onnx atau .pt)
        conf_threshold: Confidence threshold
        iou_threshold : NMS IoU threshold
    """

    def __init__(
        self,
        model_path: str | Path = "models/hero_tracker.onnx",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size: int = 352  # akan dioverride dari model metadata
        self._session = None   # onnxruntime session (ONNX backend)
        self._input_name: str = "images"
        self._model = None     # ultralytics YOLO (.pt backend)
        self._use_onnx = self.model_path.suffix.lower() == ".onnx"
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

    # ── Model Loading ───────────────────────────────────────────────────

    def _load_model(self):
        """Load model: onnxruntime untuk .onnx, ultralytics untuk .pt."""
        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_pt()

    def _load_onnx(self):
        """Load ONNX model via onnxruntime — TANPA PyTorch."""
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Pilih provider: CoreML (Apple Silicon) > CPU
        providers = []
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Buat session dengan thread count terbatas
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2

        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=providers,
        )

        # Baca input shape dari model metadata
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self.input_size = inp.shape[-1]  # e.g. 352

        logger.info(
            "ONNX model loaded via onnxruntime: %s (input=%dx%d, providers=%s)",
            self.model_path, self.input_size, self.input_size,
            [p for p in providers if p in available],
        )

    def _load_pt(self):
        """Load PyTorch .pt model via ultralytics (fallback)."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            device = self._get_device()
            if device != "cpu":
                self._model.to(device)
            logger.info("PT model loaded via ultralytics: %s (device: %s)",
                        self.model_path, device)
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def _get_device(self) -> str:
        """Auto-detect best available device untuk .pt backend."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ── ONNX Inference ──────────────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Letterbox resize + normalize → NCHW float32 blob.
        Returns: (blob, scale, pad_x, pad_y)
        """
        h, w = img.shape[:2]
        size = self.input_size

        # Letterbox resize (maintain aspect ratio)
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Pad ke input_size x input_size
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR→RGB, normalize [0,1], HWC→NCHW
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis].copy()  # (1, 3, H, W)

        return blob, scale, pad_x, pad_y

    def _postprocess_onnx(
        self, output: np.ndarray,
        orig_w: int, orig_h: int,
        scale: float, pad_x: int, pad_y: int,
    ) -> list[tuple[str, str | None, int, int, int, float]]:
        """
        Parse YOLO11 ONNX output.
        Output shape: [1, 4+num_classes, num_anchors]
        Transpose → [num_anchors, 4+num_classes]
        Per row: [cx, cy, w, h, cls0_score, ..., clsN_score]
        """
        pred = output[0].T  # (num_anchors, 4+num_classes)

        boxes_raw = []
        scores_raw = []
        class_ids = []

        for row in pred:
            class_scores = row[4:4 + NUM_CLASSES]
            best_cls = int(np.argmax(class_scores))
            best_conf = float(class_scores[best_cls])

            if best_conf < self.conf_threshold:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]

            # De-letterbox → koordinat original
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale

            # Clamp
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            boxes_raw.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores_raw.append(best_conf)
            class_ids.append(best_cls)

        if not boxes_raw:
            return []

        # NMS via OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes_raw, scores_raw,
            self.conf_threshold, self.iou_threshold,
        )
        if len(indices) == 0:
            return []

        detections = []
        for i in indices.flatten():
            x, y, bw, bh = boxes_raw[i]
            cx = x + bw // 2
            cy = y + bh // 2
            r = max(bw, bh) // 2
            cx = max(0, min(orig_w - 1, cx))
            cy = max(0, min(orig_h - 1, cy))
            team, jungle_name = YOLO_CLASS_MAP.get(class_ids[i], ("unknown", None))
            detections.append((team, jungle_name, cx, cy, r, scores_raw[i]))

        return detections

    # ── Public API ──────────────────────────────────────────────────────

    def detect(
        self, minimap_img: np.ndarray,
    ) -> list[tuple[str, str | None, int, int, int, float]]:
        """
        Detect hero & jungle camp dots in minimap.

        Returns:
            List of (team, jungle_name, cx, cy, radius, confidence)
        """
        if minimap_img is None or minimap_img.size == 0:
            return []

        if self._use_onnx:
            return self._detect_onnx(minimap_img)
        else:
            return self._detect_pt(minimap_img)

    def _detect_onnx(self, minimap_img: np.ndarray) -> list:
        """Inference via onnxruntime (pure Python, NO PyTorch)."""
        h, w = minimap_img.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(minimap_img)
        outputs = self._session.run(None, {self._input_name: blob})[0]
        return self._postprocess_onnx(outputs, w, h, scale, pad_x, pad_y)

    def _detect_pt(self, minimap_img: np.ndarray) -> list:
        """Inference via ultralytics (.pt backend)."""
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
            cx = max(0, min(w - 1, (x1 + x2) // 2))
            cy = max(0, min(h - 1, (y1 + y2) // 2))
            r = max((x2 - x1), (y2 - y1)) // 2
            team, jungle_name = YOLO_CLASS_MAP.get(cls_id, ("unknown", None))
            detections.append((team, jungle_name, cx, cy, r, conf))

        return detections


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from vision.core.cropper import crop_region

    cap = cv2.VideoCapture("videos/alpha_1.mp4")
    for _ in range(200):
        cap.read()
    _, frame = cap.read()
    cap.release()

    mm = crop_region(frame, "map")
    if mm is not None:
        detector = YOLOMinimapDetector(
            "models/hero_tracker.onnx"
        )
        dets = detector.detect(mm)
        print(f"Detected {len(dets)} objects (ONNX via onnxruntime)")
        for team, jungle_name, cx, cy, r, conf in dets:
            label = f"{team}/{jungle_name}" if jungle_name else team
            print(f"  {label}: ({cx},{cy}) r={r} conf={conf:.2f}")

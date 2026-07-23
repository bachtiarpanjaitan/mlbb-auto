"""
Skills Detector — Mendeteksi status skill (ready/cooldown/available/empty).

Menggunakan CNN classifier via ONNX Runtime:
  Input:  70×70×3 BGR crop icon skill
  Output: ready | cooldown | available | empty

Model: models/skill_classifier.onnx (31 KB, ~63K params)
  Train: tools/train_skill_classifier.py
  Data:  tools/crop_skills_dataset.py
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import cv2
import numpy as np

from ..base import BaseDetector, Detection

log = logging.getLogger(__name__)

# ONNX model path (relative to project root)
MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "skill_classifier.onnx"
)

CLASSES = ["ready", "cooldown", "available", "empty", "locked"]
INPUT_SIZE = 70


class SkillsDetector(BaseDetector):
    """
    Deteksi status skill dengan CNN classifier via ONNX.

    Fallback ke CV heuristics jika model tidak ditemukan.
    """

    def __init__(self, ocr=None, use_model: bool = True):
        super().__init__(ocr)
        self.load_config("skills")
        self._session = None
        self._input_name = None
        self._output_name = None
        self._use_model = use_model
        self._load_model()

    def set_use_model(self, use_model: bool):
        """Toggle between CNN model (True) and CV fallback (False)."""
        self._use_model = use_model

    @property
    def using_model(self) -> bool:
        """Whether CNN model is being used for detection."""
        return self._use_model and self._session is not None

    def _load_model(self):
        """Load ONNX model if available, otherwise use fallback."""
        if not os.path.isfile(MODEL_PATH):
            log.warning("ONNX model not found: %s — using CV fallback", MODEL_PATH)
            return

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(MODEL_PATH)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            log.info("Loaded ONNX skill classifier: %s (%d classes)",
                     MODEL_PATH, len(CLASSES))
        except Exception as e:
            log.warning("Failed to load ONNX model: %s — using CV fallback", e)
            self._session = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess 70×70 BGR crop → ONNX input tensor (1, 3, 70, 70)."""
        if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

        # BGR → RGB → CHW → normalize to [-1, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return np.expand_dims(tensor, axis=0)  # (1, 3, 70, 70)

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        if self._use_model and self._session is not None:
            return self._detect_onnx(image)
        else:
            return self._detect_cv_fallback(image)

    def _detect_onnx(self, image: np.ndarray) -> Detection | None:
        """Detect skill state using ONNX CNN model."""
        try:
            input_tensor = self._preprocess(image)
            outputs = self._session.run(
                [self._output_name], {self._input_name: input_tensor}
            )
            logits = outputs[0][0]  # (4,)

            # Softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)

            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            label = CLASSES[class_id]

            return Detection(
                value={
                    "ready": label in ("ready", "available"),
                    "cooldown": label == "cooldown",
                    "available": label == "available",
                    "empty": label == "empty",
                    "locked": label == "locked",
                },
                confidence=confidence,
                label=label,
                meta={
                    "model": "onnx_cnn",
                    "probs": {c: round(float(p), 3) for c, p in zip(CLASSES, probs)},
                },
            )
        except Exception as e:
            log.error("ONNX inference error: %s — falling back to CV", e)
            return self._detect_cv_fallback(image)

    def _detect_cv_fallback(self, image: np.ndarray) -> Detection | None:
        """
        Fallback: CV heuristics (HSV + Canny) — sama seperti implementasi lama.

        Digunakan ketika ONNX model tidak tersedia atau gagal.
        """
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        dark_ratio = float((gray < 60).mean())
        variance = float(gray.var())

        # HSV overlay detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark_overlay = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        overlay_ratio = cv2.countNonZero(dark_overlay) / gray.size if gray.size > 0 else 0

        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 255, 255]))
        bright_pct = cv2.countNonZero(bright_mask) / gray.size if gray.size > 0 else 0

        # Edge analysis
        edges = cv2.Canny(gray, 30, 100)
        edge_ratio = cv2.countNonZero(edges) / gray.size if gray.size > 0 else 0
        has_edges = edge_ratio > 0.015

        # Decision
        is_cooldown = overlay_ratio > 0.35 or (dark_ratio > 0.35 and not has_edges)
        is_available = bright_pct > 0.12 and not is_cooldown
        is_ready = not is_cooldown and (has_edges or mean_brightness > 80)

        if is_cooldown and overlay_ratio > 0.5:
            confidence = 0.90
        elif is_cooldown:
            confidence = 0.80
        elif is_available:
            confidence = 0.85
        elif is_ready:
            confidence = 0.80
        else:
            confidence = 0.60

        if is_cooldown:
            label = "cooldown"
        elif is_ready:
            label = "ready"
        elif is_available:
            label = "available"
        else:
            label = "unknown"

        return Detection(
            value={
                "ready": bool(is_ready or is_available),
                "cooldown": bool(is_cooldown),
                "available": bool(is_available),
                "empty": False,
                "locked": False,
            },
            confidence=confidence,
            label=label,
            meta={
                "model": "cv_fallback",
                "brightness": mean_brightness,
                "dark_pct": dark_ratio,
                "overlay_pct": overlay_ratio,
                "edge_pct": edge_ratio,
            },
        )

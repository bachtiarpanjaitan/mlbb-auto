"""
CD Number Detector — Baca angka cooldown dari skill icon pakai CNN.

Model: models/cd_number_classifier.onnx
  Input:  70×70×3 BGR (sama dengan skill icon crop)
  Output: 3 digit (0-9 atau blank=10)

Cara pakai:
  det = CDNumberDetector()
  seconds = det.read(cropped_skill_icon)  # → 12, 5, atau None
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "cd_number_classifier.onnx"
)
INPUT_SIZE = 70
DIGIT_CLASSES = 11
BLANK_CLASS = 10  # 10 = blank/no digit


class CDNumberDetector:
    """Baca angka cooldown dari skill icon 70×70 pakai ONNX model."""

    def __init__(self):
        self._session = None
        self._load_model()

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            log.warning("CD model not found: %s", MODEL_PATH)
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(MODEL_PATH)
            log.info("Loaded CD number model: %s", MODEL_PATH)
        except Exception as e:
            log.warning("Failed to load CD model: %s", e)

    @property
    def available(self) -> bool:
        return self._session is not None

    def read(self, image: np.ndarray) -> int | None:
        """
        Baca angka cooldown dari skill icon crop.

        Args:
            image: 70×70×3 BGR crop.

        Returns:
            Int (1-120) kalau ada angka, None kalau gak terdeteksi.
        """
        if self._session is None:
            return None
        if image is None or image.size == 0:
            return None

        try:
            # Preprocess
            if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            tensor = (tensor - 0.5) / 0.5
            tensor = np.expand_dims(tensor, axis=0)

            # Inference
            outputs = self._session.run(
                ["digit_0", "digit_1", "digit_2"],
                {"input": tensor},
            )

            # Decode: skip blank di kiri, break pas blank setelah mulai
            result = ""
            started = False
            for out in outputs:
                digit = int(np.argmax(out[0]))
                if digit == BLANK_CLASS:
                    if started:
                        break
                    continue
                started = True
                result += str(digit)

            if result:
                return int(result)
            return None

        except Exception as e:
            log.error("CD number inference error: %s", e)
            return None

    def read_from_prob(self, image: np.ndarray) -> tuple[int | None, float]:
        """
        Baca angka + confidence.

        Returns:
            (cd_seconds, confidence) atau (None, 0.0).
        """
        if self._session is None:
            return None, 0.0
        if image is None or image.size == 0:
            return None, 0.0

        try:
            if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            tensor = (tensor - 0.5) / 0.5
            tensor = np.expand_dims(tensor, axis=0)

            outputs = self._session.run(
                ["digit_0", "digit_1", "digit_2"],
                {"input": tensor},
            )

            result = ""
            min_conf = 1.0
            started = False
            for out in outputs:
                probs = np.exp(out[0] - np.max(out[0]))
                probs /= np.sum(probs)
                digit = int(np.argmax(out[0]))
                conf = float(probs[digit])

                if digit == BLANK_CLASS:
                    if started:
                        break
                    continue
                started = True
                result += str(digit)
                min_conf = min(min_conf, conf)

            if result:
                return int(result), min_conf
            return None, 0.0

        except Exception as e:
            log.error("CD number inference error: %s", e)
            return None, 0.0

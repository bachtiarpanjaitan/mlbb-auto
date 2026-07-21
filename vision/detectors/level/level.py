"""
Level Detector — Mendeteksi hero level dari level indicator di hero panel.
"""

from __future__ import annotations

import re
import cv2
import numpy as np

from ..base import BaseDetector, Detection


class LevelDetector(BaseDetector):
    """Detect hero level number."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("level")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def detect(self, image: np.ndarray) -> Detection | None:
        text = self.ocr.read(image, hint="number")
        if not text:
            return None
        cleaned = re.sub(r"[^0-9]", "", text)
        try:
            level = int(cleaned)
            if 1 <= level <= 30:
                return Detection(
                    value=level, confidence=0.95, label=f"level_{level}",
                )
        except (ValueError, TypeError):
            pass
        return None

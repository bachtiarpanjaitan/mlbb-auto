"""
Gold Detector — Mendeteksi gold amount dari hero panel atau top bar gold comparison.
"""

from __future__ import annotations

import re
import cv2
import numpy as np

from ..base import BaseDetector, Detection


class GoldDetector(BaseDetector):
    """Detect hero gold amount from hero_panel.gold region."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("gold")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def detect(self, image: np.ndarray) -> Detection | None:
        text = self.ocr.read(image, hint="number")
        if not text:
            return None
        cleaned = re.sub(r"[^0-9]", "", text)
        try:
            value = int(cleaned)
            return Detection(
                value=value,
                confidence=min(1.0, len(cleaned) / 5.0),
                label="gold",
                meta={"raw": text, "digits": len(cleaned)},
            )
        except ValueError:
            return Detection(value=text, confidence=0.3, label="gold_raw")

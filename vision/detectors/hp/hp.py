"""
HP Detector — Mendeteksi persentase HP dari green HP bar di hero panel.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class HPDetector(BaseDetector):
    """Detect HP percentage from green health bar."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("hp")

    def detect(self, image: np.ndarray) -> Detection | None:
        pct = self._extract_bar_pct(
            image, hue_range=(45, 90), sat_min=60, val_min=60,
        )
        if pct is None:
            return None
        pct = min(1.0, max(0.0, pct))
        return Detection(
            value=round(pct, 4),
            confidence=0.9,
            label="hp",
            meta={"percentage": pct},
        )

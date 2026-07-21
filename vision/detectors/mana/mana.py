"""
Mana Detector — Mendeteksi persentase Mana/Energy dari blue bar di hero panel.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class ManaDetector(BaseDetector):
    """Detect mana/energy percentage from blue bar."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("mana")

    def detect(self, image: np.ndarray) -> Detection | None:
        pct = self._extract_bar_pct(
            image, hue_range=(100, 130), sat_min=60, val_min=60,
        )
        if pct is None:
            return None
        pct = min(1.0, max(0.0, pct))
        return Detection(
            value=round(pct, 4),
            confidence=0.9,
            label="mana",
            meta={"percentage": pct},
        )

"""
Skills Detector — Mendeteksi status skill: ready atau cooldown.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class SkillsDetector(BaseDetector):
    """
    Deteksi status skill:
      - Ready: icon cerah, tidak ada overlay gelap
      - Cooldown: ada overlay gelap di atas icon
      - Available (highlight): border terang
    """

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("skills")

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        dark_pixels = (gray < 60).mean()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 80, 255]))
        bright_pct = cv2.countNonZero(bright_mask) / gray.size if gray.size > 0 else 0

        cooldown = dark_pixels > 0.4
        available = bright_pct > 0.15 and not cooldown

        return Detection(
            value={
                "ready": not cooldown,
                "cooldown": bool(cooldown),
                "available": bool(available),
                "brightness": round(float(mean_brightness), 1),
                "dark_ratio": round(float(dark_pixels), 3),
            },
            confidence=0.85 if (cooldown or available) else 0.6,
            label="cooldown" if cooldown else ("ready" if available else "unknown"),
            meta={"brightness": float(mean_brightness), "dark_pct": float(dark_pixels)},
        )

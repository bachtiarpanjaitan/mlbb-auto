"""
Minimap Detector — Mengekstrak minimap dan mendeteksi hero dots.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class MinimapDetector(BaseDetector):
    """
    Mengekstrak minimap dan deteksi posisi hero dots.
      - Blue team: cyan/light blue dots
      - Red team: pink/red dots
    """

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("minimap")

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        blue_dots = self._find_dots(hsv, hue_center=100, hue_range=15)
        red_dots = self._find_dots(hsv, hue_center=170, hue_range=15)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, fog_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        fog_pct = cv2.countNonZero(fog_mask) / (h * w) if h * w > 0 else 0

        def _norm(dots):
            return [{"x": round(x / w, 3), "y": round(y / h, 3)} for x, y in dots]

        return Detection(
            value={
                "blue_dots": _norm(blue_dots),
                "red_dots": _norm(red_dots),
                "fog_of_war_pct": round(fog_pct, 3),
                "dimensions": {"width": w, "height": h},
            },
            confidence=0.85 if (blue_dots or red_dots) else 0.5,
            label="minimap",
            meta={"blue_count": len(blue_dots), "red_count": len(red_dots)},
        )

    def _find_dots(self, hsv, hue_center, hue_range):
        lower = np.array([hue_center - hue_range, 80, 100])
        upper = np.array([hue_center + hue_range, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        return dots

"""
Towers Detector — Mendeteksi status tower (berdiri / hancur)
dari ikon tower di kiri/kanan top bar.

MLBB memiliki 3 tower per lane × 3 lane = 9 tower per tim (termasuk base).
Tower yang masih berdiri = icon cerah, tower hancur = icon gelap/hilang.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class TowersDetector(BaseDetector):
    """
    Mendeteksi jumlah tower yang masih berdiri dari region icon tower
    di top_bar.blue_towers dan top_bar.red_towers.
    """

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("towers")
        # MLBB: 3 lanes × 3 towers + 1 base = normally up to 9 per side
        self._max_towers = 9

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold: tower icons are bright on dark bg
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Find connected components (individual tower icons)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)

        towers_alive = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filter: tower icons are roughly small vertical rectangles
            if 30 < area < 900 and 8 < w < 50 and 15 < h < 65:
                # Aspect ratio check: tower icon is taller than wide
                aspect = h / w if w > 0 else 1
                if 0.8 < aspect < 4.0:
                    towers_alive += 1

        towers_alive = min(towers_alive, self._max_towers)

        return Detection(
            value=towers_alive,
            confidence=0.8,
            label=f"towers_{towers_alive}",
            meta={
                "alive": towers_alive,
                "destroyed": self._max_towers - towers_alive,
                "components_found": num_labels - 1,
            },
        )

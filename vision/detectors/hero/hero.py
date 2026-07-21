"""
Hero Detector — Identifikasi hero dari portrait menggunakan ORB feature matching.
Template di-load dari assets/heroes/*.png
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from ..base import BaseDetector, Detection
from ...matcher import ORBMatcher


HERO_ASSETS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "heroes",
)


class HeroDetector(BaseDetector):
    """Identify hero from portrait image using ORB matching."""

    def __init__(self, ocr=None, templates_path: str = HERO_ASSETS):
        super().__init__(ocr)
        self.load_config("hero")
        self.matcher = ORBMatcher().load_from_config()
        self._loaded = False
        self._templates_path = templates_path or HERO_ASSETS

    def _load_templates(self):
        if self._loaded:
            return
        path = self._templates_path
        if not os.path.isdir(path):
            return
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(fname)[0]
                img = cv2.imread(os.path.join(path, fname))
                if img is not None:
                    self.matcher.add_template(name, img)
        self._loaded = True

    def detect(self, image: np.ndarray) -> Detection | None:
        self._load_templates()
        if not self.matcher.templates:
            return None
        result = self.matcher.match(image)
        if result and result.success:
            return Detection(
                value=result.label,
                confidence=result.confidence,
                label=result.label,
                meta={"matches": len(result.matches) if result.matches else 0},
            )
        return None

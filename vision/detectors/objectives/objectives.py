"""
Objectives Detector — Mendeteksi Lord & Turtle status.

Mendeteksi:
- Countdown timer Lord & Turtle (muncul ~30 detik sebelum spawn)
- Notifikasi saat Lord/Turtle diambil/destroyed
- Icon Lord/Turtle di minimap
"""

from __future__ import annotations

import re
import os
import cv2
import numpy as np

from ..base import BaseDetector, Detection
from ...matcher import TemplateMatcher


CREEP_ASSETS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "creeps"
)


class ObjectivesDetector(BaseDetector):
    """
    Mendeteksi status objektif (Lord, Turtle) dari timer dan notifikasi.
    Juga mencocokkan icon objektif di minimap via template matching.
    """

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("objectives")
        self.matcher = TemplateMatcher().load_from_config()
        self._loaded = False

    def _load_templates(self):
        if self._loaded:
            return
        if os.path.isdir(CREEP_ASSETS):
            for fname in os.listdir(CREEP_ASSETS):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    name = os.path.splitext(fname)[0]
                    img = cv2.imread(os.path.join(CREEP_ASSETS, fname))
                    if img is not None:
                        self.matcher.add_template(name, img)
                        print(f"  Loaded objective template: {name}")
        self._loaded = True

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        # Try to read countdown text
        text = self.ocr.read(image, hint="countdown")

        # Also try template matching for objective icons
        self._load_templates()
        template_result = None
        if self.matcher.templates:
            template_result = self.matcher.match(image)

        value = {
            "countdown_text": text,
            "countdown_seconds": self._parse_countdown(text) if text else None,
        }

        if template_result and template_result.success:
            value["icon_match"] = template_result.label
            value["icon_confidence"] = template_result.confidence

        confidence = 0.7 if text else (0.8 if template_result else 0.3)

        return Detection(
            value=value,
            confidence=confidence,
            label=template_result.label if template_result else "objective",
            meta={
                "raw_text": text,
                "template_match": template_result.label if template_result else None,
            },
        )

    def _parse_countdown(self, text: str) -> int | None:
        """Parse countdown text like '30s' or '0:30' to seconds."""
        cleaned = text.strip().lower()
        m = re.match(r"(\d+)\s*s", cleaned)
        if m:
            return int(m.group(1))
        m = re.match(r"(\d{1,2}):(\d{2})", cleaned)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        return None

"""
Items Detector — Mendeteksi item yang sudah dibeli dari 6 slot di hero panel.
Menggunakan template matching + database items.json.
"""

from __future__ import annotations

import os
import json
import cv2
import numpy as np

from ..base import BaseDetector, Detection
from ...matcher import TemplateMatcher


ITEMS_DB = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "databases", "items.json",
)


class ItemsDetector(BaseDetector):
    """Detect purchased items from hero panel item slots."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("items")
        self.matcher = TemplateMatcher().load_from_config()
        self._loaded = False
        self._item_db: dict = {}

    def _load_assets(self):
        if self._loaded:
            return
        if os.path.isfile(ITEMS_DB):
            with open(ITEMS_DB) as f:
                items = json.load(f)
                self._item_db = {
                    item.get("key", str(i)): item
                    for i, item in enumerate(items if isinstance(items, list) else [])
                }
        self._loaded = True

    def add_template(self, name: str, image: np.ndarray):
        """Register item icon template."""
        self.matcher.add_template(name, image)

    def detect(self, image: np.ndarray) -> Detection | None:
        self._load_assets()
        if not self.matcher.templates:
            return None
        result = self.matcher.match(image)
        if result and result.success:
            info = self._item_db.get(result.label, {})
            return Detection(
                value=result.label,
                confidence=result.confidence,
                label=result.label,
                meta={"item_info": info},
            )
        return None

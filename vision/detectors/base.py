"""
Base Detector — Abstract class untuk semua detector.

Semua detector harus extend BaseDetector dan implement:
    - detect(img) -> Detection | None
    - preprocess(img) -> np.ndarray (opsional override)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time
import logging

import cv2
import numpy as np

from ..core import layout as layout_mod
from ..ocr.reader import OCRReader

logger = logging.getLogger("mlbb.vision.detector")


@dataclass
class Detection:
    """Hasil deteksi standar."""
    value: Any
    confidence: float = 1.0
    label: str | None = None          # untuk template matching
    bbox: tuple | None = None         # region relatif
    elapsed_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """
    Base class untuk semua detector.

    Args:
        ocr: OCRReader instance (di-share antar detector).
        config: Konfigurasi spesifik detector dari layout.yaml.
    """

    def __init__(self, ocr: OCRReader | None = None):
        self.ocr = ocr or OCRReader()
        self._config: dict = {}

    @abstractmethod
    def detect(self, image: np.ndarray) -> Detection | None:
        """
        Detect value dari cropped region image.

        Args:
            image: Region gambar yang sudah di-crop sesuai layout.

        Returns:
            Detection jika berhasil, None jika gagal.
        """
        ...

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing sebelum detect. Override untuk custom preprocessing.
        Default: konversi ke grayscale jika perlu.
        """
        return image

    def load_config(self, detector_name: str):
        """Load config dari layout.yaml untuk detector ini."""
        configs = layout_mod.detectors()
        self._config = configs.get(detector_name, {})

        # Load OCR if configured
        if self._config.get("method") == "ocr":
            preprocess_steps = self._config.get("preprocess", [])
            if "invert_if_dark" in preprocess_steps:
                self._do_invert = True

        return self

    def run(self, image: np.ndarray) -> Detection | None:
        """Run detector with timing."""
        t0 = time.perf_counter()
        try:
            processed = self.preprocess(image)
            result = self.detect(processed)
            if result:
                result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as e:
            logger.warning("%s error: %s", self.__class__.__name__, e)
            return None

    def _extract_bar_pct(
        self,
        image: np.ndarray,
        hue_range: tuple[int, int],
        sat_min: int = 80,
        val_min: int = 80,
    ) -> float | None:
        """Extract bar fill percentage from a color bar (HP, Mana, dll)."""
        if image is None or image.size == 0:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hue_range[0], sat_min, val_min])
        upper = np.array([hue_range[1], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Analyze horizontal fill
        h, w = mask.shape
        row_fills = []
        for y in range(h):
            row = mask[y, :]
            filled = cv2.countNonZero(row)
            row_fills.append(filled / w if w > 0 else 0)

        # Take middle rows (ignore borders)
        mid = h // 2
        valid_fills = row_fills[max(0, mid-2):min(h, mid+3)]
        return sum(valid_fills) / len(valid_fills) if valid_fills else 0.0

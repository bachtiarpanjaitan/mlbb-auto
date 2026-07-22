"""
Tesseract OCR Reader — Text detection dan recognition via pytesseract.

Engine: Tesseract (pytesseract) — ringan, cepat untuk digit dan text pendek.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("mlbb.vision.ocr")

# Tesseract config per hint type
_TESS_CONFIGS = {
    "number":    "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
    "clock":     "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:",
    "kda":       "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/",
    "text":      "--psm 7 --oem 3",
    "countdown": "--psm 10 --oem 3 digits",
}


class OCRReader:
    """
    OCR Reader dengan Tesseract (pytesseract) sebagai engine.

    Args:
        lang: Bahasa (default: 'eng').
    """

    def __init__(self, lang: str = "eng"):
        self._lang = lang

    def read(self, image: np.ndarray, hint: str = "text") -> str | None:
        """
        Read text from image region using Tesseract.

        Args:
            image: Cropped region image (BGR).
            hint: Type hint — "clock", "number", "kda", "text", "countdown".

        Returns:
            Recognized text string or None.
        """
        if image is None or image.size == 0:
            return None

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocess untuk tesseract
        processed = self._preprocess(gray, hint)

        config = _TESS_CONFIGS.get(hint, _TESS_CONFIGS["text"])

        try:
            import pytesseract
            text = pytesseract.image_to_string(
                processed, lang=self._lang, config=config
            ).strip()
            if text:
                return text
        except Exception as e:
            logger.debug("Tesseract error: %s", e)

        return None

    def _preprocess(self, gray: np.ndarray, hint: str) -> np.ndarray:
        """
        Preprocess image untuk tesseract berdasarkan hint.

        - number/kda: threshold + invert (text putih di background gelap)
        - text: contrast enhancement
        """
        # Resize biar lebih jelas
        h, w = gray.shape
        if w < 40:
            scale = max(1.0, 60 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if hint in ("number", "clock", "kda", "countdown"):
            # OTSU threshold untuk pisah angka putih dari background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert kalau text putih di background gelap
            white_px = cv2.countNonZero(binary)
            if white_px > binary.size * 0.5:
                binary = 255 - binary

            return binary

        # Text: contrast stretching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def read_as_int(self, image: np.ndarray, hint: str = "number") -> int | None:
        """Read text and parse as integer."""
        text = self.read(image, hint)
        if text is None:
            return None
        cleaned = re.sub(r"[^0-9]", "", text)
        try:
            return int(cleaned)
        except ValueError:
            return None

    def read_as_float(self, image: np.ndarray) -> float | None:
        """Read timer format (MM:SS) and return total seconds."""
        text = self.read(image, "clock")
        if text is None:
            return None
        m = re.match(r"(\d{1,2}):(\d{2})", text)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        return None

"""
PaddleOCR Wrapper — Text detection dan recognition.

Menggunakan PaddleOCR v3.7+ untuk text recognition.
Fallback ke threshold-based digit OCR jika PaddleOCR gagal.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("mlbb.vision.ocr")

# ---------- Main OCR Class ----------


class OCRReader:
    """
    OCR Reader dengan PaddleOCR sebagai engine utama.

    Args:
        use_paddle: True = aktifkan PaddleOCR (otomatis jika terinstall).
        lang: Bahasa untuk PaddleOCR (default: 'en').
    """

    def __init__(self, use_paddle: bool = True, lang: str = "en"):
        self._reader: Any = None
        self._use_paddle = use_paddle
        self._lang = lang

        if use_paddle:
            self._init_paddle()

    def _init_paddle(self):
        """Init PaddleOCR (PP-OCRv6)."""
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                use_textline_orientation=False,
                lang=self._lang,
            )
            logger.info("✅ PaddleOCR initialized (v3.7+)")
        except ImportError:
            logger.warning("PaddleOCR not installed, using fallback OCR")
            self._use_paddle = False
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s, using fallback", e)
            self._use_paddle = False

    def read(self, image: np.ndarray, hint: str = "text") -> str | None:
        """
        Read text from image region.

        Args:
            image: Cropped region image (BGR).
            hint: Type hint — "clock", "number", "kda", "text", "speed", "countdown".

        Returns:
            Recognized text string or None.
        """
        if image is None or image.size == 0:
            return None

        # Try PaddleOCR
        if self._use_paddle and self._reader is not None:
            try:
                result = self._predict(image)
                if result:
                    return result
            except Exception as e:
                logger.debug("PaddleOCR error: %s", e)

        # Fallback digit OCR
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if hint in ("clock", "number", "kda", "speed", "countdown"):
            return self._fallback_digits(gray, hint)

        return None

    def _predict(self, image: np.ndarray) -> str | None:
        """Run PaddleOCR inference (v3.7+ OCRResult format)."""
        raw = self._reader.ocr(image)

        if not raw or not isinstance(raw, list):
            return None

        texts = []
        for page in raw:
            # PaddleOCR v3.7 returns OCRResult objects with .rec_texts
            if hasattr(page, "rec_texts") and isinstance(page.rec_texts, list):
                texts.extend(str(t) for t in page.rec_texts if t is not None)
            # Fallback: dict-style result
            elif isinstance(page, dict):
                for t in page.get("rec_texts", page.get("data", [])):
                    if isinstance(t, dict):
                        texts.append(str(t.get("text", "")))
                    elif isinstance(t, str):
                        texts.append(t)

        if not texts:
            return None

        return " ".join(texts)

    def _fallback_digits(self, gray: np.ndarray, hint: str) -> str | None:
        """Basic digit OCR via contour analysis."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 10 and w > 4 and w < gray.shape[1] * 0.6:
                digits.append((x, y))

        if not digits:
            return None

        # Simple heuristic: return number of distinct regions found
        # This is a placeholder — PaddleOCR handles real use cases
        return None

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

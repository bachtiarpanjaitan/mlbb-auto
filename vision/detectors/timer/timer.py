"""
Timer Detector — Mendeteksi game time (MM:SS) dari region match_timer di top bar.
"""

from __future__ import annotations

import re
import cv2
import numpy as np

from ..base import BaseDetector, Detection


class TimerDetector(BaseDetector):
    """Detect match timer in MM:SS format from top bar center."""

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("timer")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # PaddleOCR works better on color images — skip aggressive threshold
        # Just ensure image is BGR
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def detect(self, image: np.ndarray) -> Detection | None:
        raw = self.ocr.read(image, hint="clock")
        if not raw:
            return None

        # Clean: keep only digits, colon, and decimal
        cleaned = re.sub(r"[^0-9:.]", "", raw)

        # Try various timer patterns
        patterns = [
            r"(\d{1,2}):(\d{2})(?:\.\d)?",       # MM:SS or MM:SS.T
            r"(\d{1,2}):(\d{2})",                  # MM:SS
            r"(\d{2})(\d{2})",                      # MMSS (4 digits)
        ]

        for pattern in patterns:
            m = re.search(pattern, cleaned)
            if m:
                if len(m.groups()) == 2:
                    minutes = int(m.group(1))
                    seconds = int(m.group(2))
                    if 0 <= minutes < 100 and 0 <= seconds < 60:
                        fmt = f"{minutes:02d}:{seconds:02d}"
                        total_sec = minutes * 60 + seconds
                        return Detection(
                            value=fmt,
                            confidence=min(1.0, len(cleaned) / 5.0),
                            label="match_time",
                            meta={
                                "total_seconds": total_sec,
                                "minutes": minutes,
                                "seconds": seconds,
                                "raw": raw,
                            },
                        )

        # Fallback: return raw OCR with low confidence
        # Filter out obvious non-timer text ($, gold values)
        if "$" not in raw and len(raw) >= 3:
            return Detection(value=raw, confidence=0.3, meta={"raw": raw})

        return None

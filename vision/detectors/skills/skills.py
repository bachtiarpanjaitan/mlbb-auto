"""
Skills Detector — Mendeteksi status skill: ready atau cooldown.

Strategi deteksi multi-level:
  1. Cooldown overlay (layer gelap di atas icon): HSV darkness + edge loss
  2. Ready: icon clear dengan edge definition normal
  3. Available (highlight/border terang): HSV high-value pixels
"""

from __future__ import annotations

import cv2
import numpy as np

from ..base import BaseDetector, Detection


class SkillsDetector(BaseDetector):
    """
    Deteksi status skill dengan multi-level comparison.

    Menggabungkan:
      - Dark pixel ratio (overlay cooldown)
      - Variance/edge analysis (cooldown overlay bikin texture kabur)
      - Bright pixel ratio (highlight border untuk available)
    """

    def __init__(self, ocr=None):
        super().__init__(ocr)
        self.load_config("skills")
        # Thresholds (dapat di-override via layout.yaml detectors.skills)
        self._dark_threshold: float = 0.35       # min ratio pixel gelap untuk cooldown
        self._ready_brightness: float = 100.0    # min mean brightness untuk ready
        self._var_threshold: float = 80.0        # min variance untuk ready (cooldown overlay = low var)

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        dark_ratio = float((gray < 60).mean())
        variance = float(gray.var())

        # ── HSV-based cooldown overlay detection ──
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Dark overlay = low Value (V < 80) — tanpa batas Saturation
        dark_overlay = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        overlay_ratio = cv2.countNonZero(dark_overlay) / gray.size if gray.size > 0 else 0

        # Bright pixels = highlight/available border (high Value, semua Saturation)
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 255, 255]))
        bright_pct = cv2.countNonZero(bright_mask) / gray.size if gray.size > 0 else 0

        # ── Edge analysis ──
        # Cooldown overlay = edges tereduksi drastis
        edges = cv2.Canny(gray, 30, 100)
        edge_ratio = cv2.countNonZero(edges) / gray.size if gray.size > 0 else 0
        # Icon normal biasanya punya edge_ratio > 0.02, cooldown < 0.01
        has_edges = edge_ratio > 0.015

        # ── Decision logic ──
        # Cooldown: overlay gelap signifikan ATAU brightness sangat rendah + no edges
        is_cooldown = overlay_ratio > 0.35 or (dark_ratio > self._dark_threshold and not has_edges)

        # Available: border terang atau highlight tanpa overlay cooldown
        is_available = bright_pct > 0.12 and not is_cooldown

        # Ready: brightness normal, ada edges, dan tidak cooldown
        is_ready = not is_cooldown and (has_edges or mean_brightness > self._ready_brightness * 0.8)

        # Confidence score
        if is_cooldown and overlay_ratio > 0.5:
            confidence = 0.90
        elif is_cooldown:
            confidence = 0.80
        elif is_available:
            confidence = 0.85
        elif is_ready:
            confidence = 0.80
        else:
            confidence = 0.60

        # Label
        if is_cooldown:
            label = "cooldown"
        elif is_ready:
            label = "ready"
        elif is_available:
            label = "available"
        else:
            label = "unknown"

        return Detection(
            value={
                "ready": bool(is_ready),
                "cooldown": bool(is_cooldown),
                "available": bool(is_available),
                "brightness": round(mean_brightness, 1),
                "dark_ratio": round(dark_ratio, 3),
                "overlay_ratio": round(overlay_ratio, 3),
                "edge_ratio": round(edge_ratio, 4),
                "variance": round(variance, 1),
            },
            confidence=confidence,
            label=label,
            meta={
                "brightness": mean_brightness,
                "dark_pct": dark_ratio,
                "overlay_pct": overlay_ratio,
                "edge_pct": edge_ratio,
            },
        )

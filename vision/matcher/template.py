"""
Template Matcher — Direct pixel-based template matching via OpenCV.

Menggunakan cv2.TM_CCOEFF_NORMED untuk mencocokkan template dengan region.
Cocok untuk icon item, skill, tower, dan elemen UI yang bentuknya konsisten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np

from ..core import layout as layout_mod


@dataclass
class MatchResult:
    """Hasil matching: template ditemukan atau tidak."""
    success: bool
    label: str | None = None          # nama template yang cocok
    confidence: float = 0.0           # 0.0 – 1.0
    bbox: tuple[int, int, int, int] | None = None  # [x, y, w, h] relatif ke region
    center: tuple[int, int] | None = None           # (cx, cy)


class TemplateMatcher:
    """
    Direct template matching dengan scoring CCOEFF_NORMED.

    Args:
        threshold: Minimal confidence untuk dianggap match (0.0 – 1.0).
        templates: Dict {name: np.ndarray} atau path ke folder templates.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        templates: dict[str, np.ndarray] | None = None,
        scale_range: tuple[float, float] = (0.85, 1.15),
    ):
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = templates or {}
        self.scale_range = scale_range

    def add_template(self, name: str, image: np.ndarray):
        """Register template."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.templates[name] = image

    def load_from_config(self):
        """Load templates config from layout.yaml matchers section."""
        config = layout_mod.matchers().get("template", {})
        self.threshold = config.get("threshold", self.threshold)
        if "scale_range" in config:
            self.scale_range = tuple(config["scale_range"])
        return self

    def match(self, region: np.ndarray) -> MatchResult | None:
        """
        Cari template terbaik di dalam region.

        Args:
            region: Region gambar yang akan di-match (grayscale atau BGR).

        Returns:
            MatchResult terbaik atau None jika region kosong.
        """
        if region is None or region.size == 0:
            return None

        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        best: MatchResult | None = None

        for name, template in self.templates.items():
            if template.ndim == 3:
                t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                t_gray = template

            th, tw = t_gray.shape

            if gray.shape[0] < th or gray.shape[1] < tw:
                continue  # template larger than region

            result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= self.threshold and (best is None or max_val > best.confidence):
                best = MatchResult(
                    success=True,
                    label=name,
                    confidence=float(max_val),
                    bbox=(max_loc[0], max_loc[1], tw, th),
                    center=(max_loc[0] + tw // 2, max_loc[1] + th // 2),
                )

        return best

    def match_multi(
        self,
        region: np.ndarray,
        max_results: int = 5,
        min_distance: int = 20,
    ) -> list[MatchResult]:
        """
        Cari semua kemunculan template di region (multi-detection).

        Args:
            region: Region gambar.
            max_results: Maksimal hasil return.
            min_distance: Jarak minimal antar detection (non-max suppression).

        Returns:
            List MatchResult terurut oleh confidence descending.
        """
        if region is None or region.size == 0:
            return []

        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        all_matches: list[MatchResult] = []

        for name, template in self.templates.items():
            if template.ndim == 3:
                t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                t_gray = template

            th, tw = t_gray.shape
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue

            result = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)

            # Non-max suppression
            h, w = result.shape
            for y in range(h):
                for x in range(w):
                    val = result[y, x]
                    if val < self.threshold:
                        continue

                    # Check if this is local maximum
                    y1, y2 = max(0, y - min_distance), min(h, y + min_distance + 1)
                    x1, x2 = max(0, x - min_distance), min(w, x + min_distance + 1)
                    if val < result[y1:y2, x1:x2].max():
                        continue

                    all_matches.append(MatchResult(
                        success=True,
                        label=name,
                        confidence=float(val),
                        bbox=(x, y, tw, th),
                        center=(x + tw // 2, y + th // 2),
                    ))

        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        return all_matches[:max_results]

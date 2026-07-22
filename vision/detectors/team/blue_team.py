"""
Team Detectors — Deteksi 5 hero dari scoreboard untuk blue & red team.
Menggunakan TemplateMatcher (CCOEFF_NORMED) — template di-resize ke ukuran
portrait scoreboard lalu dicocokkan secara pixel-based.
"""

from __future__ import annotations

import os
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any

from ..base import BaseDetector, Detection
from ...matcher import TemplateMatcher


logger = logging.getLogger("mlbb.vision.team")


HERO_ASSETS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "heroes",
)
_DEBUG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".tmp"))


@dataclass
class HeroPortraitResult:
    """Hasil deteksi 1 hero portrait."""
    slot: int              # 1-5
    hero_name: str | None  # nama hero jika ketemu
    confidence: float      # confidence matching
    bbox: tuple[int, int, int, int]  # x, y, w, h di frame
    hp_pct: float | None = None  # persentase HP dari green bar


class TeamDetector(BaseDetector):
    """
    Deteksi 5 hero dari scoreboard untuk satu team.

    Args:
        team_name: "blue_team" atau "red_team" — nama key di layout.yaml
        templates_path: Path ke assets/heroes/
        confidence_threshold: Threshold template matching
    """

    def __init__(self, team_name: str = "blue_team", ocr=None,
                 templates_path: str = HERO_ASSETS, confidence_threshold: float = 0.30):
        super().__init__(ocr)
        self.team_name = team_name
        self.load_config(team_name)
        self.matcher = TemplateMatcher(threshold=confidence_threshold)
        self._loaded = False
        self._templates_path = templates_path or HERO_ASSETS
        self.confidence_threshold = confidence_threshold
        # Tiap slot: (portrait_key, hp_bar_key)
        self._slot_keys = [
            ("portrait_hero_1", "hp_bar_hero_1"),
            ("portrait_hero_2", "hp_bar_hero_2"),
            ("portrait_hero_3", "hp_bar_hero_3"),
            ("portrait_hero_4", "hp_bar_hero_4"),
            ("portrait_hero_5", "hp_bar_hero_5"),
        ]
        # Ukuran target template (dari bbox portrait pertama)
        self._target_w = 60
        self._target_h = 60
        first_cfg = self._config.get("portrait_hero_1", {})
        if first_cfg and "bbox" in first_cfg:
            _, _, self._target_w, self._target_h = first_cfg["bbox"]

    def _load_templates(self):
        if self._loaded:
            return
        path = self._templates_path
        if not os.path.isdir(path):
            logger.warning("Templates path not found: %s", path)
            return

        tw, th = self._target_w, self._target_h
        logger.info("[%s] Resizing hero templates to %dx%d", self.team_name, tw, th)

        count = 0
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(fname)[0]
                img = cv2.imread(os.path.join(path, fname))
                if img is not None:
                    resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    self.matcher.add_template(name, resized)
                    count += 1
        self._loaded = True
        logger.info("[%s] Loaded %d hero templates (resized to %dx%d)",
                    self.team_name, count, tw, th)

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        Detect 5 hero portraits dari scoreboard.

        Args:
            image: Full frame (2400x1080)

        Returns:
            Detection dengan value = dict {
                "heroes": [...],
                "detected_count": int,
                "all_detected": bool
            }
        """
        self._load_templates()
        if not self.matcher.templates:
            return Detection(
                value={"heroes": [], "detected_count": 0, "all_detected": False},
                confidence=0.0, label=self.team_name,
                meta={"error": "No templates loaded"}
            )

        region = self._config
        tw, th = self._target_w, self._target_h
        results: list[HeroPortraitResult] = []
        detected_count = 0

        for i, (pt_key, hp_key) in enumerate(self._slot_keys, 1):
            # ── Portrait ──
            portrait_cfg = region.get(pt_key, {})
            if not portrait_cfg or "bbox" not in portrait_cfg:
                logger.warning("[%s] Slot %d: no bbox for %s", self.team_name, i, pt_key)
                results.append(HeroPortraitResult(
                    slot=i, hero_name=None, confidence=0.0, bbox=(0, 0, 0, 0)
                ))
                continue

            px, py, pw, ph = portrait_cfg["bbox"]
            fh, fw = image.shape[:2]

            if not (0 <= px < fw and 0 <= py < fh and px + pw <= fw and py + ph <= fh):
                results.append(HeroPortraitResult(
                    slot=i, hero_name=None, confidence=0.0, bbox=(px, py, pw, ph)
                ))
                continue

            portrait_img = image[py:py+ph, px:px+pw]
            if portrait_img.size == 0:
                results.append(HeroPortraitResult(
                    slot=i, hero_name=None, confidence=0.0, bbox=(px, py, pw, ph)
                ))
                continue

            # Resize ke ukuran template
            if portrait_img.shape[1] != tw or portrait_img.shape[0] != th:
                portrait_img = cv2.resize(portrait_img, (tw, th), interpolation=cv2.INTER_AREA)

            # Debug: save once
            if not hasattr(self, '_debug_saved_crops'):
                self._debug_saved_crops = set()
            crop_key = f"{self.team_name}_portrait_{i}"
            if crop_key not in self._debug_saved_crops:
                self._debug_saved_crops.add(crop_key)
                cv2.imwrite(os.path.join(_DEBUG_DIR, f"{crop_key}.png"), portrait_img)
                logger.info("[%s] Saved %s.png to .tmp/", self.team_name, crop_key)

            match_result = self.matcher.match(portrait_img)
            if match_result and match_result.success:
                hero_name = match_result.label
                conf = match_result.confidence
                detected_count += 1
                logger.info("[%s] Slot %d MATCH: %s (conf=%.3f)",
                           self.team_name, i, hero_name, conf)
            else:
                hero_name = None
                conf = match_result.confidence if match_result else 0.0
                logger.info("[%s] Slot %d NO MATCH (conf=%.3f, threshold=%.2f)",
                           self.team_name, i, conf, self.confidence_threshold)

            # ── HP Bar ──
            hp_pct = None
            hp_cfg = region.get(hp_key, {})
            if hp_cfg and "bbox" in hp_cfg:
                hx, hy, hw, hh = hp_cfg["bbox"]
                if 0 <= hx < fw and 0 <= hy < fh and hx + hw <= fw and hy + hh <= fh:
                    hp_img = image[hy:hy+hh, hx:hx+hw]
                    if hp_img.size > 0:
                        raw = self._extract_bar_pct(
                            hp_img, hue_range=(45, 90), sat_min=40, val_min=30,
                        )
                        if raw is not None and raw > 0.01:
                            hp_pct = round(min(1.0, raw), 4)

            results.append(HeroPortraitResult(
                slot=i, hero_name=hero_name,
                confidence=round(conf, 3), bbox=(px, py, pw, ph),
                hp_pct=hp_pct,
            ))

        all_detected = detected_count == 5

        return Detection(
            value={
                "heroes": [
                    {"slot": r.slot, "hero_name": r.hero_name,
                     "confidence": r.confidence, "bbox": r.bbox,
                     "hp_pct": r.hp_pct}
                    for r in results
                ],
                "detected_count": detected_count,
                "all_detected": all_detected,
            },
            confidence=0.9 if all_detected else 0.5,
            label=self.team_name,
            meta={"slots_processed": 5}
        )


class BlueTeamDetector(TeamDetector):
    """Deteksi 5 hero tim biru."""
    def __init__(self, **kwargs):
        super().__init__(team_name="blue_team", **kwargs)


class RedTeamDetector(TeamDetector):
    """Deteksi 5 hero tim merah."""
    def __init__(self, **kwargs):
        super().__init__(team_name="red_team", **kwargs)


def create_blue_team_detector(confidence_threshold: float = 0.30) -> BlueTeamDetector:
    return BlueTeamDetector(confidence_threshold=confidence_threshold)


def create_red_team_detector(confidence_threshold: float = 0.30) -> RedTeamDetector:
    return RedTeamDetector(confidence_threshold=confidence_threshold)

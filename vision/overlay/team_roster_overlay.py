"""
Team Roster Overlay — Visualisasi roster hero di frame/video.

Menampilkan:
- Tim Biru: [Portrait] Nama Hero
- Tim Merah: [Portrait] Nama Hero
- Status scanning (scanning... / complete)
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from ..scanners.team_roster_scanner import TeamRoster, HeroInfo


@dataclass
class OverlayConfig:
    """Konfigurasi tampilan overlay."""
    # Posisi overlay
    x: int = 50
    y: int = 50
    width: int = 350
    line_height: int = 35
    portrait_size: int = 30

    # Warna
    bg_color: tuple[int, int, int] = (0, 0, 0)          # hitam semi-transparan
    bg_alpha: float = 0.7
    blue_color: tuple[int, int, int] = (100, 200, 255)  # biru muda
    red_color: tuple[int, int, int] = (255, 100, 100)   # merah muda
    white: tuple[int, int, int] = (255, 255, 255)
    gray: tuple[int, int, int] = (180, 180, 180)
    green: tuple[int, int, int] = (100, 255, 100)

    # Font
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 1

    # Asset path
    hero_assets_path: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "assets", "heroes"
    )


class TeamRosterOverlay:
    """
    Render team roster ke frame.

    Usage:
        overlay = TeamRosterOverlay()
        frame = overlay.draw(frame, roster, scanning=True)
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        self._portrait_cache: dict[str, np.ndarray] = {}

    def draw(
        self,
        frame: np.ndarray,
        roster: TeamRoster,
        scanning: bool = True,
        scan_count: int = 0,
    ) -> np.ndarray:
        """
        Draw roster overlay on frame.

        Args:
            frame: Input frame (BGR)
            roster: TeamRoster object
            scanning: True jika masih scanning
            scan_count: Jumlah scan yang sudah dilakukan

        Returns:
            Frame dengan overlay
        """
        if frame is None or frame.size == 0:
            return frame

        overlay = frame.copy()
        cfg = self.config

        # Calculate overlay height
        blue_count = len(roster.blue)
        red_count = len(roster.red)
        total_lines = 2 + blue_count + 1 + red_count + 1  # headers + heroes + status
        overlay_h = total_lines * cfg.line_height + 20

        # Draw semi-transparent background
        x, y = cfg.x, cfg.y
        w, h = cfg.width, overlay_h
        cv2.rectangle(overlay, (x, y), (x + w, y + h), cfg.bg_color, -1)
        cv2.addWeighted(overlay, cfg.bg_alpha, frame, 1 - cfg.bg_alpha, 0, frame)

        # Reset overlay for text drawing
        overlay = frame.copy()

        # Draw content
        cur_y = y + 15

        # ── BLUE TEAM ─────────────────────────────────────────────────
        self._draw_team_header(overlay, "TIM BIRU (ALLY)", cfg.blue_color, x, cur_y)
        cur_y += cfg.line_height

        for hero in roster.blue:
            cur_y = self._draw_hero_row(overlay, hero, cfg.blue_color, x, cur_y, cfg)
            cur_y += 2

        cur_y += 10  # spacer

        # ── RED TEAM ──────────────────────────────────────────────────
        self._draw_team_header(overlay, "TIM MERAH (ENEMY)", cfg.red_color, x, cur_y)
        cur_y += cfg.line_height

        for hero in roster.red:
            cur_y = self._draw_hero_row(overlay, hero, cfg.red_color, x, cur_y, cfg)
            cur_y += 2

        cur_y += 10

        # ── STATUS ────────────────────────────────────────────────────
        if scanning:
            status_text = f"Scanning... ({scan_count}) | {len(roster.blue)}/5 Blue, {len(roster.red)}/5 Red"
            status_color = cfg.gray
        else:
            status_text = "Roster Complete!" if roster.is_complete else "Scan Timeout"
            status_color = cfg.green if roster.is_complete else (0, 165, 255)  # orange

        cv2.putText(
            overlay, status_text,
            (x + 10, cur_y),
            cfg.font, cfg.font_scale, status_color, cfg.font_thickness
        )

        return overlay

    def _draw_team_header(self, frame: np.ndarray, text: str, color: tuple, x: int, y: int):
        """Draw team header."""
        cfg = self.config
        cv2.putText(
            frame, text,
            (x + 10, y),
            cfg.font, cfg.font_scale + 0.1, color, cfg.font_thickness + 1
        )

    def _draw_hero_row(
        self,
        frame: np.ndarray,
        hero: HeroInfo,
        team_color: tuple,
        x: int,
        y: int,
        cfg: OverlayConfig
    ) -> int:
        """Draw single hero row: [portrait] nama hero."""
        ps = cfg.portrait_size

        # Load portrait
        portrait = self._get_portrait(hero.name, (ps, ps))

        # Draw portrait box
        px, py = x + 10, y
        if portrait is not None:
            frame[py:py+ps, px:px+ps] = portrait
        else:
            # Placeholder box
            cv2.rectangle(frame, (px, py), (px+ps, py+ps), team_color, 2)
            cv2.putText(
                frame, "?",
                (px + ps//2 - 5, py + ps//2 + 5),
                cfg.font, 0.5, team_color, 1
            )

        # Draw hero name
        name_x = px + ps + 10
        name_y = py + ps - 5
        cv2.putText(
            frame, hero.name,
            (name_x, name_y),
            cfg.font, cfg.font_scale, cfg.white, cfg.font_thickness
        )

        # Draw confidence indicator (small dot)
        conf_color = cfg.green if hero.confidence > 0.6 else (0, 255, 255)  # yellow if low
        cv2.circle(frame, (name_x - 15, name_y - 10), 4, conf_color, -1)

        return y + cfg.line_height

    def _get_portrait(self, hero_name: str, size: tuple[int, int]) -> Optional[np.ndarray]:
        """Load hero portrait from assets, resize to size."""
        if hero_name in self._portrait_cache:
            cached = self._portrait_cache[hero_name]
            if cached.shape[:2] == size[::-1]:
                return cached

        # Try multiple naming conventions
        possible_names = [
            hero_name,
            hero_name.lower(),
            hero_name.replace(" ", "_"),
            hero_name.replace(" ", ""),
        ]

        for name in possible_names:
            for ext in [".png", ".jpg", ".jpeg"]:
                path = os.path.join(self.config.hero_assets_path, f"{name}{ext}")
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                        self._portrait_cache[hero_name] = resized
                        return resized

        return None


def create_overlay(**kwargs) -> TeamRosterOverlay:
    """Factory function."""
    config = OverlayConfig(**kwargs)
    return TeamRosterOverlay(config)
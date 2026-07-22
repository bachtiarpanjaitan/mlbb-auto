"""
Team HP Tracker — Thread khusus untuk membaca HP bar 5 hero kedua tim secara periodik.

Berjalan di background thread terpisah, membaca HP bar blue_team dan red_team
setiap ~0.5 detik.
"""

from __future__ import annotations

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..core import layout as layout_mod
from ..core.frame_reader import FrameReader

logger = logging.getLogger("mlbb.vision.team_hp")

_DEBUG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".tmp"))


@dataclass
class SlotHP:
    """HP state satu slot hero."""
    team: str           # "blue" | "red"
    slot: int           # 1-5
    hp_pct: float | None = None
    last_update_frame: int = 0
    last_update_time: float = 0.0


class TeamHPTracker:
    """
    Monitor HP bar 5 hero blue team + 5 hero red team di background thread.
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        interval_sec: float = 0.5,
    ):
        self.reader = frame_reader
        self.interval_sec = interval_sec

        # Load HP bar bboxes kedua tim dari layout
        self._hp_bboxes: list[dict] = self._load_hp_bars()
        logger.info("Loaded %d HP bar bboxes from layout (blue + red)", len(self._hp_bboxes))

        # Shared state: 10 slots (5 blue + 5 red)
        self._slots: list[SlotHP] = [
            SlotHP(team="blue", slot=i) for i in range(1, 6)
        ] + [
            SlotHP(team="red", slot=i) for i in range(1, 6)
        ]
        self._lock = threading.Lock()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None
        self._target_frame: int | None = None

    def _load_hp_bars(self) -> list[dict]:
        """Load HP bar bboxes dari blue_team dan red_team di layout."""
        configs = layout_mod.detectors()
        bboxes = []
        for team in ("blue_team", "red_team"):
            cfg = configs.get(team, {})
            for i in range(1, 6):
                hp_cfg = cfg.get(f"hp_bar_hero_{i}", {})
                if hp_cfg and "bbox" in hp_cfg:
                    x, y, w, h = hp_cfg["bbox"]
                    bboxes.append({
                        "team": team.replace("_team", ""),
                        "slot": i,
                        "bbox": (x, y, w, h),
                        "key": f"hp_bar_hero_{i}",
                    })
        return bboxes

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Team HP tracker started (interval=%.1fs, 10 slots)", self.interval_sec)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        frame_idx = 0
        while self._running:
            with self._lock:
                if self._target_frame is not None:
                    frame_idx = self._target_frame
                    self._target_frame = None

            frame = self.reader.read(frame_idx)
            if frame is None:
                frame_idx = 0
                time.sleep(self.interval_sec)
                continue

            video_time = frame_idx / self.reader.fps
            self._extract_all_hp(frame, frame_idx, video_time)

            frame_idx += 1
            if frame_idx >= self.reader.total_frames:
                frame_idx = 0

            time.sleep(self.interval_sec)

        self._running = False

    def _extract_all_hp(self, frame: np.ndarray, frame_idx: int, video_time: float):
        """Extract HP untuk semua slot (blue + red)."""
        fh, fw = frame.shape[:2]

        for hp_info in self._hp_bboxes:
            team = hp_info["team"]
            slot = hp_info["slot"]
            hx, hy, hw, hh = hp_info["bbox"]

            if not (0 <= hx < fw and 0 <= hy < fh and hx + hw <= fw and hy + hh <= fh):
                continue

            hp_img = frame[hy:hy+hh, hx:hx+hw]
            if hp_img.size == 0:
                continue

            # Debug: save first time
            debug_slot_name = f"hp_{team}_{slot}"
            if not hasattr(self, '_debug_saved_hp'):
                self._debug_saved_hp = set()
            if debug_slot_name not in self._debug_saved_hp:
                self._debug_saved_hp.add(debug_slot_name)
                cv2.imwrite(os.path.join(_DEBUG_DIR, f"{debug_slot_name}.png"), hp_img)

            # Extract green bar fill (same as main hero)
            hp_pct = self._extract_green_pct(hp_img)

            # Update shared state
            with self._lock:
                for s in self._slots:
                    if s.team == team and s.slot == slot:
                        s.hp_pct = hp_pct
                        s.last_update_frame = frame_idx
                        s.last_update_time = video_time
                        break

    def _extract_green_pct(self, image: np.ndarray) -> float | None:
        """Sama seperti BaseDetector._extract_bar_pct — count green pixels per row."""
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([45, 40, 30])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        mid = h // 2
        start_r = max(0, mid - 2)
        end_r = min(h, mid + 3)
        mid_rows = mask[start_r:end_r, :]
        if mid_rows.size == 0:
            return None

        ratios = []
        for y in range(mid_rows.shape[0]):
            row = mid_rows[y, :]
            green = cv2.countNonZero(row)
            ratios.append(green / w if w > 0 else 0.0)

        pct = sum(ratios) / len(ratios)
        return min(1.0, pct) if pct > 0.01 else None

    # ── Public API ──

    def get_hp(self, team: str, slot: int) -> float | None:
        with self._lock:
            for s in self._slots:
                if s.team == team and s.slot == slot:
                    return s.hp_pct
        return None

    def get_all_hp(self) -> list[dict]:
        """Return [{"team": "blue", "slot": 1, "hp_pct": 0.75}, ...]"""
        with self._lock:
            return [
                {"team": s.team, "slot": s.slot, "hp_pct": s.hp_pct}
                for s in self._slots
            ]

    def get_team_hp(self, team: str) -> dict[int, float | None]:
        """Return {slot: hp_pct} untuk satu team."""
        with self._lock:
            return {s.slot: s.hp_pct for s in self._slots if s.team == team}

    def sync_frame(self, frame_idx: int):
        with self._lock:
            self._target_frame = frame_idx


def create_team_hp_tracker(frame_reader: FrameReader, **kwargs) -> TeamHPTracker:
    return TeamHPTracker(frame_reader, **kwargs)
"""
Team HP Tracker — Thread khusus untuk membaca HP bar 5 hero tim biru secara periodik.

Berjalan di background thread terpisah dari hero scanner, sehingga HP
bisa di-update lebih sering (setiap ~0.5s video time).
"""

from __future__ import annotations

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from ..core import layout as layout_mod
from ..core.frame_reader import FrameReader

logger = logging.getLogger("mlbb.vision.team_hp")

# Debug dir
_DEBUG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".tmp"))


@dataclass
class SlotHP:
    """HP state satu slot hero."""
    slot: int
    hp_pct: float | None = None
    last_update_frame: int = 0
    last_update_time: float = 0.0


class TeamHPTracker:
    """
    Monitor HP bar 5 hero tim biru di background thread.

    Flow:
    - Setup dengan FrameReader + layout config
    - Thread loop membaca frame, crop HP bar per slot, extract green %
    - Menyimpan HP terbaru untuk diakses thread utama
    - Interval update bisa dikonfigurasi (default 0.5s video)
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        interval_sec: float = 0.5,
        hue_range: tuple[int, int] = (45, 90),
        sat_min: int = 60,
        val_min: int = 60,
    ):
        self.reader = frame_reader
        self.interval_sec = interval_sec
        self.hue_range = hue_range
        self.sat_min = sat_min
        self.val_min = val_min

        # Load HP bar bboxes dari layout
        self._hp_bboxes: list[dict] = self._load_hp_bars()
        logger.info("Loaded %d HP bar bboxes from layout", len(self._hp_bboxes))

        # Shared state
        self._slots: list[SlotHP] = [
            SlotHP(slot=i+1) for i in range(5)
        ]
        self._lock = threading.Lock()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._target_frame: int | None = None  # sync dari main loop

    def _load_hp_bars(self) -> list[dict]:
        """Load HP bar bboxes dari layout.yaml blue_team section."""
        config = layout_mod.detectors().get("blue_team", {})
        bboxes = []
        for i in range(1, 6):
            hp_cfg = config.get(f"hp_bar_hero_{i}", {})
            if hp_cfg and "bbox" in hp_cfg:
                x, y, w, h = hp_cfg["bbox"]
                bboxes.append({
                    "slot": i,
                    "bbox": (x, y, w, h),
                    "key": f"hp_bar_hero_{i}",
                })
        return bboxes

    def start(self):
        """Mulai thread monitoring HP bar."""
        if self._running:
            return
        self._running = True
        self._frame_count = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Team HP tracker started (interval=%.1fs)", self.interval_sec)

    def stop(self):
        """Hentikan thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        """
        Loop utama: baca frame secara periodik, ekstrak HP.

        - Baca frame di posisi _target_frame jika diset (sync dari main loop)
        - Atau baca frame sequential dari 0 jika tidak ada target
        - Loop terus (indefinite), wrap ke 0 saat sampai akhir video
        - Pacing: sleep interval_sec detik tiap iterasi
        """
        fps = self.reader.fps
        frame_idx = 0

        while self._running:
            # Pakai target frame dari main loop jika ada
            with self._lock:
                if self._target_frame is not None:
                    frame_idx = self._target_frame
                    self._target_frame = None

            frame = self.reader.read(frame_idx)
            if frame is None:
                frame_idx = 0
                time.sleep(self.interval_sec)
                continue

            video_time = frame_idx / fps
            self._extract_all_hp(frame, frame_idx, video_time)

            # Sequential fallback: maju 1 frame (bukan frames_per_step)
            # supaya lebih responsif saat tidak ada target
            frame_idx += 1
            if frame_idx >= self.reader.total_frames:
                frame_idx = 0

            time.sleep(self.interval_sec)

        self._running = False

    def sync_frame(self, frame_idx: int):
        """Sync tracker ke frame tertentu (dipanggil dari main loop)."""
        with self._lock:
            self._target_frame = frame_idx

    def _extract_all_hp(self, frame: np.ndarray, frame_idx: int, video_time: float):
        """Extract HP untuk semua slot dari 1 frame."""
        fh, fw = frame.shape[:2]

        for hp_info in self._hp_bboxes:
            slot = hp_info["slot"]
            hx, hy, hw, hh = hp_info["bbox"]

            # Bounds check
            if not (0 <= hx < fw and 0 <= hy < fh and hx + hw <= fw and hy + hh <= fh):
                logger.warning("Slot %d HP: bbox [%d,%d,%d,%d] out of frame [%dx%d]",
                               slot, hx, hy, hw, hh, fw, fh)
                continue

            hp_img = frame[hy:hy+hh, hx:hx+hw]
            if hp_img.size == 0:
                continue

            # Debug: save cropped HP bar + log metadata (first time only)
            if not hasattr(self, '_debug_saved_hp'):
                self._debug_saved_hp = set()
            hp_key = f"hp_bar_{slot}"
            if hp_key not in self._debug_saved_hp:
                self._debug_saved_hp.add(hp_key)
                cv2.imwrite(os.path.join(_DEBUG_DIR, f"{hp_key}.png"), hp_img)
                hsv_mean = cv2.mean(cv2.cvtColor(hp_img, cv2.COLOR_BGR2HSV))
                logger.info("Saved %s.png (%dx%d) meanHSV=(%.0f,%.0f,%.0f) pos=[%d,%d,%d,%d]",
                           hp_key, hw, hh, hsv_mean[0], hsv_mean[1], hsv_mean[2], hx, hy, hw, hh)

            # Extract bar fill percentage (detect any fill, not just green)
            hp_pct = self._extract_bar_fill(hp_img, hp_key)

            # Update shared state
            with self._lock:
                idx = slot - 1
                if 0 <= idx < len(self._slots):
                    if hp_pct is not None:
                        logger.debug("Slot %d HP: %.1f%%", slot, hp_pct * 100)
                    self._slots[idx].hp_pct = hp_pct
                    self._slots[idx].last_update_frame = frame_idx
                    self._slots[idx].last_update_time = video_time

    def _extract_bar_fill(self, image: np.ndarray, debug_key: str = "") -> float | None:
        """
        Deteksi fill ratio — SAMA persis seperti main hero (_extract_bar_pct).

        Count green pixels per middle row → average fill ratio.
        Threshold S=40, V=30 (lebih rendah dari main hero karena bar team lebih gelap).
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([45, 40, 30])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Baris tengah (seperti _extract_bar_pct)
        mid = h // 2
        start_r = max(0, mid - 2)
        end_r = min(h, mid + 3)
        mid_rows = mask[start_r:end_r, :]

        if mid_rows.size == 0:
            return None

        # Hitung green pixel ratio per baris (seperti _extract_bar_pct)
        ratios = []
        for y in range(mid_rows.shape[0]):
            row = mid_rows[y, :]
            green = cv2.countNonZero(row)
            ratios.append(green / w if w > 0 else 0.0)

        fill_pct = sum(ratios) / len(ratios)

        # Debug
        if debug_key and not hasattr(self, '_debug_saved_mask'):
            if not hasattr(self, '_debug_saved_mask_set'):
                self._debug_saved_mask_set = set()
            if debug_key not in self._debug_saved_mask_set:
                self._debug_saved_mask_set.add(debug_key)
                cv2.imwrite(os.path.join(_DEBUG_DIR, f"{debug_key}_green.png"), mask)
                fill_str = f"{fill_pct:.1%}" if fill_pct > 0 else "0%"
                logger.info("Saved %s_green.png — fill=%s (H=45-90 S>=40 V>=30)",
                           debug_key, fill_str)

        if fill_pct > 0.01:
            return min(1.0, fill_pct)
        return None

    # ── Public API (thread-safe) ──

    def get_hp(self, slot: int) -> float | None:
        """HP% untuk slot tertentu (1-5)."""
        with self._lock:
            idx = slot - 1
            if 0 <= idx < len(self._slots):
                return self._slots[idx].hp_pct
        return None

    def get_all_hp(self) -> list[dict]:
        """Semua HP dalam list [{"slot": 1, "hp_pct": 0.75}, ...]."""
        with self._lock:
            return [
                {"slot": s.slot, "hp_pct": s.hp_pct}
                for s in self._slots
            ]

    def get_hp_dict(self) -> dict[int, float | None]:
        """Dict {slot: hp_pct}."""
        with self._lock:
            return {s.slot: s.hp_pct for s in self._slots}

    @property
    def is_running(self) -> bool:
        return self._running


def create_team_hp_tracker(frame_reader: FrameReader, **kwargs) -> TeamHPTracker:
    """Factory function."""
    return TeamHPTracker(frame_reader, **kwargs)
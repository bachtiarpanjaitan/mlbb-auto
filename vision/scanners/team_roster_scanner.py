"""
Team Roster Scanner — Periodic scanner untuk deteksi hero lineup.

Runs every 3 seconds until all 10 heroes (5 blue + 5 red) are identified.
Stops when roster is complete or timeout reached.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..core.layout import bbox
from ..core.frame_reader import FrameReader
from ..detectors.team import BlueTeamDetector, create_blue_team_detector
from ..matcher import ORBMatcher


@dataclass
class HeroInfo:
    """Informasi hero yang terdeteksi."""
    name: str
    team: str          # "blue" | "red"
    slot: int          # 1-5
    confidence: float
    portrait_bbox: tuple[int, int, int, int]  # x, y, w, h in frame coords


@dataclass
class TeamRoster:
    """Complete team roster."""
    blue: list[HeroInfo] = field(default_factory=list)
    red: list[HeroInfo] = field(default_factory=list)

    @property
    def blue_names(self) -> list[str]:
        return [h.name for h in self.blue]

    @property
    def red_names(self) -> list[str]:
        return [h.name for h in self.red]

    @property
    def is_complete(self) -> bool:
        return len(self.blue) >= 5 and len(self.red) >= 5

    @property
    def all_heroes(self) -> list[HeroInfo]:
        return self.blue + self.red

    def get_hero(self, name: str) -> Optional[HeroInfo]:
        for h in self.all_heroes:
            if h.name.lower() == name.lower():
                return h
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blue": [{"name": h.name, "slot": h.slot, "confidence": h.confidence} for h in self.blue],
            "red": [{"name": h.name, "slot": h.slot, "confidence": h.confidence} for h in self.red],
            "complete": self.is_complete,
        }


class TeamRosterScanner:
    """
    Scanner periodik untuk deteksi lineup hero.

    Flow:
    1. Start scanner dengan interval 3 detik
    2. Setiap interval: capture frame → detect blue team → detect red team
    3. Akumulasi hero yang terdeteksi (dedup by name)
    4. Stop ketika 10 hero terdeteksi atau timeout
    5. Callback on_complete(roster) atau on_update(roster)
    """

    def __init__(
        self,
        frame_reader: FrameReader,
        interval_sec: float = 3.0,
        timeout_sec: float = 60.0,
        confidence_threshold: float = 0.35,
        on_update: Optional[Callable[[TeamRoster], None]] = None,
        on_complete: Optional[Callable[[TeamRoster], None]] = None,
    ):
        self.frame_reader = frame_reader
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec
        self.on_update = on_update
        self.on_complete = on_complete

        # Detectors
        self.blue_detector = create_blue_team_detector(confidence_threshold)
        self.red_detector = create_blue_team_detector(confidence_threshold)  # same logic, different region

        # Matcher untuk hero portrait (shared)
        self.matcher = ORBMatcher().load_from_config()

        # State
        self.roster = TeamRoster()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._scan_count = 0

        # Layout bbox untuk scoreboard regions
        self._blue_bbox = bbox("scoreboard", "blue_team") or (450, 250, 1500, 500)
        self._red_bbox = bbox("scoreboard", "red_team") or (450, 250, 1500, 500)

    def start(self, frame_idx: int = 0):
        """Mulai scanning di background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._scan_count = 0
        self.roster = TeamRoster()
        self._thread = threading.Thread(target=self._run_loop, args=(frame_idx,), daemon=True)
        self._thread.start()

    def stop(self):
        """Hentikan scanner."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self, start_frame: int):
        frame_idx = start_frame
        while self._running:
            # Check timeout
            if time.time() - self._start_time > self.timeout_sec:
                print(f"[TeamRosterScanner] Timeout ({self.timeout_sec}s), stopping")
                break

            # Check completion
            if self.roster.is_complete:
                print(f"[TeamRosterScanner] Roster complete! {len(self.roster.blue)} blue, {len(self.roster.red)} red")
                if self.on_complete:
                    self.on_complete(self.roster)
                break

            # Read frame
            frame = self.frame_reader.read(frame_idx)
            if frame is None:
                frame_idx += 30  # skip ~1 detik
                time.sleep(self.interval_sec)
                continue

            # Scan this frame
            self._scan_frame(frame, frame_idx)
            self._scan_count += 1

            # Callback update
            if self.on_update:
                self.on_update(self.roster)

            # Next frame (interval * fps)
            frame_idx += int(self.interval_sec * self.frame_reader.fps)
            time.sleep(self.interval_sec)

        self._running = False

    def _scan_frame(self, frame: np.ndarray, frame_idx: int):
        """Scan single frame untuk blue + red team."""
        # Crop blue team region
        bx, by, bw, bh = self._blue_bbox
        blue_region = frame[by:by+bh, bx:bx+bw] if bw > 0 and bh > 0 else None

        # Crop red team region
        rx, ry, rw, rh = self._red_bbox
        red_region = frame[ry:ry+rh, rx:rx+rw] if rw > 0 and rh > 0 else None

        # Detect blue team
        if blue_region is not None and blue_region.size > 0:
            blue_result = self.blue_detector.detect(blue_region)
            if blue_result and blue_result.value:
                self._process_team_result(blue_result.value, "blue", frame_idx, (bx, by))

        # Detect red team
        if red_region is not None and red_region.size > 0:
            red_result = self.red_detector.detect(red_region)
            if red_result and red_result.value:
                self._process_team_result(red_result.value, "red", frame_idx, (rx, ry))

    def _process_team_result(
        self,
        portraits: list,
        team: str,
        frame_idx: int,
        region_offset: tuple[int, int]
    ):
        """Process detection result, add unique heroes to roster."""
        ox, oy = region_offset
        for slot_idx, p in enumerate(portraits, 1):
            if not p.name or p.confidence < 0.3:
                continue

            # Check if already in roster (by name)
            existing = self.roster.get_hero(p.name)
            if existing:
                # Update confidence if higher
                if p.confidence > existing.confidence:
                    existing.confidence = p.confidence
                continue

            # Add new hero
            hero = HeroInfo(
                name=p.name,
                team=team,
                slot=slot_idx,
                confidence=p.confidence,
                portrait_bbox=(
                    ox + p.bbox[0],
                    oy + p.bbox[1],
                    p.bbox[2],
                    p.bbox[3],
                ),
            )
            if team == "blue":
                self.roster.blue.append(hero)
            else:
                self.roster.red.append(hero)

            print(f"[TeamRosterScanner] Detected: {p.name} ({team} slot {slot_idx}) "
                  f"conf={p.confidence:.2f}")

    def get_roster(self) -> TeamRoster:
        """Get current roster."""
        return self.roster

    def wait_complete(self, timeout: Optional[float] = None) -> TeamRoster:
        """Block until complete or timeout."""
        if self._thread:
            self._thread.join(timeout=timeout or self.timeout_sec)
        return self.roster


def create_scanner(
    frame_reader: FrameReader,
    **kwargs
) -> TeamRosterScanner:
    """Factory function."""
    return TeamRosterScanner(frame_reader, **kwargs)
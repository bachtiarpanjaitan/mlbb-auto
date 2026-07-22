#!/usr/bin/env python3
"""
Demo: Blue Team Hero Detection + Overlay

Flow:
1. Load video
2. Start detector thread: scan every 3 seconds until 5 heroes detected
3. When done, display overlay with [portrait][nama hero] grouped by team
4. Stop scanning
"""

from __future__ import annotations

import os
import sys
import time
import threading
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.core.frame_reader import FrameReader
from vision.core.pipeline import Pipeline
from vision.detectors.blue_team_detector import BlueTeamDetector
from vision.overlay import TeamRosterOverlay, OverlayConfig, create_overlay


class BlueTeamScanner:
    """
    Thread yang scan tim biru setiap 3 detik sampai semua 5 hero ketemu.
    """

    def __init__(
        self,
        video_path: str,
        scan_interval: float = 3.0,
        max_scans: int = 20,
    ):
        self.video_path = video_path
        self.scan_interval = scan_interval
        self.max_scans = max_scans

        self.reader = FrameReader(video_path)
        self.pipeline = Pipeline(self.reader)

        # Register blue team detector
        self.detector = BlueTeamDetector(confidence_threshold=0.35)
        self.pipeline.register("blue_team", self.detector.detect)

        # Results
        self.detected_heroes: list[dict] = []
        self.all_found = False
        self.scan_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Callback untuk UI update
        self.on_update = None  # callable(detected_heroes, scan_count, all_found)

    def start(self):
        """Mulai scanning di background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Hentikan scanning."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _scan_loop(self):
        """Loop scanning setiap scan_interval detik."""
        fps = self.reader.fps
        frames_per_scan = int(fps * self.scan_interval)

        for frame_idx in range(0, self.reader.total_frames, frames_per_scan):
            if self._stop_event.is_set():
                break
            if self.scan_count >= self.max_scans:
                break

            # Process frame
            frame = self.reader.read(frame_idx)
            if frame is None:
                break

            timestamp = frame_idx / fps
            result = self.pipeline.process_frame(frame, frame_idx, timestamp)

            # Extract blue team detection
            blue_det = next((d for d in result.detections if d.region_path == "blue_team"), None)
            if blue_det and blue_det.value:
                self._process_detection(blue_det.value)

            self.scan_count += 1

            # Callback update
            if self.on_update:
                self.on_update(
                    self.detected_heroes,
                    self.scan_count,
                    self.all_found
                )

            if self.all_found:
                print(f"\n✅ All 5 heroes found after {self.scan_count} scans!")
                break

            # Wait for next scan
            time.sleep(self.scan_interval)

        if not self.all_found:
            print(f"\n⚠️ Scan limit reached ({self.max_scans}). Found {len(self.detected_heroes)}/5 heroes.")

    def _process_detection(self, detection_value: dict):
        """Process detection result, update hero list."""
        heroes = detection_value.get("heroes", [])
        detected_count = detection_value.get("detected_count", 0)
        all_detected = detection_value.get("all_detected", False)

        # Update detected heroes
        new_heroes = []
        for h in heroes:
            if h["hero_name"]:
                new_heroes.append({
                    "name": h["hero_name"],
                    "slot": h["slot"],
                    "confidence": h["confidence"],
                })

        # Merge with existing (keep highest confidence per slot)
        for new_h in new_heroes:
            existing = next((h for h in self.detected_heroes if h["slot"] == new_h["slot"]), None)
            if not existing or new_h["confidence"] > existing["confidence"]:
                if existing:
                    self.detected_heroes.remove(existing)
                self.detected_heroes.append(new_h)

        self.all_found = all_detected and len(self.detected_heroes) == 5


def demo_overlay():
    """Demo dengan overlay display."""
    video_path = "path/to/your/replay.mp4"  # Ganti dengan path video kamu

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Usage: python demo_blue_team.py <video_path>")
        return

    # Setup scanner
    scanner = BlueTeamScanner(video_path, scan_interval=3.0, max_scans=20)

    # Setup overlay
    overlay_config = OverlayConfig(
        x=50, y=50,
        width=400,
        bg_color=(0, 0, 0, 200),
        show_team_label=True,
    )
    overlay = TeamRosterOverlay(overlay_config)

    # Callback untuk update overlay data
    def on_update(detected_heroes, scan_count, all_found):
        # Update overlay data
        overlay.set_blue_team(detected_heroes)
        overlay.set_red_team([])  # Red team not scanned in this demo

        # Print status
        names = [h["name"] for h in detected_heroes]
        print(f"Scan #{scan_count}: {len(detected_heroes)}/5 - {', '.join(names) or 'none'}")

    scanner.on_update = on_update

    # Start scanning
    print("🔍 Starting blue team scan (every 3 seconds)...")
    scanner.start()

    # Display loop
    reader = FrameReader(video_path)
    frame_idx = 0

    try:
        while frame_idx < reader.total_frames:
            frame = reader.read(frame_idx)
            if frame is None:
                break

            # Draw overlay
            frame = overlay.draw(frame)

            # Add scan status text
            status = f"Scanning... {scanner.scan_count}/{scanner.max_scans}"
            if scanner.all_found:
                status = "✅ All heroes found!"
            cv2.putText(frame, status, (50, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Blue Team Scanner", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space = pause/resume
                cv2.waitKey(0)

            frame_idx += 1

    finally:
        scanner.stop()
        cv2.destroyAllWindows()
        reader.release()

    # Final result
    print("\n=== FINAL RESULT ===")
    for h in scanner.detected_heroes:
        print(f"  Slot {h['slot']}: {h['name']} (conf: {h['confidence']:.2f})")


def demo_headless():
    """Headless mode - just scan and print results."""
    video_path = "path/to/your/replay.mp4"

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    scanner = BlueTeamScanner(video_path, scan_interval=3.0, max_scans=20)

    def on_update(detected_heroes, scan_count, all_found):
        names = [h["name"] for h in detected_heroes]
        print(f"Scan #{scan_count}: {len(detected_heroes)}/5 - {', '.join(names) or 'none'}")

    scanner.on_update = on_update
    scanner.start()
    scanner._thread.join()  # Wait for completion

    print("\n=== FINAL RESULT ===")
    for h in scanner.detected_heroes:
        print(f"  Slot {h['slot']}: {h['name']} (conf: {h['confidence']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow passing video path as argument
        import sys
        sys.argv[0] = sys.argv[0]  # keep script name
        if sys.argv[1] == "--headless":
            demo_headless()
        else:
            # Override video path
            print(f"Video path: {sys.argv[1]}")
            # Update the demo function to use this path
    else:
        print("Usage:")
        print("  python demo_blue_team.py <video_path>        # With overlay UI")
        print("  python demo_blue_team.py <video_path> --headless  # Headless scan only")
        print("\nEdit the script to set your video path, or pass as argument.")
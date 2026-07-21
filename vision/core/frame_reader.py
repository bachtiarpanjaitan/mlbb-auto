"""
MLBB Vision — Frame Reader
Membaca video frame-by-frame dengan dukungan seeking dan caching.
"""

from __future__ import annotations

import os
from typing import Generator
import cv2
import numpy as np


class FrameReader:
    """Read frames from a video file with optional caching."""

    def __init__(self, video_path: str, cache_size: int = 30):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cache_size = cache_size
        self._cache_order: list[int] = []

    def read(self, frame_idx: int | None = None) -> np.ndarray | None:
        """
        Read frame at given index (0-based).
        If frame_idx is None, read next frame sequentially.
        """
        if frame_idx is not None:
            if frame_idx in self._frame_cache:
                self._update_cache_order(frame_idx)
                return self._frame_cache[frame_idx]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            return None

        if frame_idx is not None:
            self._cache_frame(frame_idx, frame)

        return frame

    def read_at_timestamp(self, seconds: float) -> np.ndarray | None:
        """Read frame at given timestamp (seconds from start)."""
        frame_idx = int(seconds * self.fps)
        return self.read(frame_idx)

    def frames(self, start: int = 0, end: int | None = None, step: int = 1) -> Generator[np.ndarray, None, None]:
        """Iterate over frames in a range."""
        if end is None:
            end = self.total_frames
        for i in range(start, end, step):
            frame = self.read(i)
            if frame is None:
                break
            yield frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Sequential stream of all frames (no seeking)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def seek(self, frame_idx: int):
        """Seek to specific frame (next read() will return that frame)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def _cache_frame(self, idx: int, frame: np.ndarray):
        """Cache frame with LRU eviction."""
        if len(self._cache_order) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._frame_cache.pop(oldest, None)
        self._frame_cache[idx] = frame
        self._update_cache_order(idx)

    def _update_cache_order(self, idx: int):
        if idx in self._cache_order:
            self._cache_order.remove(idx)
        self._cache_order.append(idx)

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    def release(self):
        """Release video capture resources."""
        self.cap.release()
        self._frame_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()

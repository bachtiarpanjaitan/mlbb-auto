"""
MLBB Vision — Pipeline
Pipeline utama yang mengoordinasikan semua detector.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import layout
from .frame_reader import FrameReader
from .cropper import crop_region

logger = logging.getLogger("mlbb.vision.pipeline")


@dataclass
class DetectionResult:
    """Hasil deteksi dari satu detector pada satu region."""
    region_path: str
    value: Any
    confidence: float = 1.0
    raw: Any = None
    elapsed_ms: float = 0.0


@dataclass
class FrameResult:
    """Hasil deteksi untuk satu frame."""
    frame_idx: int
    timestamp: float
    detections: list[DetectionResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_idx,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "detections": [
                {
                    "region": d.region_path,
                    "value": d.value,
                    "confidence": d.confidence,
                    "elapsed_ms": d.elapsed_ms,
                }
                for d in self.detections
            ],
        }


class Pipeline:
    """
    Vision processing pipeline.

    Mendaftarkan detector function per region, lalu menjalankannya
    pada setiap frame untuk mengekstrak game state.
    """

    def __init__(self, reader: FrameReader):
        self.reader = reader
        self._detectors: dict[str, Callable] = {}

    def register(self, region_path: str, detector_fn: Callable):
        """
        Daftarkan detector untuk region path tertentu.

        Args:
            region_path: dot-path ke region di layout, e.g. "hero_panel.timer"
            detector_fn: callable(cropped_frame) -> value
        """
        self._detectors[region_path] = detector_fn

    def register_many(self, mapping: dict[str, Callable]):
        """Register multiple detectors at once."""
        self._detectors.update(mapping)

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameResult:
        """Process a single frame through all registered detectors."""
        result = FrameResult(frame_idx=frame_idx, timestamp=timestamp)
        t0 = time.perf_counter()

        for region_path, detector_fn in self._detectors.items():
            t1 = time.perf_counter()
            try:
                keys = region_path.split(".")
                region_img = crop_region(frame, *keys)
                if region_img is None or region_img.size == 0:
                    logger.debug("Empty region: %s", region_path)
                    continue

                value = detector_fn(region_img)
                dr = DetectionResult(
                    region_path=region_path,
                    value=value,
                    elapsed_ms=(time.perf_counter() - t1) * 1000,
                )
                result.detections.append(dr)

            except Exception as e:
                logger.warning("Detector error on %s: %s", region_path, e)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    def run(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
        progress_cb: Callable | None = None,
        yield_every: int = 1,
    ):
        """
        Run pipeline over a range of frames.

        Args:
            start_frame: Frame index to start from.
            end_frame: Frame index to end at (exclusive). None = all.
            step: Process every N frames.
            progress_cb: Optional callback(frame_idx, total, result).
            yield_every: Yield result every N frames (for streaming).

        Yields:
            FrameResult for each processed frame matching yield_every.
        """
        if end_frame is None:
            end_frame = self.reader.total_frames

        total = (end_frame - start_frame) // step
        for i, f_idx in enumerate(range(start_frame, end_frame, step)):
            frame = self.reader.read(f_idx)
            if frame is None:
                break

            timestamp = f_idx / self.reader.fps
            result = self.process_frame(frame, f_idx, timestamp)

            if progress_cb:
                progress_cb(f_idx, total, result)

            if i % yield_every == 0:
                yield result

    def run_single(self, frame_idx: int) -> FrameResult:
        """Process a single frame."""
        frame = self.reader.read(frame_idx)
        if frame is None:
            raise ValueError(f"Cannot read frame {frame_idx}")
        timestamp = frame_idx / self.reader.fps
        return self.process_frame(frame, frame_idx, timestamp)

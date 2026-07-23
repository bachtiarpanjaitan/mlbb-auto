"""
Region Mapper — Memetakan posisi minimap ke region dari regions.json.

regions.json berisi polygon region (river, top_lane, jungle camps, dll)
yang dibuat via tools/region_editor.py. Koordinat di regions.json
adalah dalam skala MINIMAP ASLI (pixel), bukan game coordinates.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core import layout as layout_mod

logger = logging.getLogger("mlbb.vision.region_mapper")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REGIONS_JSON = _PROJECT_ROOT / "assets" / "databases" / "regions.json"


@dataclass
class Region:
    """Satu region dari regions.json."""
    id: str
    name: str
    points: list[list[int]]  # [[x, y], ...] dalam pixel minimap
    _contour: np.ndarray | None = None

    def __post_init__(self):
        if self.points:
            self._contour = np.array(self.points, dtype=np.int32)

    def contains(self, x: float, y: float) -> bool:
        """Cek apakah titik (x, y) berada di dalam region ini."""
        if self._contour is None or len(self._contour) < 3:
            return False
        # cv2.pointPolygonTest: >0 inside, =0 on edge, <0 outside
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0


class RegionMapper:
    """
    Mapper untuk menentukan region dari koordinat minimap.

    Load regions.json (dibuat oleh region_editor.py) dan menyediakan
    metode untuk query region di posisi tertentu.
    """

    def __init__(self, regions_path: str | Path | None = None):
        self.regions_path = Path(regions_path) if regions_path else _REGIONS_JSON
        self._regions: list[Region] = []
        self._load_regions()

    def _load_regions(self):
        """Load regions dari JSON file."""
        if not self.regions_path.exists():
            logger.warning("Regions file not found: %s", self.regions_path)
            return

        try:
            with open(self.regions_path, "r") as f:
                data = json.load(f)
            self._regions = [Region(**r) for r in data]
            logger.info("Loaded %d regions from %s", len(self._regions), self.regions_path)
        except Exception as e:
            logger.error("Failed to load regions: %s", e)

    def get_region_at(self, x: float, y: float, minimap_w: int = 350, minimap_h: int = 340) -> str:
        """
        Dapatkan ID region di posisi (x, y).

        Args:
            x: Koordinat x dalam pixel minimap atau normalized (0.0-1.0)
            y: Koordinat y dalam pixel minimap atau normalized (0.0-1.0)

        Returns:
            Region ID (id) atau string kosong jika tidak ada region.
        """
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x = x * minimap_w
            y = y * minimap_h
        for region in self._regions:
            if region.contains(x, y):
                return region.id
        return ""

    def get_region_name_at(self, x: float, y: float, minimap_w: int = 350, minimap_h: int = 340) -> str:
        """
        Dapatkan display name region di posisi (x, y).

        Returns:
            Region display name atau string kosong.
        """
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x = x * minimap_w
            y = y * minimap_h
        for region in self._regions:
            if region.contains(x, y):
                return region.name
        return ""

    def get_all_regions(self) -> list[Region]:
        """Return semua region yang terload."""
        return list(self._regions)

    def has_regions(self) -> bool:
        """Cek apakah ada region yang terload."""
        return len(self._regions) > 0


# Singleton instance
_region_mapper: RegionMapper | None = None


def get_region_mapper(regions_path: str | Path | None = None) -> RegionMapper:
    """Get singleton RegionMapper instance."""
    global _region_mapper
    if _region_mapper is None:
        _region_mapper = RegionMapper(regions_path)
    return _region_mapper


def reset_region_mapper():
    """Reset singleton (untuk testing/reload)."""
    global _region_mapper
    _region_mapper = None
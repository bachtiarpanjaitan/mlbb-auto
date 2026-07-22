"""
Coordinate Mapper — Mengkonversi posisi minimap ke koordinat game map.

Minimap pada 2400×1080 replay (layout.yaml):
  - Bbox: [80, 0, 350, 340] (350×340 pixel, biasanya square ~340)
  - Mapping: minimap normalized (0-1) → game coordinates (10000 × 10000 units)

Note: Koordinat minimap dicek dari layout.yaml secara otomatis via from_layout().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import MAP_SIZE, ALL_LANDMARKS, nearest_landmark, get_lane_for_position
from ..core import layout as layout_mod


@dataclass
class MapPosition:
    """Posisi terkoordinasi di peta."""
    x: float           # game units (0 — MAP_SIZE)
    y: float           # game units (0 — MAP_SIZE)
    norm_x: float = 0.0  # normalized 0-1 (relative to minimap)
    norm_y: float = 0.0
    lane: str = "mid"  # top, mid, bottom
    nearest_landmark: str = ""
    distance_to_landmark: float = 0.0


class CoordinateMapper:
    """
    Mapper minimap → game coordinates.

    Args:
        minimap_size: Size of minimap side in pixels (used for scale calc).
                       Default from layout.yaml atau fallback 340.
        map_size: Game map size in units (default 10000).
    """

    def __init__(
        self,
        minimap_size: int | None = None,
        map_size: int = MAP_SIZE,
    ):
        # Ambil dari layout.yaml jika tidak specified
        if minimap_size is None:
            mm_bbox = layout_mod.bbox("map")
            if mm_bbox:
                _, _, mm_w, mm_h = mm_bbox
                self.minimap_size = max(mm_w, mm_h)
            else:
                self.minimap_size = 340  # fallback
        else:
            self.minimap_size = minimap_size

        self.map_size = map_size

        # Scale factor: game units per normalized unit
        # Normalized 0-1 maps to 0-MAP_SIZE
        self.scale = map_size

    @classmethod
    def from_layout(cls) -> CoordinateMapper:
        """Create mapper from layout.yaml minimap bbox."""
        mm_bbox = layout_mod.bbox("map")
        if mm_bbox:
            _, _, mm_w, mm_h = mm_bbox
            mm_size = max(mm_w, mm_h)
        else:
            mm_size = 340
        return cls(minimap_size=mm_size)

    def minimap_to_game(self, norm_x: float, norm_y: float) -> MapPosition:
        """
        Convert normalized minimap position (0-1) to game coordinates.

        Args:
            norm_x: Normalized x on minimap (0=left, 1=right).
            norm_y: Normalized y on minimap (0=top, 1=bottom).

        Returns:
            MapPosition with game coordinates.
        """
        # Clamp
        nx = max(0.0, min(1.0, norm_x))
        ny = max(0.0, min(1.0, norm_y))

        # Linear mapping: normalized 0-1 → game 0-MAP_SIZE
        # MLBB: top-left minimap = top-left game map (origin)
        game_x = nx * self.map_size
        game_y = ny * self.map_size

        # Find nearest landmark
        lm = nearest_landmark(game_x, game_y)
        lm_name = lm.key if lm else ""
        lm_dist = ((lm.x - game_x) ** 2 + (lm.y - game_y) ** 2) ** 0.5 if lm else 0.0

        lane = get_lane_for_position(game_x, game_y)

        return MapPosition(
            x=round(game_x, 1),
            y=round(game_y, 1),
            norm_x=round(nx, 4),
            norm_y=round(ny, 4),
            lane=lane,
            nearest_landmark=lm_name,
            distance_to_landmark=round(lm_dist, 1),
        )

    def minimap_pixel_to_game(self, px: int, py: int, mm_w: int | None = None, mm_h: int | None = None) -> MapPosition:
        """
        Convert minimap pixel coordinates to game coordinates.

        Args:
            px: Pixel x within minimap region.
            py: Pixel y within minimap region.
            mm_w: Minimap width in pixels (default: minimap_size).
            mm_h: Minimap height in pixels (default: minimap_size).

        Returns:
            MapPosition.
        """
        w = mm_w or self.minimap_size
        h = mm_h or self.minimap_size
        norm_x = px / max(1, w)
        norm_y = py / max(1, h)
        return self.minimap_to_game(norm_x, norm_y)

    def game_to_minimap(self, game_x: float, game_y: float) -> tuple[float, float]:
        """
        Convert game coordinates back to normalized minimap position.

        Returns:
            (norm_x, norm_y) in 0-1 range.
        """
        nx = max(0.0, min(1.0, game_x / self.map_size))
        ny = max(0.0, min(1.0, game_y / self.map_size))
        return (nx, ny)

    def distance(self, a: MapPosition, b: MapPosition) -> float:
        """Distance between two map positions in game units."""
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def get_zone(self, pos: MapPosition) -> str:
        """
        Determine zone (blue jungle, red jungle, river, lane).

        Simplified heuristic based on river diagonal.
        """
        # River runs roughly from (0,0) to (10000,10000)
        river_width = 1200
        dist_to_river = abs(pos.x - pos.y) / 2 ** 0.5

        if dist_to_river < river_width:
            return "river"

        if pos.x + pos.y < MAP_SIZE:
            return "blue_jungle"
        else:
            return "red_jungle"

    def is_in_base(self, pos: MapPosition, team: str = "blue") -> bool:
        """Check if position is near team base."""
        base = ALL_LANDMARKS.get(f"base_{team}")
        if not base:
            return False
        dist = ((pos.x - base.x) ** 2 + (pos.y - base.y) ** 2) ** 0.5
        return dist < 2000

    def is_in_lane(self, pos: MapPosition, lane: str = "mid") -> bool:
        """Check if position is in a specific lane area."""
        return pos.lane == lane

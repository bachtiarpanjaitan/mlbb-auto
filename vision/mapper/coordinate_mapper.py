"""
Coordinate Mapper — Mengkonversi posisi minimap ke koordinat game map.

Minimap pada 2400×1080 replay:
  - Letak: [2130, 5, 265, 265] (265×265 pixel)
  - Maping: minimap pixels → game coordinates (10000 × 10000 units)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import MAP_SIZE, ALL_LANDMARKS, nearest_landmark, get_lane_for_position


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
        minimap_size: Size of minimap in pixels (width=height since square).
        map_size: Game map size in units.
        offset_x: X offset of minimap in frame.
        offset_y: Y offset of minimap in frame.
    """

    def __init__(
        self,
        minimap_size: int = 265,
        map_size: int = MAP_SIZE,
        offset_x: int = 2130,
        offset_y: int = 5,
    ):
        self.minimap_size = minimap_size
        self.map_size = map_size
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Scale factor: game units per minimap pixel
        self.scale = map_size / minimap_size

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

        # Linear mapping
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

    def minimap_pixel_to_game(self, px: int, py: int) -> MapPosition:
        """
        Convert minimap pixel coordinates to game coordinates.

        Args:
            px: Pixel x within minimap (0 — minimap_size).
            py: Pixel y within minimap (0 — minimap_size).

        Returns:
            MapPosition.
        """
        norm_x = px / self.minimap_size if self.minimap_size > 0 else 0
        norm_y = py / self.minimap_size if self.minimap_size > 0 else 0
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

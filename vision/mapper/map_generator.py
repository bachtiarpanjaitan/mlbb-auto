"""
Map Generator — Menghasilkan maps.json dari hasil tracking.

maps.json adalah output final dari vision engine, siap dikonsumsi
oleh AI untuk analisis gameplay.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from .coordinate_mapper import CoordinateMapper, MapPosition
from .landmarks import ALL_LANDMARKS, MAP_SIZE
from ..trackers.hero_tracker import HeroTracker, TrackedHero
from ..trackers.object_tracker import ObjectTracker, GameEvent


@dataclass
class HeroTrajectory:
    """Trajectory satu hero dalam match."""
    name: str
    team: str
    positions: list[dict] = field(default_factory=list)  # [{timestamp, x, y, lane}]
    level_progression: list[dict] = field(default_factory=list)
    gold_progression: list[dict] = field(default_factory=list)
    items_over_time: list[dict] = field(default_factory=list)
    total_distance: float = 0.0
    kills: int = 0
    deaths: int = 0
    assists: int = 0


@dataclass
class MatchMap:
    """Complete match map data."""
    metadata: dict = field(default_factory=dict)
    heroes: list[HeroTrajectory] = field(default_factory=list)
    objectives: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    heatmap: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


class MapGenerator:
    """
    Generator maps.json.

    Menggabungkan data dari HeroTracker + ObjectTracker + CoordinateMapper
    menjadi satu file maps.json terstruktur.
    """

    def __init__(
        self,
        hero_tracker: HeroTracker,
        object_tracker: ObjectTracker,
        mapper: CoordinateMapper | None = None,
    ):
        self.hero_tracker = hero_tracker
        self.object_tracker = object_tracker
        self.mapper = mapper or CoordinateMapper()
        self._match_data: list[dict] = []  # raw per-frame snapshots

    def add_frame_snapshot(self, frame_idx: int, timestamp: float, game_state: dict):
        """
        Record raw snapshot from one frame.

        Args:
            frame_idx: Frame number.
            timestamp: Timestamp in seconds.
            game_state: Dict dari GameState (raw field).
        """
        snapshot = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 2),
            "game_time": game_state.get("match_time"),
            "score": game_state.get("team_score"),
            "heroes": [],
        }

        # Process hero positions
        for hero_name, hero in self.hero_tracker.heroes.items():
            pos = None
            if hero.latest and hero.latest.position:
                mp = self.mapper.minimap_to_game(*hero.latest.position)
                pos = {"x": mp.x, "y": mp.y, "lane": mp.lane, "zone": self.mapper.get_zone(mp)}

            hero_entry = {
                "name": hero.name,
                "team": hero.team,
                "position": pos,
                "level": hero.latest.level if hero.latest else None,
                "hp_pct": hero.latest.hp_pct if hero.latest else None,
                "mana_pct": hero.latest.mana_pct if hero.latest else None,
                "gold": hero.latest.gold if hero.latest else None,
                "kda": hero.latest.kda if hero.latest else None,
                "alive": hero.latest.alive if hero.latest else True,
            }
            snapshot["heroes"].append(hero_entry)

        self._match_data.append(snapshot)

    def generate(self, output_path: str | None = None) -> dict:
        """
        Generate maps.json content from all recorded data.

        Args:
            output_path: Optional path untuk menyimpan file.

        Returns:
            Dict siap di-serialize ke JSON.
        """
        match_map = MatchMap()

        # ── Metadata ──────────────────────────────────────────────
        match_map.metadata = {
            "map_size": MAP_SIZE,
            "total_frames": len(self._match_data),
            "duration_seconds": self._match_data[-1]["timestamp"] if self._match_data else 0,
            "total_heroes": len(self.hero_tracker.heroes),
            "generated_at": datetime.now().isoformat(),
        }

        # ── Hero Trajectories ─────────────────────────────────────
        for hero_name, tracked in self.hero_tracker.heroes.items():
            traj = HeroTrajectory(
                name=hero_name,
                team=tracked.team,
            )

            prev_x, prev_y = None, None
            for snapshot in tracked.history:
                pos = None
                if snapshot.position:
                    mp = self.mapper.minimap_to_game(*snapshot.position)
                    pos = {
                        "timestamp": snapshot.timestamp,
                        "frame": snapshot.frame,
                        "x": mp.x,
                        "y": mp.y,
                        "lane": mp.lane,
                        "zone": self.mapper.get_zone(mp),
                    }
                    traj.positions.append(pos)

                    # Distance
                    if prev_x is not None and prev_y is not None:
                        dx = mp.x - prev_x
                        dy = mp.y - prev_y
                        traj.total_distance += (dx ** 2 + dy ** 2) ** 0.5
                    prev_x, prev_y = mp.x, mp.y

                if snapshot.level is not None:
                    traj.level_progression.append({
                        "timestamp": snapshot.timestamp,
                        "frame": snapshot.frame,
                        "level": snapshot.level,
                    })

                if snapshot.gold is not None:
                    traj.gold_progression.append({
                        "timestamp": snapshot.timestamp,
                        "frame": snapshot.frame,
                        "gold": snapshot.gold,
                    })

                if snapshot.items and snapshot.items != traj.items_over_time[-1]["items"] if traj.items_over_time else []:
                    traj.items_over_time.append({
                        "timestamp": snapshot.timestamp,
                        "frame": snapshot.frame,
                        "items": list(snapshot.items),
                    })

            # Parse KDA from latest
            if tracked.latest and tracked.latest.kda:
                try:
                    parts = str(tracked.latest.kda).split("/")
                    traj.kills = int(parts[0]) if len(parts) > 0 else 0
                    traj.deaths = int(parts[1]) if len(parts) > 1 else 0
                    traj.assists = int(parts[2]) if len(parts) > 2 else 0
                except (ValueError, IndexError):
                    pass

            match_map.heroes.append(traj)

        # ── Objectives ────────────────────────────────────────────
        for obj_name, tracked in self.object_tracker.objects.items():
            for state in tracked.history:
                match_map.objectives.append({
                    "object": obj_name,
                    "timestamp": state.timestamp,
                    "frame": state.frame,
                    "status": state.status,
                    "value": str(state.value) if state.value else None,
                })

        # ── Events ────────────────────────────────────────────────
        for ev in self.object_tracker.events:
            match_map.events.append({
                "type": ev.event_type,
                "timestamp": ev.timestamp,
                "frame": ev.frame,
                "detail": ev.detail,
                "heroes": ev.related_heroes,
            })

        # ── Heatmap ───────────────────────────────────────────────
        match_map.heatmap = self._generate_heatmap()

        # ── Stats ─────────────────────────────────────────────────
        match_map.stats = self._generate_stats()

        # ── Serialize ─────────────────────────────────────────────
        result = asdict(match_map)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✅ Maps data saved to {output_path}")

        return result

    def _generate_heatmap(self) -> dict:
        """Generate heatmap zones berdasarkan posisi hero."""
        grid_size = 20  # 20×20 grid
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

        for snapshot in self._match_data:
            for hero_entry in snapshot.get("heroes", []):
                pos = hero_entry.get("position")
                if pos and "x" in pos and "y" in pos:
                    gx = min(grid_size - 1, int(pos["x"] / MAP_SIZE * grid_size))
                    gy = min(grid_size - 1, int(pos["y"] / MAP_SIZE * grid_size))
                    grid[gy][gx] += 1

        # Normalize 0-100
        max_val = max(max(row) for row in grid) if any(any(c for c in row) for row in grid) else 1
        normalized = [[round(c / max_val * 100, 1) for c in row] for row in grid]

        return {
            "grid_size": grid_size,
            "map_size": MAP_SIZE,
            "grid": normalized,
        }

    def _generate_stats(self) -> dict:
        """Generate match statistics."""
        hero_stats = {}
        total_distance = 0

        for traj in self._match_data[-1].get("heroes", []) if self._match_data else []:
            name = traj.get("name", "unknown")
            hero_stats[name] = {
                "team": traj.get("team", "unknown"),
                "last_known": {
                    "position": traj.get("position"),
                    "level": traj.get("level"),
                    "gold": traj.get("gold"),
                    "kda": traj.get("kda"),
                    "alive": traj.get("alive", True),
                },
            }

        for h in self.hero_tracker.heroes.values():
            traj_inst = [t for t in self._match_data[-1].get("heroes", []) if t.get("name") == h.name] if self._match_data else []
        return {
            "hero_counts": {
                "blue": len(self.hero_tracker.get_team_heroes("blue")),
                "red": len(self.hero_tracker.get_team_heroes("red")),
            },
            "events_count": len(self.object_tracker.events),
            "snapshots_count": len(self._match_data),
        }


def generate_maps_json(
    hero_tracker: HeroTracker,
    object_tracker: ObjectTracker,
    output_path: str = "maps.json",
    mapper: CoordinateMapper | None = None,
) -> dict:
    """
    Convenience function: generate maps.json from trackers.

    Args:
        hero_tracker: Populated HeroTracker instance.
        object_tracker: Populated ObjectTracker instance.
        output_path: Path to save maps.json.
        mapper: Optional CoordinateMapper.

    Returns:
        maps data dict.
    """
    generator = MapGenerator(hero_tracker, object_tracker, mapper)
    return generator.generate(output_path)

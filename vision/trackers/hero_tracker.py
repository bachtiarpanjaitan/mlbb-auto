"""
Hero Tracker — Melacak hero secara konsisten antar frame.

Menjaga identitas hero (nama, team) dan riwayat (posisi, HP, gold, level)
dari frame ke frame dengan korelasi posisi dan appearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class HeroSnapshot:
    """State hero pada satu frame."""
    frame: int
    timestamp: float
    hp_pct: float | None = None
    mana_pct: float | None = None
    level: int | None = None
    gold: int | None = None
    kda: str | None = None
    position: tuple[float, float] | None = None  # (x, y) normalized 0-1
    items: list[str] = field(default_factory=list)
    alive: bool = True


@dataclass
class TrackedHero:
    """Data tracked untuk satu hero."""
    name: str
    team: str  # "blue" | "red"
    first_seen: int = 0
    last_seen: int = 0
    snapshot_count: int = 0
    latest: HeroSnapshot | None = None
    history: list[HeroSnapshot] = field(default_factory=list)

    @property
    def last_known_position(self) -> tuple[float, float] | None:
        return self.latest.position if self.latest else None

    @property
    def current_level(self) -> int | None:
        return self.latest.level if self.latest else None


class HeroTracker:
    """
    Melacak semua hero dalam match.
    Mengkorelasikan deteksi dengan nama hero dan posisi minimap.
    """

    def __init__(self, max_history: int = 1000):
        self.heroes: dict[str, TrackedHero] = {}  # name -> TrackedHero
        self.max_history = max_history
        self._next_id = 0

    def update(
        self,
        detections: list[dict],
        frame_idx: int,
        timestamp: float,
    ) -> dict[str, TrackedHero]:
        """
        Update tracked heroes dengan deteksi terbaru.

        Args:
            detections: List deteksi dari hero detector + pipeline.
                        Setiap dict harus punya 'name' atau 'key'.
            frame_idx: Frame number saat ini.
            timestamp: Timestamp frame.

        Returns:
            Dict {hero_name: TrackedHero}.
        """
        for det in detections:
            name = det.get("name") or det.get("label") or det.get("key")
            if not name:
                continue

            if name not in self.heroes:
                self.heroes[name] = TrackedHero(
                    name=name,
                    team=det.get("team", "unknown"),
                    first_seen=frame_idx,
                )

            hero = self.heroes[name]
            hero.last_seen = frame_idx
            hero.snapshot_count += 1
            if det.get("team"):
                hero.team = det["team"]

            snapshot = HeroSnapshot(
                frame=frame_idx,
                timestamp=timestamp,
                hp_pct=det.get("hp_pct"),
                mana_pct=det.get("mana_pct"),
                level=det.get("level"),
                gold=det.get("gold"),
                kda=det.get("kda"),
                position=det.get("position_minimap"),
                items=det.get("items", []),
                alive=det.get("alive", True),
            )
            hero.latest = snapshot
            hero.history.append(snapshot)

            # Trim history
            if len(hero.history) > self.max_history:
                hero.history = hero.history[-self.max_history:]

        return self.heroes

    def update_from_state(self, state, frame_idx: int, timestamp: float):
        """
        Update dari GameState object.

        Args:
            state: GameState dari StateBuilder.
            frame_idx: Frame number.
            timestamp: Timestamp frame.
        """
        detections = []
        if state.selected_hero:
            det = {"name": state.selected_hero.name, "team": "blue"}
            if state.selected_hero.level is not None:
                det["level"] = state.selected_hero.level
            if state.selected_hero.hp_pct is not None:
                det["hp_pct"] = state.selected_hero.hp_pct
            if state.selected_hero.mana_pct is not None:
                det["mana_pct"] = state.selected_hero.mana_pct
            if state.selected_hero.gold is not None:
                det["gold"] = state.selected_hero.gold
            if state.selected_hero.kda is not None:
                det["kda"] = state.selected_hero.kda
            if state.selected_hero.position_minimap:
                det["position_minimap"] = state.selected_hero.position_minimap
            if state.selected_hero.items:
                det["items"] = state.selected_hero.items
            detections.append(det)
        return self.update(detections, frame_idx, timestamp)

    def get_hero(self, name: str) -> TrackedHero | None:
        """Get tracked hero by name."""
        return self.heroes.get(name)

    def get_roster(self) -> list[TrackedHero]:
        """Return all tracked heroes."""
        return list(self.heroes.values())

    def get_alive_heroes(self) -> list[TrackedHero]:
        """Return heroes that were alive at last snapshot."""
        return [
            h for h in self.heroes.values()
            if h.latest and h.latest.alive
        ]

    def get_team_heroes(self, team: str) -> list[TrackedHero]:
        """Return heroes filtered by team."""
        return [h for h in self.heroes.values() if h.team == team]

    def get_gold_leader(self, team: str | None = None) -> TrackedHero | None:
        """Return hero with highest gold."""
        candidates = self.get_team_heroes(team) if team else self.heroes.values()
        with_gold = [
            (h, h.latest.gold) for h in candidates
            if h.latest and h.latest.gold is not None
        ]
        if not with_gold:
            return None
        return max(with_gold, key=lambda x: x[1])[0]

    def reset(self):
        """Clear all tracked data."""
        self.heroes.clear()

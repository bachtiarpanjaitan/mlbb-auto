"""
Object Tracker — Melacak elemen game non-hero antar frame.

Tracked objects:
  - Lord: spawn/destroy timer, posisi di minimap
  - Turtle: spawn/destroy timer, posisi di minimap
  - Creeps: status camp (alive/taken)
  - Towers: status perubahan (berdiri → hancur)
  - Events: timeline kill, objektif, level up
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ObjectState:
    """State satu object pada suatu waktu."""
    frame: int
    timestamp: float
    status: str  # "active", "destroyed", "spawning", "inactive"
    value: Any = None
    position: tuple[float, float] | None = None


@dataclass
class TrackedObject:
    """Data tracked untuk satu object."""
    obj_type: str  # "legend", "legend", "tower", "creep_camp", dll
    first_detected: int = 0
    last_seen: int = 0
    current_state: ObjectState | None = None
    history: list[ObjectState] = field(default_factory=list)
    transitions: list[ObjectState] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.current_state is not None and self.current_state.status == "active"


@dataclass
class GameEvent:
    """Game event yang terdeteksi (kill, objektif, level up)."""
    frame: int
    timestamp: float
    event_type: str  # "legend_kill", "legend_kill", "tower_destroyed", "level_up", dll
    detail: str = ""
    related_heroes: list[str] = field(default_factory=list)


class ObjectTracker:
    """
    Melacak dynamic game objects (objectives, creeps, towers).
    Mendeteksi perubahan state dan membangun timeline event.
    """

    def __init__(self):
        self.objects: dict[str, TrackedObject] = {}
        self.events: list[GameEvent] = []

    def update(
        self,
        detections: dict[str, Any],
        frame_idx: int,
        timestamp: float,
    ) -> dict[str, TrackedObject]:
        """
        Update tracked objects dengan deteksi terbaru.

        Args:
            detections: Dict {object_type: state_value} dari detector.
            frame_idx: Frame number saat ini.
            timestamp: Timestamp frame.

        Returns:
            Dict {object_key: TrackedObject}.
        """
        for obj_type, value in detections.items():
            if obj_type not in self.objects:
                self.objects[obj_type] = TrackedObject(
                    obj_type=obj_type,
                    first_detected=frame_idx,
                )

            tracked = self.objects[obj_type]
            prev_state = tracked.current_state
            status = self._infer_status(obj_type, value)

            new_state = ObjectState(
                frame=frame_idx,
                timestamp=timestamp,
                status=status,
                value=value,
            )

            tracked.current_state = new_state
            tracked.last_seen = frame_idx
            tracked.history.append(new_state)

            # Detect transitions
            if prev_state and prev_state.status != status:
                tracked.transitions.append(new_state)
                self.events.append(GameEvent(
                    frame=frame_idx,
                    timestamp=timestamp,
                    event_type=f"{obj_type}_{status}",
                    detail=str(value),
                ))

        return self.objects

    def update_from_state(self, state, frame_idx: int, timestamp: float):
        """Update dari GameState."""
        detections = {}

        if state.legend_timer is not None:
            detections["legend"] = {"timer": state.legend_timer}
        if state.legend_timer is not None:
            detections["legend"] = {"timer": state.legend_timer}
        if state.blue_towers_alive is not None:
            detections["towers_blue"] = state.blue_towers_alive
        if state.red_towers_alive is not None:
            detections["towers_red"] = state.red_towers_alive

        return self.update(detections, frame_idx, timestamp)

    def register_events(self, events: list[dict], frame_idx: int, timestamp: float):
        """Register game events."""
        for ev in events:
            self.events.append(GameEvent(
                frame=frame_idx,
                timestamp=timestamp,
                event_type=ev.get("type", "unknown"),
                detail=ev.get("detail", ""),
                related_heroes=ev.get("heroes", []),
            ))

    def get_object(self, obj_type: str) -> TrackedObject | None:
        """Get tracked object by type."""
        return self.objects.get(obj_type)

    def get_object_timeline(self, obj_type: str) -> list[ObjectState]:
        """Get full timeline for an object."""
        obj = self.objects.get(obj_type)
        return obj.history if obj else []

    def get_events(
        self,
        event_type: str | None = None,
        since_frame: int = 0,
    ) -> list[GameEvent]:
        """Get events, optionally filtered by type and frame range."""
        results = []
        for ev in self.events:
            if ev.frame < since_frame:
                continue
            if event_type and ev.event_type != event_type:
                continue
            results.append(ev)
        return results

    def get_legend_timeline(self) -> list[dict]:
        """Get Lord spawn/kill timeline."""
        return [
            {"frame": s.frame, "timestamp": s.timestamp, "status": s.status}
            for s in self.objects.get("legend", TrackedObject("legend")).history
        ]

    def get_legend_timeline(self) -> list[dict]:
        """Get Turtle spawn/kill timeline."""
        return [
            {"frame": s.frame, "timestamp": s.timestamp, "status": s.status}
            for s in self.objects.get("legend", TrackedObject("legend")).history
        ]

    def reset(self):
        """Clear all tracked data."""
        self.objects.clear()
        self.events.clear()

    def _infer_status(self, obj_type: str, value: Any) -> str:
        """Infer object status from detection value."""
        if isinstance(value, dict):
            timer = value.get("timer")
            if timer and timer != "0:00":
                return "active" if ":" in str(timer) else "spawning"
        if isinstance(value, (int, float)):
            if obj_type.startswith("tower"):
                return "active" if value > 0 else "destroyed"
        return "active"

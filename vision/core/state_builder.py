"""
MLBB Vision — State Builder
Menggabungkan hasil deteksi menjadi structured game state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .pipeline import DetectionResult, FrameResult


@dataclass
class HeroState:
    """State satu hero pada suatu momen."""
    name: str | None = None
    level: int | None = None
    hp_pct: float | None = None
    mana_pct: float | None = None
    hp_current: int | None = None
    hp_max: int | None = None
    mana_current: int | None = None
    mana_max: int | None = None
    gold: int | None = None
    kda: str | None = None
    skills_available: dict[str, bool] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)
    team: str | None = None
    position_minimap: tuple[float, float] | None = None


@dataclass
class GameState:
    """Full game state snapshot."""
    timestamp: float
    frame: int

    match_time: str | None = None
    team_score: tuple[int, int] | None = None
    gold_blue: int | None = None
    gold_red: int | None = None
    elapsed_game_sec: float | None = None

    selected_hero: HeroState | None = None
    blue_heroes: list[HeroState] = field(default_factory=list)
    red_heroes: list[HeroState] = field(default_factory=list)

    minimap: Any = None
    lord_timer: str | None = None
    turtle_timer: str | None = None

    blue_towers_alive: int | None = None
    red_towers_alive: int | None = None

    recent_events: list[dict] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class StateBuilder:
    """
    Menggabungkan hasil deteksi dari pipeline menjadi GameState terstruktur.

    Cara pakai:
        builder = StateBuilder()
        for result in pipeline.run(...):
            state = builder.build(result)
    """

    def __init__(self):
        self.prev_state: GameState | None = None
        self._events_buffer: list[dict] = []

    def build(self, frame_result: FrameResult) -> GameState:
        """Convert FrameResult -> GameState."""
        state = GameState(
            timestamp=frame_result.timestamp,
            frame=frame_result.frame_idx,
        )

        detections = {d.region_path: d for d in frame_result.detections}

        # ── Game Info ────────────────────────────────────────────────
        if timer := detections.get("top_bar.center_info.match_timer"):
            state.match_time = str(timer.value)

        if score := detections.get("top_bar.center_info.team_score"):
            try:
                parts = str(score.value).replace(" ", "").split("-")
                if len(parts) == 2:
                    state.team_score = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        if gb := detections.get("top_bar.center_info.gold_blue"):
            try:
                state.gold_blue = int(str(gb.value).replace(",", ""))
            except (ValueError, TypeError):
                pass

        if gr := detections.get("top_bar.center_info.gold_red"):
            try:
                state.gold_red = int(str(gr.value).replace(",", ""))
            except (ValueError, TypeError):
                pass

        # ── Selected Hero ───────────────────────────────────────────
        hero = HeroState(team="blue")

        if name := detections.get("hero_panel.hero_name"):
            hero.name = str(name.value).strip()

        if lvl := detections.get("hero_panel.level"):
            try:
                hero.level = int(str(lvl.value))
            except (ValueError, TypeError):
                pass

        if hp := detections.get("hero_panel.hp_bar"):
            if isinstance(hp.value, (int, float)):
                hero.hp_pct = float(hp.value)

        if mana := detections.get("hero_panel.mana_bar"):
            if isinstance(mana.value, (int, float)):
                hero.mana_pct = float(mana.value)

        if gold := detections.get("hero_panel.gold"):
            try:
                hero.gold = int(str(gold.value).replace(",", ""))
            except (ValueError, TypeError):
                pass

        if kda := detections.get("hero_panel.kda"):
            hero.kda = str(kda.value).strip()

        for skill_name in ("passive", "skill_1", "skill_2", "skill_3", "battle_spell"):
            key = f"hero_panel.skills.{skill_name}"
            if key in detections:
                hero.skills_available[skill_name] = bool(detections[key].value)

        for i in range(1, 7):
            key = f"hero_panel.items.item_{i}"
            if key in detections and detections[key].value:
                hero.items.append(str(detections[key].value))

        if any(v is not None for v in (hero.name, hero.level, hero.hp_pct)):
            state.selected_hero = hero

        # ── Objectives ──────────────────────────────────────────────
        if lt := detections.get("objective_timers.lord_timer"):
            state.lord_timer = str(lt.value)
        if tt := detections.get("objective_timers.turtle_timer"):
            state.turtle_timer = str(tt.value)

        # ── Towers ───────────────────────────────────────────────────
        if bt := detections.get("top_bar.blue_towers"):
            if isinstance(bt.value, int):
                state.blue_towers_alive = bt.value
        if rt := detections.get("top_bar.red_towers"):
            if isinstance(rt.value, int):
                state.red_towers_alive = rt.value

        # ── Events ───────────────────────────────────────────────────
        self._detect_events(detections, state)

        # ── Raw ──────────────────────────────────────────────────────
        state.raw = {k: d.value for k, d in detections.items()}

        self.prev_state = state
        return state

    def _detect_events(self, detections: dict[str, DetectionResult], state: GameState):
        """Detect new events by comparing with previous state."""
        prev = self.prev_state
        if prev is None:
            return

        events = []

        if state.selected_hero and prev.selected_hero:
            if (state.selected_hero.level or 0) > (prev.selected_hero.level or 0):
                events.append({
                    "type": "level_up",
                    "hero": state.selected_hero.name,
                    "level": state.selected_hero.level,
                    "timestamp": state.timestamp,
                })

            if state.selected_hero.kda != prev.selected_hero.kda:
                events.append({
                    "type": "kda_change",
                    "hero": state.selected_hero.name,
                    "kda": state.selected_hero.kda,
                    "timestamp": state.timestamp,
                })

        if events:
            state.recent_events = events
            self._events_buffer.extend(events)

    def get_events(self, clear: bool = True) -> list[dict]:
        """Get buffered events."""
        events = list(self._events_buffer)
        if clear:
            self._events_buffer.clear()
        return events

    def reset(self):
        """Reset state builder."""
        self.prev_state = None
        self._events_buffer.clear()


def state_to_flat_dict(state: GameState) -> dict[str, Any]:
    """Convert GameState to flat dict for JSON serialization."""
    d = asdict(state)
    d.pop("raw", None)
    return d

"""
Landmarks — Posisi-posisi penting di peta MLBB.

Map coordinate system:
  - Origin (0,0) = top-left of minimap / Northwest
  - X: left → right (0 → MAP_SIZE)
  - Y: top → bottom (0 → MAP_SIZE) — Blue base di bottom area
  - MAP_SIZE = 10000 game units (square)
"""

from __future__ import annotations

from dataclasses import dataclass


MAP_SIZE = 10000  # game units


@dataclass
class Landmark:
    """Posisi landmark di peta."""
    name: str
    key: str
    x: float  # game units
    y: float  # game units
    type: str = ""  # base, tower, buff, objective, lane


# ── Team Bases ────────────────────────────────────────────────────────────
BASES = {
    "blue": Landmark("Blue Base", "base_blue", 1850, 7800, "base"),
    "red":  Landmark("Red Base",  "base_red",  7800, 1850, "base"),
}

# ── Objectives ────────────────────────────────────────────────────────────
OBJECTIVES = {
    "lord":   Landmark("Lord Pit",   "lord",   5000, 5000, "objective"),
    "turtle": Landmark("Turtle Pit", "turtle", 5000, 3300, "objective"),
}

# ── Lanes ─────────────────────────────────────────────────────────────────
# MLBB has 3 lanes: Top, Mid, Bottom
# Each lane has 3 tier of towers + 1 base tower
LANES = {
    "top": {
        "name": "Top Lane (EXP)",
        "towers": {
            "blue_tier_1": Landmark("Blue Top T1", "tower_top_blue_1", 1100, 4400, "tower"),
            "blue_tier_2": Landmark("Blue Top T2", "tower_top_blue_2", 2000, 5300, "tower"),
            "blue_tier_3": Landmark("Blue Top T3", "tower_top_blue_3", 2600, 6000, "tower"),
            "red_tier_1":  Landmark("Red Top T1",  "tower_top_red_1",  4400, 1100, "tower"),
            "red_tier_2":  Landmark("Red Top T2",  "tower_top_red_2",  5300, 2000, "tower"),
            "red_tier_3":  Landmark("Red Top T3",  "tower_top_red_3",  6000, 2600, "tower"),
        },
    },
    "mid": {
        "name": "Mid Lane",
        "towers": {
            "blue_tier_1": Landmark("Blue Mid T1", "tower_mid_blue_1", 4100, 6400, "tower"),
            "blue_tier_2": Landmark("Blue Mid T2", "tower_mid_blue_2", 4800, 7100, "tower"),
            "blue_tier_3": Landmark("Blue Mid T3", "tower_mid_blue_3", 3300, 8700, "tower"),
            "red_tier_1":  Landmark("Red Mid T1",  "tower_mid_red_1",  6400, 4100, "tower"),
            "red_tier_2":  Landmark("Red Mid T2",  "tower_mid_red_2",  7100, 4800, "tower"),
            "red_tier_3":  Landmark("Red Mid T3",  "tower_mid_red_3",  8700, 3300, "tower"),
        },
    },
    "bottom": {
        "name": "Bottom Lane (Gold)",
        "towers": {
            "blue_tier_1": Landmark("Blue Bot T1", "tower_bot_blue_1", 5600, 2000, "tower"),
            "blue_tier_2": Landmark("Blue Bot T2", "tower_bot_blue_2", 6700, 2800, "tower"),
            "blue_tier_3": Landmark("Blue Bot T3", "tower_bot_blue_3", 7700, 3500, "tower"),
            "red_tier_1":  Landmark("Red Bot T1",  "tower_bot_red_1",  2000, 5600, "tower"),
            "red_tier_2":  Landmark("Red Bot T2",  "tower_bot_red_2",  2800, 6700, "tower"),
            "red_tier_3":  Landmark("Red Bot T3",  "tower_bot_red_3",  3500, 7700, "tower"),
        },
    },
}

# ── Creep Camps ───────────────────────────────────────────────────────────
CREEP_CAMPS = {
    "blue_warrior":     Landmark("Blue Warrior Camp",  "warrior_blue",     2750, 6750, "creep"),
    "blue_lizard":      Landmark("Blue Lizard Camp",   "lizard_blue",      3500, 6250, "creep"),
    "blue_golem":       Landmark("Blue Golem Camp",    "golem_blue",       3000, 7750, "creep"),
    "blue_crab_top":    Landmark("Blue Top Crab",      "crab_top_blue",    3000, 4000, "creep"),
    "blue_crab_bot":    Landmark("Blue Bot Crab",      "crab_bot_blue",    5500, 2500, "creep"),
    "blue_beetle":      Landmark("Blue Beetle Camp",   "beetle_blue",      4000, 3500, "creep"),
    "red_warrior":      Landmark("Red Warrior Camp",   "warrior_red",      6750, 2750, "creep"),
    "red_lizard":       Landmark("Red Lizard Camp",    "lizard_red",       6250, 3500, "creep"),
    "red_golem":        Landmark("Red Golem Camp",     "golem_red",        7750, 3000, "creep"),
    "red_crab_top":     Landmark("Red Top Crab",       "crab_top_red",     4000, 3000, "creep"),
    "red_crab_bot":     Landmark("Red Bot Crab",       "crab_bot_red",     2500, 5500, "creep"),
    "red_beetle":       Landmark("Red Beetle Camp",    "beetle_red",       3500, 4000, "creep"),
    "lithowanderer_top":   Landmark("Top Lithowanderer",   "litho_top",    2500, 4500, "creep"),
    "lithowanderer_bot":   Landmark("Bot Lithowanderer",   "litho_bot",    4500, 2500, "creep"),
}

# ── Buff Camps ────────────────────────────────────────────────────────────
BUFFS = {
    "blue_red_buff":  Landmark("Blue Red Buff",  "red_buff_blue",  3000, 7000, "buff"),
    "blue_blue_buff": Landmark("Blue Blue Buff", "blue_buff_blue", 4000, 8000, "buff"),
    "red_red_buff":   Landmark("Red Red Buff",   "red_buff_red",   7000, 3000, "buff"),
    "red_blue_buff":  Landmark("Red Blue Buff",  "blue_buff_red",  8000, 4000, "buff"),
}

# ── Combined ──────────────────────────────────────────────────────────────
ALL_LANDMARKS: dict[str, Landmark] = {}
for d in [BASES, OBJECTIVES, *[lane["towers"] for lane in LANES.values()],
          CREEP_CAMPS, BUFFS]:
    ALL_LANDMARKS.update(d)


def nearest_landmark(x: float, y: float, max_dist: float = 1000) -> Landmark | None:
    """Find nearest landmark to (x, y) within max_dist game units."""
    best = None
    best_dist = max_dist
    for lm in ALL_LANDMARKS.values():
        dist = ((lm.x - x) ** 2 + (lm.y - y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = lm
    return best


def get_lane_for_position(x: float, y: float) -> str:
    """Determine which lane a position is closest to."""
    lane_positions = {
        "top":    (3000, 3000),
        "mid":    (5000, 5000),
        "bottom": (7000, 7000),
    }
    best_lane = "mid"
    best_dist = float("inf")
    for lane, (lx, ly) in lane_positions.items():
        dist = ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_lane = lane
    return best_lane

"""
Game State → Parquet Exporter

Menyimpan game state dari debug_vision ke format Parquet untuk training AI.

Cara pakai:
    exporter = GameStateExporter("data/game_states")
    for frame in video:
        exporter.append(frame_idx, video_time, current_status)
    exporter.flush("match_alpha_1")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("mlbb.export.parquet")

# ── Optional dependencies ──
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pd = None
    pa = None
    pq = None


# ── Schema utama ────────────────────────────────────────────────────────
# Setiap row = 1 frame. Semua nested data di-flatten ke nama kolom dot-path.

FRAME_COLUMNS = [
    # Frame info
    "frame_idx",           # int — frame ke-n dalam video
    "video_time",          # float — detik dari awal video

    # Match info (dari top_bar / scoreboard)
    "match_time",          # str — timer in-game (MM:SS)
    "match_elapsed_sec",   # float — detik in-game
    "team_score_blue",     # int — kill blue team
    "team_score_red",      # int — kill red team
    "gold_blue",           # int — total gold blue
    "gold_red",            # int — total gold red

    # Selected hero (hero_panel)
    "hero_name",           # str — nama hero yang dipilih
    "level",               # int — level hero
    "hp_pct",              # float — 0.0 - 1.0
    "mana_pct",            # float — 0.0 - 1.0
    "gold",                # int — gold hero
    "kda",                 # str — KDA string
    "cdr_pct",             # float — cooldown reduction %

    # Items (6 slot)
    "item_1", "item_2", "item_3", "item_4", "item_5", "item_6",

    # Skills (5 slot)
    "skill_passive_ready",     # bool
    "skill_passive_cooldown",  # bool
    "skill_1_ready",
    "skill_1_cooldown",
    "skill_1_remaining_cd",
    "skill_2_ready",
    "skill_2_cooldown",
    "skill_2_remaining_cd",
    "skill_3_ready",
    "skill_3_cooldown",
    "skill_3_remaining_cd",
    "skill_4_ready",
    "skill_4_cooldown",
    "skill_4_remaining_cd",
    "battle_spell_ready",
    "battle_spell_cooldown",
    "battle_spell_remaining_cd",
    "battle_spell_name",       # str — nama spell

    # Blue team (5 hero dari scoreboard)
    "blue_team_complete",      # bool
]
# Blue/Red hero slots 1-5
for team in ("blue", "red"):
    for slot in range(1, 6):
        FRAME_COLUMNS += [
            f"{team}_hero_{slot}_name",
            f"{team}_hero_{slot}_hp_pct",
        ]

# Minimap hero positions (10 hero + jungle)
for team in ("blue", "red"):
    for slot in range(1, 6):
        FRAME_COLUMNS += [
            f"{team}_mm_{slot}_name",
            f"{team}_mm_{slot}_norm_x",
            f"{team}_mm_{slot}_norm_y",
            f"{team}_mm_{slot}_game_x",
            f"{team}_mm_{slot}_game_y",
            f"{team}_mm_{slot}_confidence",
            f"{team}_mm_{slot}_region",
            f"{team}_mm_{slot}_lane",
        ]

# Jungle objectives (dinamis — max 15)
for j in range(1, 16):
    FRAME_COLUMNS += [
        f"jungle_{j}_name",
        f"jungle_{j}_norm_x",
        f"jungle_{j}_norm_y",
        f"jungle_{j}_game_x",
        f"jungle_{j}_game_y",
    ]

# Jungle camp status — 15 fixed camps
JUNGLE_CAMP_IDS = [
    "legend_top", "legend_bot",
    "blue_buff_blue", "red_buff_blue", "blue_buff_red", "red_buff_red",
    "litho_center",
    "crab_top", "crab_bot",
    "golem_blue", "golem_red",
    "beetle_blue", "beetle_red",
    "lizard_blue", "lizard_red",
]

# Fixed camp positions (from manual label averages)
JUNGLE_CAMP_POSITIONS = {
    "legend_top": (113/350, 96/340),
    "legend_bot": (241/350, 241/340),
    "blue_buff_blue": (91/350, 175/340),
    "red_buff_blue": (187/350, 72/340),
    "blue_buff_red": (263/350, 162/340),
    "red_buff_red": (165/350, 264/340),
    "litho_center": (139/350, 132/340),
    "crab_top": (69/350, 88/340),
    "crab_bot": (282/350, 249/340),
    "golem_blue": (174/350, 89/340),
    "golem_red": (179/350, 247/340),
    "beetle_blue": (134/350, 61/340),
    "beetle_red": (218/350, 276/340),
    "lizard_blue": (62/350, 144/340),
    "lizard_red": (290/350, 192/340),
}

for cid in JUNGLE_CAMP_IDS:
    FRAME_COLUMNS += [
        f"jungle_{cid}_norm_x",
        f"jungle_{cid}_norm_y",
        f"jungle_{cid}_status",
    ]

# Towers
FRAME_COLUMNS += [
    "towers_alive",
]

# Objectives
FRAME_COLUMNS += [
    "legend_timer",
]


def _flatten_status(
    frame_idx: int,
    video_time: float,
    status: dict[str, Any],
) -> dict[str, Any]:
    """
    Flatten nested `current_status` dict → flat dict untuk 1 row parquet.

    Semua key None / missing → None (Parquet jadi null, training tinggal dropna).
    """
    row: dict[str, Any] = {col: None for col in FRAME_COLUMNS}
    row["frame_idx"] = frame_idx
    row["video_time"] = round(video_time, 3)

    # ── Selected hero ──
    row["hero_name"] = status.get("hero_name")
    row["level"] = status.get("level")
    row["hp_pct"] = status.get("hp_pct")
    row["mana_pct"] = status.get("mana_pct")
    row["gold"] = status.get("gold")
    row["kda"] = status.get("kda")
    row["cdr_pct"] = status.get("cdr_pct")

    # ── Items ──
    items = status.get("items") or []
    for i in range(6):
        row[f"item_{i + 1}"] = items[i] if i < len(items) else None

    # ── Skills ──
    skills = status.get("skills") or {}
    skill_map = {
        "passive": "passive",
        "skill_1": "1",
        "skill_2": "2",
        "skill_3": "3",
        "skill_4": "4",
        "battle_spell": "battle_spell",
    }
    for skey, slabel in skill_map.items():
        s = skills.get(skey) or {}
        s_col = slabel if slabel in ("passive", "battle_spell") else f"skill_{slabel}"
        row[f"{s_col}_ready"] = s.get("ready")
        row[f"{s_col}_cooldown"] = s.get("cooldown")
        row[f"{s_col}_remaining_cd"] = s.get("remaining_cd")
        if skey == "battle_spell":
            row["battle_spell_name"] = s.get("spell_name")

    # ── Team roster (hero slots 1-10, no blue/red split) ──
    all_heroes = []
    for team, team_key in [("blue", "blue_team_heroes"), ("red", "red_team_heroes")]:
        for h in (status.get(team_key) or []):
            h["team"] = team
            all_heroes.append(h)
    row["team_complete"] = len(all_heroes) == 10
    for slot, h in enumerate(all_heroes[:10], 1):
        row[f"hero_{slot}_name"] = h.get("name")
        row[f"hero_{slot}_team"] = h.get("team")
        row[f"hero_{slot}_hp_pct"] = h.get("hp_pct")

    # ── Match info ──
    row["match_time"] = status.get("match_time")
    row["match_elapsed_sec"] = status.get("match_elapsed_sec")
    score = status.get("team_score")
    if score and isinstance(score, (list, tuple)) and len(score) == 2:
        row["team_score_blue"], row["team_score_red"] = int(score[0]), int(score[1])
    row["gold_blue"] = status.get("gold_blue")
    row["gold_red"] = status.get("gold_red")

    # ── Minimap heroes (10 hero, no blue/red split) ──
    mm_heroes = status.get("minimap_heroes") or []
    all_mm = [h for h in mm_heroes if h.get("team") in ("blue", "red")]
    jungle_mm = [h for h in mm_heroes if h.get("team") == "jungle"]

    for slot, h in enumerate(all_mm[:10], 1):
        row[f"mm_{slot}_name"] = h.get("name")
        row[f"mm_{slot}_team"] = h.get("team")
        row[f"mm_{slot}_norm_x"] = h.get("norm_x")
        row[f"mm_{slot}_norm_y"] = h.get("norm_y")
        row[f"mm_{slot}_game_x"] = h.get("game_x")
        row[f"mm_{slot}_game_y"] = h.get("game_y")
        row[f"mm_{slot}_confidence"] = h.get("confidence")
        row[f"mm_{slot}_region"] = h.get("region")
        row[f"mm_{slot}_lane"] = h.get("lane")

    # Jungle objectives (max 15)
    for j in range(1, 16):
        if j <= len(jungle_mm):
            h = jungle_mm[j - 1]
            row[f"jungle_{j}_name"] = h.get("name")
            row[f"jungle_{j}_norm_x"] = h.get("norm_x")
            row[f"jungle_{j}_norm_y"] = h.get("norm_y")
            row[f"jungle_{j}_game_x"] = h.get("game_x")
            row[f"jungle_{j}_game_y"] = h.get("game_y")

    # ── Jungle camp positions (fixed) + status ──
    jungle_status = status.get("jungle_status", {}) or {}
    for cid in JUNGLE_CAMP_IDS:
        row[f"jungle_{cid}_norm_x"] = JUNGLE_CAMP_POSITIONS.get(cid, (None, None))[0]
        row[f"jungle_{cid}_norm_y"] = JUNGLE_CAMP_POSITIONS.get(cid, (None, None))[1]
        row[f"jungle_{cid}_status"] = jungle_status.get(cid, "unknown")

    # ── Towers ──
    row["towers_alive"] = (status.get("blue_towers_alive") or 0) + (status.get("red_towers_alive") or 0)

    # ── Objectives ──
    row["legend_timer"] = status.get("lord_timer") or status.get("turtle_timer")

    return row


class GameStateExporter:
    """
    Collect game state frames and export to Parquet.

    Args:
        output_dir: Direktori output untuk file .parquet.
        flush_every: Auto-flush setiap N frame (0 = manual).
        auto_install: Coba pip install pyarrow+pandas jika belum ada.
    """

    def __init__(
        self,
        output_dir: str = "data/game_states",
        flush_every: int = 0,
        auto_install: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []
        self._frame_count = 0

        if auto_install and not HAS_PYARROW:
            self._auto_install()

    def _auto_install(self):
        """Install pyarrow + pandas if missing."""
        global HAS_PYARROW, pd, pa, pq
        try:
            import subprocess, sys
            logger.info("Installing pyarrow + pandas for parquet export...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyarrow", "pandas", "-q"],
            )
            import pandas as _pd
            import pyarrow as _pa
            import pyarrow.parquet as _pq
            pd, pa, pq = _pd, _pa, _pq
            HAS_PYARROW = True
            logger.info("pyarrow + pandas installed")
        except Exception as e:
            logger.warning("Failed to install pyarrow+pandas: %s", e)

    def append(self, frame_idx: int, video_time: float, status: dict[str, Any]):
        """Collect satu frame game state ke buffer."""
        row = _flatten_status(frame_idx, video_time, status)
        self._buffer.append(row)
        self._frame_count += 1

        if self.flush_every > 0 and self._frame_count % self.flush_every == 0:
            self.flush(f"batch_{self._frame_count // self.flush_every}")

    def flush(self, match_id: str, keep_buffer: bool = True):
        """
        Tulis buffer ke file Parquet.

        Args:
            match_id: Nama file (e.g. "match_alpha_1").
            keep_buffer: False = kosongkan buffer setelah nulis.
        """
        if not HAS_PYARROW:
            logger.warning("pyarrow tidak tersedia, skip parquet export")
            return

        if not self._buffer:
            logger.debug("Buffer kosong, skip flush")
            return

        df = pd.DataFrame(self._buffer)
        filepath = self.output_dir / f"{match_id}.parquet"

        # Convert dtype optimal
        for col in df.columns:
            if df[col].dtype == object:
                # Cek apakah kolom bisa jadi float (nullable)
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, filepath, compression="snappy")

        logger.info(
            "✅ Exported %d rows → %s (%s)",
            len(df), filepath, _format_size(filepath.stat().st_size),
        )

        if not keep_buffer:
            self._buffer.clear()

    def flush_and_reset(self, match_id: str):
        """Tulis buffer lalu kosongkan."""
        self.flush(match_id, keep_buffer=False)

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def total_exported(self) -> int:
        return self._frame_count


def _format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.1f} KB"
    else:
        return f"{bytes_ / 1024 ** 2:.1f} MB"


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    exporter = GameStateExporter(".tmp/test_export", flush_every=0)
    for i in range(5):
        exporter.append(i, i / 30.0, {
            "hero_name": "alpha",
            "level": 4,
            "hp_pct": 0.85,
            "mana_pct": 0.50,
            "gold": 3200,
            "kda": "5/1/3",
            "items": ["warrior_boots", "bloodlust_axe"],
            "skills": {
                "skill_1": {"ready": True, "cooldown": False},
                "skill_2": {"ready": False, "cooldown": True, "remaining_cd": 3.2},
                "skill_3": {"ready": True, "cooldown": False},
                "battle_spell": {"ready": True, "cooldown": False, "spell_name": "flicker"},
            },
            "blue_team_heroes": [
                {"slot": 1, "name": "alpha", "hp_pct": 0.85, "confidence": 0.9},
                {"slot": 2, "name": "franco", "hp_pct": 0.72, "confidence": 0.8},
            ],
            "red_team_heroes": [
                {"slot": 1, "name": "chou", "confidence": 0.7},
            ],
            "minimap_heroes": [
                {"name": "alpha", "team": "blue", "norm_x": 0.3, "norm_y": 0.5,
                 "game_x": 3000, "game_y": 5000, "confidence": 0.9, "region": "mid_lane", "lane": "mid"},
            ],
        })
    exporter.flush_and_reset("test_match")

    # Baca balik pake pyarrow langsung (lebih stabil dari pandas.read_parquet)
    if pa:
        table = pq.read_table(".tmp/test_export/test_match.parquet")
        print(f"\n✅ Read back: {table.num_rows} rows, {table.num_columns} columns")
        print("Columns with data:")
        for col in table.schema.names:
            col_data = table.column(col)
            non_null = col_data.null_count
            if non_null < table.num_rows:
                print(f"  {col}: {table.num_rows - non_null} non-null")
    else:
        print("pyarrow not available for read-back test")

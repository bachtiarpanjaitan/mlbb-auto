"""
Minimap Simulation — Replay hero positions from parquet game state data.

Reads a .parquet file from data/game_states/ and animates hero dots on a
1200×1200 minimap window, showing movement based on norm_x/norm_y columns.

Usage:
    python tools/simulate_minimap.py              # interactive file picker
    python tools/simulate_minimap.py alpha_1      # direct file name

Controls (all toggleable inside the window via keyboard):
    Space   — Pause / resume
    A / D   — Frame step (when paused)
    J       — Toggle jungle dots
    T       — Toggle trail mode
    G       — Toggle grid overlay
    L       — Toggle hero name labels
    Z       — Toggle minimap region zones
    C       — Toggle combat contact rings
    H       — Toggle help overlay
    + / =   — Speed up (×1.25)
    -       — Slow down (÷1.25)
    R       — Restart from frame 0
    Q / ESC — Quit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("simulate_minimap")

# ── Constants ──────────────────────────────────────────────────────────
WINDOW_SIZE = 1600
DOT_RADIUS = 16
GLOW_RADIUS = 24
FONT_SCALE = WINDOW_SIZE / 1000  # scale text proportionally to canvas size
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colour presets (BGR)
BG_COLOR = (30, 50, 30)
GRID_COLOR = (50, 70, 50)
LANE_COLOR = (70, 100, 70)
TEXT_COLOR = (200, 230, 200)
BLUE_DOT = (255, 150, 50)
BLUE_GLOW = (200, 100, 20)
RED_DOT = (50, 50, 220)
RED_GLOW = (30, 30, 180)
JUNGLE_DOT = (50, 230, 50)
JUNGLE_GLOW = (30, 180, 30)
UNKNOWN_DOT = (150, 150, 150)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "game_states"

# ── Region constants ──────────────────────────────────────────────────
REGIONS_PATH = BASE_DIR / "assets" / "databases" / "regions.json"
# Minimap dimensions from layout.yaml (used to normalise region points)
MM_WIDTH = 350
MM_HEIGHT = 340

# Region colour mapping (BGR, with alpha)
REGION_COLORS = {
    "river":              (180, 120,  60, 0.25),  # blue-ish
    "top_lane":           ( 60, 180,  60, 0.20),  # green
    "mid_lane":           ( 60, 180,  60, 0.20),
    "bottom_lane":        ( 60, 180,  60, 0.20),
    "base":               ( 60,  60, 180, 0.15),  # red-ish (ally)
    "enemy_base":         ( 60,  60, 180, 0.15),
    "ally_red_buff":      (180,  60, 180, 0.30),  # purple
    "enemy_red_buff":     (180,  60, 180, 0.30),
    "ally_blue_buff":     (180,  60, 180, 0.30),
    "enemy_blue_buff":    (180,  60, 180, 0.30),
    "bottom_pit_lord":    ( 60, 160, 255, 0.35),  # gold
    "top_pit_lord":       ( 60, 160, 255, 0.35),
    # Blue towers (cyan)
    "blue_tower_top_outer": (200, 200,  50, 0.40),
    "blue_tower_top_inner": (200, 200,  50, 0.40),
    "blue_tower_top_base":  (200, 200,  50, 0.40),
    "blue_tower_mid_outer": (200, 200,  50, 0.40),
    "blue_tower_mid_inner": (200, 200,  50, 0.40),
    "blue_tower_mid_base":  (200, 200,  50, 0.40),
    "blue_tower_bot_outer": (200, 200,  50, 0.40),
    "blue_tower_bot_inner": (200, 200,  50, 0.40),
    "blue_tower_bot_base":  (200, 200,  50, 0.40),
    # Red towers (red)
    "red_tower_top_base":   ( 60,  60, 200, 0.40),
    "red_tower_top_inner":  ( 60,  60, 200, 0.40),
    "red_tower_top_outer":  ( 60,  60, 200, 0.40),
    "red_tower_mid_base":   ( 60,  60, 200, 0.40),
    "red_tower_mid_inner":  ( 60,  60, 200, 0.40),
    "red_tower_mid_outer":  ( 60,  60, 200, 0.40),
    "red_tower_bot_base":   ( 60,  60, 200, 0.40),
    "red_tower_bot_inner":  ( 60,  60, 200, 0.40),
    "red_tower_bot_outer":  ( 60,  60, 200, 0.40),
}

# Fallback for any unnamed region
_DEFAULT_REGION_COLOR = (120, 120, 120, 0.15)

# ── Lane reference points (normalised 0-1) ──
LANE_SEGMENTS = {
    "mid": [
        (0.50, 0.08), (0.50, 0.20), (0.50, 0.35), (0.50, 0.50),
        (0.50, 0.65), (0.50, 0.80), (0.50, 0.92),
    ],
}


def _to_pixel(nx: float, ny: float, margin: int = 20) -> tuple[int, int]:
    """Convert normalised (0-1) coords to pixel coords on the 1000×1000 canvas."""
    scale = WINDOW_SIZE - 2 * margin
    px = int(margin + nx * scale)
    py = int(margin + ny * scale)
    return px, py


def _draw_minimap_bg(canvas: np.ndarray):
    """Draw the minimap background with lane markers and base icons."""
    h, w = canvas.shape[:2]

    # ── Background gradient ──
    for y in range(h):
        blend = 1.0 - (y / h) * 0.15
        canvas[y, :] = (
            int(BG_COLOR[0] * blend),
            int(BG_COLOR[1] * blend),
            int(BG_COLOR[2] * blend),
        )

    # ── Thin grid ──
    for i in range(11):
        x = int(i * w / 10)
        y = int(i * h / 10)
        cv2.line(canvas, (x, 0), (x, h), GRID_COLOR, 1)
        cv2.line(canvas, (0, y), (w, y), GRID_COLOR, 1)

    # ── Lane lines ──
    for lane_pts in LANE_SEGMENTS.values():
        pts = [_to_pixel(nx, ny, 20) for nx, ny in lane_pts]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], LANE_COLOR, 2, cv2.LINE_AA)

    # ── Bases ──
    # Blue (ally) base: bottom-left; Red (enemy) base: top-right
    for nx, ny, color, label in [
        (0.15, 0.85, (255, 150,  50), "BLUE BASE"),
        (0.85, 0.15, ( 50,  50, 220), "RED BASE" ),
    ]:
        px, py = _to_pixel(nx, ny, 20)
        cv2.circle(canvas, (px, py), 16, color, -1)
        cv2.circle(canvas, (px, py), 20, (200, 200, 200), 1)
        cv2.putText(canvas, label, (px - 20, py + 30), FONT, 0.72, (200, 200, 200), 1, cv2.LINE_AA)

    # ── River center ──
    river_cx, river_cy = _to_pixel(0.5, 0.5, 20)
    cv2.circle(canvas, (river_cx, river_cy), 10, (60, 120, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIVER", (river_cx - 24, river_cy + 4), FONT, 0.56, (60, 120, 180), 1, cv2.LINE_AA)


def _draw_dot(canvas: np.ndarray, px: int, py: int,
              fill_color: tuple, glow_color: tuple,
              name: str | None = None, confidence: float | None = None,
              show_labels: bool = True, show_glow: bool = True):
    """Draw a hero dot with glow and optional label."""
    # Glow
    if show_glow:
        cv2.circle(canvas, (px, py), GLOW_RADIUS, glow_color, 2, cv2.LINE_AA)
    # Fill
    cv2.circle(canvas, (px, py), DOT_RADIUS, fill_color, -1, cv2.LINE_AA)
    # White inner dot
    cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Label
    if name and show_labels:
        label = name[:10]
        if confidence is not None and confidence < 0.8:
            label += f" {confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 1)
        lx = px + DOT_RADIUS + 6
        ly = py + th // 3
        pad = 4

        # Background box
        cv2.rectangle(canvas,
                      (lx - pad, ly - th - pad),
                      (lx + tw + pad, ly + pad),
                      (0, 0, 0), -1)
        cv2.rectangle(canvas,
                      (lx - pad, ly - th - pad),
                      (lx + tw + pad, ly + pad),
                      glow_color, 1)
        cv2.putText(canvas, label, (lx, ly), FONT, 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, video_time: float, frame_idx: int,
              speed: float, paused: bool, frame_count: int, total_frames: int,
              toggles: dict | None = None):
    """Draw status HUD at the top-left with toggle states."""
    lines = [
        (f"Time: {video_time:.1f}s  Frame: {frame_idx}/{total_frames}", TEXT_COLOR),
        (f"Speed: {speed:.2f}x  {'⏸ PAUSED' if paused else '▶ PLAYING'}", (0, 200, 255) if paused else TEXT_COLOR),
    ]

    # Toggle status line
    if toggles:
        parts = []
        for key, label in [("jungle", "Jungle"), ("trails", "Trail"),
                           ("grid", "Grid"), ("labels", "Labels"),
                           ("regions", "Zones"),
                           ("contacts", "Combat"),
                           ("help", "Help")]:
            val = toggles.get(key, False)
            on_color = "\033[32m"  # green
            parts.append(f"{label}:{'ON' if val else 'OFF'}")
        status = " | ".join(parts)
        lines.append((status, (100, 200, 100) if any(toggles.values()) else (120, 140, 160)))

    for i, (text, color) in enumerate(lines):
        y = 25 + i * 24
        tw = cv2.getTextSize(text, FONT, 0.8, 1)[0][0]
        cv2.rectangle(canvas, (8, y - 16), (min(700, 8 + tw + 12), y + 4),
                      (0, 0, 0, 0.6), -1)
        cv2.putText(canvas, text, (14, y), FONT, 0.8, color, 1, cv2.LINE_AA)

    # Shortcut bar at bottom (above progress bar)
    _draw_shortcut_bar(canvas, toggles)

    # Progress bar at bottom
    pb_y = WINDOW_SIZE - 12
    pb_w = WINDOW_SIZE - 40
    progress = frame_count / max(total_frames - 1, 1)
    cv2.rectangle(canvas, (20, pb_y), (20 + pb_w, pb_y + 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (20, pb_y), (20 + int(pb_w * progress), pb_y + 6), (100, 200, 100), -1)


# ── Shortcut bar ──────────────────────────────────────────────────────

def _draw_shortcut_bar(canvas: np.ndarray, toggles: dict | None = None):
    """Draw compact shortcut info bar at the bottom of the window."""
    h, w = canvas.shape[:2]
    if toggles is None:
        toggles = {}

    # Bar position: 5px above progress bar
    bar_y = h - 26
    bar_h = 18
    pad_x = 12

    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (pad_x, bar_y), (w - pad_x, bar_y + bar_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Shortcut groups: (keys, label, toggle_key)
    shortcuts = [
        ("Space", "Pause", None),
        ("A/D", "Step", None),
        ("J", "Jungle", "jungle"),
        ("T", "Trail", "trails"),
        ("G", "Grid", "grid"),
        ("L", "Labels", "labels"),
        ("Z", "Zones", "regions"),
        ("C", "Combat", "contacts"),
        ("+/-", "Speed", None),
        ("R", "Reset", None),
        ("H", "Help", "help"),
        ("Q", "Quit", None),
    ]

    x = pad_x + 6
    fs = 0.64
    gap = 4

    for keys, label, toggle_key in shortcuts:
        # Key label (e.g. "Space", "J")
        key_text = f"{keys}"
        (kw, kh), _ = cv2.getTextSize(key_text, FONT, fs, 1)
        cv2.putText(canvas, key_text, (x, bar_y + 13), FONT, fs,
                    (0, 220, 220), 1, cv2.LINE_AA)
        x += kw + gap

        # Description (e.g. "Pause")
        desc_text = label
        if toggle_key and toggle_key in toggles:
            val = toggles[toggle_key]
            desc_text += ":ON" if val else ""
            desc_color = (100, 255, 100) if val else (120, 120, 120)
        else:
            desc_color = (160, 160, 160)

        (dw, dh), _ = cv2.getTextSize(desc_text, FONT, fs, 1)
        cv2.putText(canvas, desc_text, (x, bar_y + 13), FONT, fs,
                    desc_color, 1, cv2.LINE_AA)
        x += dw + 10

    # Divider line above bar
    cv2.line(canvas, (pad_x + 4, bar_y), (w - pad_x - 4, bar_y), (50, 50, 60), 1)


# ── Help overlay ──────────────────────────────────────────────────────

def _draw_help(canvas: np.ndarray):
    """Draw key bindings help overlay at center."""
    h, w = canvas.shape[:2]
    lines = [
        ("───── Controls ─────", (200, 200, 255)),
        ("Space", "Pause / resume"),
        ("A / D", "Frame step back / forward"),
        ("J", "Toggle jungle objective dots"),
        ("T", "Toggle trail mode (movement history)"),
        ("G", "Toggle grid overlay"),
        ("L", "Toggle hero name labels"),
        ("Z", "Toggle minimap region zones"),
        ("C", "Toggle combat contact rings"),
        ("H", "Toggle this help"),
        ("+ / =", "Speed up"),
        ("-", "Slow down"),
        ("R", "Restart from frame 0"),
        ("Q / ESC", "Quit"),
    ]
    line_h = 22
    pad = 12
    total_h = len(lines) * line_h + pad * 2
    x = w // 2 - 160
    y = h // 2 - total_h // 2

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + 320, y + total_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + 320, y + total_h), (80, 80, 120), 1)

    yy = y + pad + line_h - 6
    for line in lines:
        if len(line) == 1:
            # Section header
            t, c = line[0], (200, 200, 255)
            cv2.putText(canvas, t, (x + pad + 10, yy), FONT, 0.72, c, 1, cv2.LINE_AA)
        elif isinstance(line[0], str) and isinstance(line[1], str):
            k, desc = line
            cv2.putText(canvas, f"  {k}:", (x + pad + 4, yy), FONT, 0.64, (0, 200, 200), 1, cv2.LINE_AA)
            tw = cv2.getTextSize(f"  {k}:", FONT, 0.64, 1)[0][0]
            cv2.putText(canvas, desc, (x + pad + tw + 10, yy), FONT, 0.64, (200, 200, 200), 1, cv2.LINE_AA)
        yy += line_h


# ── Region overlay ────────────────────────────────────────────────────

def _load_regions() -> list[dict]:
    """Load minimap regions from regions.json and convert to normalised coords."""
    try:
        import json
        with open(REGIONS_PATH) as f:
            raw = json.load(f)
    except Exception as e:
        log.warning("Failed to load regions: %s", e)
        return []

    regions = []
    for r in raw:
        rid = r.get("id", "unknown")
        pts = r.get("points", [])
        if len(pts) < 3:
            continue

        # Normalise pixel coords to 0-1, then convert to canvas pixels
        canvas_pts = []
        for px, py in pts:
            nx = px / MM_WIDTH
            ny = py / MM_HEIGHT
            cx, cy = _to_pixel(nx, ny)
            canvas_pts.append([cx, cy])

        b, g, r_c, alpha = REGION_COLORS.get(rid, _DEFAULT_REGION_COLOR)
        label = rid.replace("_", " ").title()

        regions.append({
            "id": rid,
            "label": label,
            "points": np.array(canvas_pts, dtype=np.int32),
            "color": (int(b), int(g), int(r_c)),
            "alpha": float(alpha),
            "center": np.mean(canvas_pts, axis=0).astype(int).tolist(),
        })

    log.info("Loaded %d minimap regions", len(regions))
    return regions


def _draw_regions(canvas: np.ndarray, regions: list[dict]):
    """Draw minimap regions with rounded corners and soft glow.

    Each polygon's corners are rounded via blur+threshold, then a soft glow
    extends outward from the rounded shape.
    """
    if not regions:
        return

    h, w = canvas.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    base_f32 = canvas.astype(np.float32)

    for r in regions:
        pts = r["points"]
        color = np.array(r["color"], dtype=np.float32)
        alpha = r["alpha"]

        # Step 1: draw filled polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Step 2: blur + threshold → rounded corners
        k_round = 101
        rounded = cv2.GaussianBlur(mask, (k_round, k_round), 0)
        _, rounded = cv2.threshold(rounded, 50, 255, cv2.THRESH_BINARY)

        # Step 3: blur again → soft glow outward
        k_glow = 81
        glow = cv2.GaussianBlur(rounded, (k_glow, k_glow), 0)
        glow_f = glow.astype(np.float32) / 255.0

        # Composite: colour × glow mask × alpha
        for c in range(3):
            overlay[:, :, c] += glow_f * (color[c] / 255.0) * alpha

    # Clamp overlay so overlapping regions don't wash out
    overlay = np.clip(overlay, 0, 1)

    # Blend: overlay on top of base
    blended = base_f32 / 255.0 * (1.0 - overlay * 0.6) + overlay * 0.6
    np.copyto(canvas, (np.clip(blended * 255, 0, 255)).astype(np.uint8))

    # Region labels (on top, readable)
    for r in regions:
        cx, cy = r["center"]
        label = r["label"]
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.56, 1)
        lx = max(4, min(cx - tw // 2, WINDOW_SIZE - tw - 4))
        ly = max(12, cy)
        cv2.rectangle(canvas, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
        cv2.putText(canvas, label, (lx, ly), FONT, 0.56, (200, 200, 200), 1, cv2.LINE_AA)


# ── Contact detection ────────────────────────────────────────────────

def _detect_contacts(data: dict, frame_idx: int,
                     prev_hp: dict, prev_hp_frame: dict) -> list[dict]:
    """Detect combat contacts via HP drops with enemy proximity.

    Returns list of contact events: [{pos, team, intensity}, ...]
    """
    if frame_idx < 1:
        return []

    contacts = []
    contact_dist = 0.10  # max normalised distance for "near enemy"

    for team in ("blue", "red"):
        enemy_team = "red" if team == "blue" else "blue"
        for slot in range(1, 6):
            key = f"{team}_hero_{slot}_hp_pct"
            hp = data.get(key, np.array([np.nan]))[frame_idx]
            prev_val = prev_hp.get(key)
            prev_fr = prev_hp_frame.get(key, -1)

            if np.isnan(hp) or prev_val is None or np.isnan(prev_val):
                continue

            # Only trigger on actual HP change (not stale data)
            if abs(hp - prev_val) < 0.001 or (frame_idx - prev_fr) > 3:
                continue

            drop = prev_val - hp
            if drop < 0.03:  # minimum 3% HP drop
                continue

            # Get this hero's position
            nx = data.get(f"{team}_mm_{slot}_norm_x", np.array([np.nan]))[frame_idx]
            ny = data.get(f"{team}_mm_{slot}_norm_y", np.array([np.nan]))[frame_idx]
            if np.isnan(nx) or np.isnan(ny):
                continue

            # Find nearest enemy within contact_dist
            best_enemy = None
            best_dist = float("inf")
            for eslot in range(1, 6):
                enx = data.get(f"{enemy_team}_mm_{eslot}_norm_x", np.array([np.nan]))[frame_idx]
                eny = data.get(f"{enemy_team}_mm_{eslot}_norm_y", np.array([np.nan]))[frame_idx]
                if np.isnan(enx) or np.isnan(eny):
                    continue
                dist = ((nx - enx) ** 2 + (ny - eny) ** 2) ** 0.5
                if dist < contact_dist and dist < best_dist:
                    best_dist = dist
                    best_enemy = (enx, eny)

            if best_enemy is not None:
                # Contact position = at the hero who took damage
                cx, cy = _to_pixel(nx, ny)
                intensity = min(1.0, drop / 0.3)
                contacts.append({
                    "pos": (cx, cy),
                    "intensity": intensity,
                    "team": team,
                    "drop": round(drop, 3),
                    "frame": frame_idx,
                })

    return contacts


def _draw_contact_effects(canvas: np.ndarray,
                          active_contacts: list[dict]):
    """Draw expanding ring effects for active combat contacts."""
    for c in active_contacts:
        px, py = c["pos"]
        age = c.get("age", 0)
        intensity = c["intensity"]

        # Ring expands and fades over 20 frames
        max_age = 20
        progress = age / max_age
        if progress >= 1.0:
            continue

        ring_radius = int(10 + progress * 60)
        alpha = (1.0 - progress) * intensity * 0.8

        # Team colour for the ring
        if c["team"] == "blue":
            color = (255, 200, 100)  # gold-orange
        else:
            color = (100, 100, 255)  # red

        # Draw ring with thickness that fades
        thickness = max(1, int(4 * (1.0 - progress)))
        cv2.circle(canvas, (px, py), ring_radius, color, thickness, cv2.LINE_AA)

        # Inner glow dot
        glow_r = max(2, int(8 * (1.0 - progress)))
        cv2.circle(canvas, (px, py), glow_r, color, -1, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────

def load_data(parquet_path: str) -> dict:
    """Load parquet file into memory-friendly numpy arrays."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("❌ pyarrow not installed. Run: pip install pyarrow")
        sys.exit(1)

    table = pq.read_table(parquet_path)
    columns = table.column_names

    data: dict = {
        "frame_idx": np.array(table.column("frame_idx").to_pylist(), dtype=np.int64),
        "video_time": np.array(table.column("video_time").to_pylist(), dtype=np.float64),
    }

    data["num_frames"] = len(data["frame_idx"])

    for team in ("blue", "red"):
        for slot in range(1, 6):
            prefix = f"{team}_mm_{slot}"
            for attr in ("norm_x", "norm_y", "name", "confidence"):
                col = f"{prefix}_{attr}"
                if col in columns:
                    arr = table.column(col).to_pylist()
                    if attr in ("norm_x", "norm_y", "confidence"):
                        data[col] = np.array(arr, dtype=np.float64)
                    else:
                        data[col] = np.array([(s or "") for s in arr], dtype=object)
                else:
                    if attr in ("norm_x", "norm_y", "confidence"):
                        data[col] = np.full(data["num_frames"], np.nan, dtype=np.float64)
                    else:
                        data[col] = np.full(data["num_frames"], "", dtype=object)

    # Jungle (limited support)
    for j in range(1, 16):
        for attr in ("norm_x", "norm_y", "name"):
            col = f"jungle_{j}_{attr}"
            if col in columns:
                arr = table.column(col).to_pylist()
                if attr in ("norm_x", "norm_y"):
                    data[col] = np.array(arr, dtype=np.float64)
                else:
                    data[col] = np.array([(s or "") for s in arr], dtype=object)

    # ── Hero HP data (for contact detection) ──
    for team in ("blue", "red"):
        for slot in range(1, 6):
            col = f"{team}_hero_{slot}_hp_pct"
            if col in columns:
                arr = table.column(col).to_pylist()
                data[col] = np.array(arr, dtype=np.float64)
            else:
                data[col] = np.full(data["num_frames"], np.nan, dtype=np.float64)

    log.info("Loaded %d frames from %s", data["num_frames"], parquet_path)
    return data


def pick_file() -> str:
    """Interactive file picker for parquet files in data/game_states/."""
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        sys.exit(1)

    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        print(f"❌ No .parquet files found in {DATA_DIR}")
        print("   Run debug_vision.py with Parquet export enabled (P key) first.")
        sys.exit(1)

    print("\n📊 Minimap Simulation — Pilih file parquet:")
    for idx, fp in enumerate(files, 1):
        size_kb = fp.stat().st_size / 1024
        print(f"  [{idx}] {fp.stem} ({size_kb:.0f} KB)")

    while True:
        try:
            choice = input(f"\nMasukkan nomor (1-{len(files)}) [default: 1]: ").strip()
            if not choice:
                return str(files[0])
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(files[idx])
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\n❌ Input dibatalkan.")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Replay minimap hero positions from parquet data")
    ap.add_argument("file", nargs="?", help="Parquet file name (without path/extension)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    args = ap.parse_args()

    # ── Resolve file ──
    if args.file:
        candidate = args.file
        if not candidate.endswith(".parquet"):
            candidate += ".parquet"
        fp = DATA_DIR / candidate
        if not fp.exists():
            fp = Path(candidate)
        if not fp.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        parquet_path = str(fp)
    else:
        parquet_path = pick_file()

    # ── Load data ──
    data = load_data(parquet_path)
    num_frames = data["num_frames"]

    # ── Window setup ──
    cv2.namedWindow("Minimap Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Minimap Simulation", WINDOW_SIZE, WINDOW_SIZE)
    cv2.moveWindow("Minimap Simulation", 100, 50)
    print(f"🎬 Playing: {Path(parquet_path).stem} ({num_frames} frames)")
    print("   Press H for help overlay in window")

    paused = False
    speed_mult = args.speed
    frame_idx = 0
    trails: dict[str, list] = {}  # "b1","b2",..,"r5" → list of (px,py)

    # Toggle states (all controlled via keyboard in-window)
    toggles = {
        "jungle": True,    # J — show jungle dots
        "trails": False,   # T — movement trails
        "grid": False,     # G — grid overlay
        "labels": True,    # L — hero name labels
        "help": False,     # H — help overlay
        "regions": True,   # Z — minimap region polygons
        "contacts": True,  # C — combat contact rings
    }

    # ── Load minimap regions ──
    regions = _load_regions()
    log.info("Loaded %d minimap regions", len(regions))

    # Pre-render static background
    base_bg = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    _draw_minimap_bg(base_bg)

    # Pre-render region glow overlay (render sekali, simpan, blend tiap frame)
    region_overlay: np.ndarray | None = None
    if regions:
        ro = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        _draw_regions(ro, regions)
        region_overlay = ro

    # ── Contact detection state ──
    prev_hp: dict[str, float] = {}
    prev_hp_frame: dict[str, int] = {}
    active_contacts: list[dict] = []

    # ── Tracking for trail mode ──
    def draw_frame(idx: int) -> np.ndarray:
        nonlocal region_overlay
        canvas = base_bg.copy()

        if idx >= num_frames:
            return canvas

        vt = float(data["video_time"][idx])

        # ── Grid toggle ──
        if toggles["grid"]:
            for i in range(11):
                x = int(i * WINDOW_SIZE / 10)
                y = int(i * WINDOW_SIZE / 10)
                cv2.line(canvas, (x, 0), (x, WINDOW_SIZE), (60, 80, 60), 1)
                cv2.line(canvas, (0, y), (WINDOW_SIZE, y), (60, 80, 60), 1)

        # ── Region polygons (blend dari pre-rendered overlay) ──
        if toggles["regions"] and region_overlay is not None:
            cv2.addWeighted(region_overlay, 0.6, canvas, 0.4, 0, canvas)

        # ── Draw trails (per-hero lines) ──
        if toggles["trails"]:
            for key, pts in trails.items():
                if len(pts) > 1:
                    color = (200, 100, 20) if key.startswith("b") else (30, 30, 180)
                    for i in range(len(pts) - 1):
                        cv2.line(canvas, pts[i], pts[i+1], color, 2, cv2.LINE_AA)

        # ── Contact effects ──
        if toggles["contacts"] and active_contacts:
            _draw_contact_effects(canvas, active_contacts)

        # ── Draw hero dots ──
        for team, dot_color, glow_color in [
            ("blue", BLUE_DOT, BLUE_GLOW),
            ("red", RED_DOT, RED_GLOW),
        ]:
            for slot in range(1, 6):
                nx = data.get(f"{team}_mm_{slot}_norm_x", np.array([np.nan]))[idx]
                ny = data.get(f"{team}_mm_{slot}_norm_y", np.array([np.nan]))[idx]
                name = str(data.get(f"{team}_mm_{slot}_name", np.array([""]))[idx])
                conf = data.get(f"{team}_mm_{slot}_confidence", np.array([np.nan]))[idx]

                if np.isnan(nx) or np.isnan(ny):
                    continue

                px, py = _to_pixel(nx, ny)
                display_name = name if name and name != "nan" else None
                display_conf = float(conf) if not np.isnan(conf) else None

                _draw_dot(canvas, px, py, dot_color, glow_color,
                          display_name, display_conf,
                          show_labels=toggles["labels"])

        # ── Jungle dots ──
        if toggles["jungle"]:
            for j in range(1, 16):
                nx = data.get(f"jungle_{j}_norm_x", np.array([np.nan]))[idx]
                ny = data.get(f"jungle_{j}_norm_y", np.array([np.nan]))[idx]
                name = str(data.get(f"jungle_{j}_name", np.array([""]))[idx])
                if np.isnan(nx) or np.isnan(ny):
                    continue
                px, py = _to_pixel(nx, ny)
                display_name = name if name and name != "nan" else "jungle"
                _draw_dot(canvas, px, py, JUNGLE_DOT, JUNGLE_GLOW, display_name,
                          show_labels=toggles["labels"])

        # ── HUD ──
        _draw_hud(canvas, vt, idx, speed_mult, paused, idx, num_frames, toggles)

        # ── Legend ──
        legend_items = [
            (BLUE_DOT, "Blue Team"),
            (RED_DOT, "Red Team"),
        ]
        if toggles["jungle"]:
            legend_items.append((JUNGLE_DOT, "Jungle"))
        if toggles["trails"]:
            legend_items.append(((200, 200, 100), "Trail"))
        lx, ly = WINDOW_SIZE - 140, 10
        for color, label in legend_items:
            if isinstance(color, tuple) and len(color) == 3:
                cv2.circle(canvas, (lx + 6, ly + 8), 6, color, -1)
                cv2.circle(canvas, (lx + 6, ly + 8), 6, (200, 200, 200), 1)
            cv2.putText(canvas, label, (lx + 18, ly + 12), FONT, 0.64, TEXT_COLOR, 1, cv2.LINE_AA)
            ly += 22

        # ── Help overlay ──
        if toggles["help"]:
            _draw_help(canvas)

        # ── Frame indicator at top-right ──
        total_str = f"Frame {idx + 1}/{num_frames}"
        (tw, th), _ = cv2.getTextSize(total_str, FONT, 0.64, 1)
        fx = WINDOW_SIZE - tw - 14
        cv2.putText(canvas, total_str, (fx, 20), FONT, 0.64, (150, 150, 150), 1, cv2.LINE_AA)

        return canvas

    # ── Main loop ──
    while frame_idx < num_frames:
        if not paused:
            vis = draw_frame(frame_idx)

            # Update trails (sample every 5 frames, per-hero)
            if toggles["trails"] and frame_idx % 5 == 0:
                for team, prefix in [("blue", "b"), ("red", "r")]:
                    for slot in range(1, 6):
                        nx = data.get(f"{team}_mm_{slot}_norm_x", np.array([np.nan]))[frame_idx]
                        ny = data.get(f"{team}_mm_{slot}_norm_y", np.array([np.nan]))[frame_idx]
                        if np.isnan(nx) or np.isnan(ny):
                            continue
                        key = f"{prefix}{slot}"
                        if key not in trails:
                            trails[key] = []
                        trails[key].append(_to_pixel(nx, ny))
                        if len(trails[key]) > 50:
                            trails[key] = trails[key][-50:]

        # ── Contact detection (before render, so rings appear immediately) ──
        if toggles["contacts"]:
            new_contacts = _detect_contacts(data, frame_idx, prev_hp, prev_hp_frame)
            for c in new_contacts:
                c["age"] = 0
                active_contacts.append(c)

            # Age all active contacts, remove expired
            for c in active_contacts:
                c["age"] = c.get("age", 0) + 1
            active_contacts[:] = [c for c in active_contacts if c.get("age", 0) < 20]
            if len(active_contacts) > 50:
                active_contacts[:] = active_contacts[-50:]

        # Track HP history for next frame detection
        for team in ("blue", "red"):
            for slot in range(1, 6):
                key = f"{team}_hero_{slot}_hp_pct"
                hp = data.get(key, np.array([np.nan]))[frame_idx]
                if not np.isnan(hp):
                    prev_hp[key] = float(hp)
                    prev_hp_frame[key] = frame_idx

            cv2.imshow("Minimap Simulation", vis)

        # ── Controls ──
        delay = max(1, int(33 / speed_mult))  # ~30fps base
        k = cv2.waitKey(delay)

        if k in (ord("q"), 27):  # q or ESC
            break

        if k == ord(" "):
            paused ^= True
            print(f"{'⏸ PAUSED' if paused else '▶ RESUMED'} at frame {frame_idx}")

        if k == ord("z"):
            toggles["regions"] ^= True
            print(f"Regions: {'ON' if toggles['regions'] else 'OFF'}")

        # Frame stepping (works with A/D and ←/→)
        if k in (ord("a"), 2, 0xf702, 0x250000):  # left / A
            if not paused:
                paused = True
            frame_idx = max(0, frame_idx - 1)
            vis = draw_frame(frame_idx)
            cv2.imshow("Minimap Simulation", vis)
            print(f"Frame: {frame_idx + 1}/{num_frames}")
        if k in (ord("d"), 3, 0xf703, 0x270000):  # right / D
            if not paused:
                paused = True
            frame_idx = min(num_frames - 1, frame_idx + 1)
            vis = draw_frame(frame_idx)
            cv2.imshow("Minimap Simulation", vis)
            print(f"Frame: {frame_idx + 1}/{num_frames}")
        if k in (ord("="), ord("+")):
            speed_mult = min(10.0, speed_mult * 1.25)
            print(f"⏩ Speed: {speed_mult:.2f}x")
        if k == ord("-"):
            speed_mult = max(0.1, speed_mult / 1.25)
            print(f"⏪ Speed: {speed_mult:.2f}x")
        if k == ord("r"):
            frame_idx = 0
            trails.clear()
            print("🔄 Restart")
        if k == ord("t"):
            toggles["trails"] ^= True
            if not toggles["trails"]:
                trails.clear()
            print(f"Trails: {'ON' if toggles['trails'] else 'OFF'}")
        if k == ord("j"):
            toggles["jungle"] ^= True
            print(f"Jungle: {'ON' if toggles['jungle'] else 'OFF'}")
        if k == ord("g"):
            toggles["grid"] ^= True
            print(f"Grid: {'ON' if toggles['grid'] else 'OFF'}")
        if k == ord("l"):
            toggles["labels"] ^= True
            print(f"Labels: {'ON' if toggles['labels'] else 'OFF'}")
        if k == ord("h"):
            toggles["help"] ^= True
            print(f"Help: {'ON' if toggles['help'] else 'OFF'}")
        if k == ord("c"):
            toggles["contacts"] ^= True
            print(f"Contacts: {'ON' if toggles['contacts'] else 'OFF'}")

        if not paused:
            frame_idx += 1

    # ── Cleanup ──
    cv2.destroyAllWindows()
    print(f"\n✅ Done — replayed {min(frame_idx, num_frames)} frames")
    print(f"   File: {parquet_path}")


if __name__ == "__main__":
    main()

"""
MLBB Vision — Coordinate Grid Tool + Hero Status Overlay
Tampilkan grid, region hero_panel, dan status deteksi hero di overlay kiri.
"""

from __future__ import annotations
import sys, os, argparse, logging, yaml
from typing import Any
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.core import layout
from vision.core.cropper import crop_region
from vision.ocr.reader import OCRReader
from vision.detectors import HPDetector, ManaDetector, LevelDetector, GoldDetector
from vision.detectors import SkillsDetector
from vision.matcher.template import TemplateMatcher
import json

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("debug_vision")
log.setLevel(logging.INFO)


# ── Detector Manager ──────────────────────────────────────────────────

class DetectorManager:
    """Run all hero_panel detectors on a frame and collect status."""

    def __init__(self):
        self.ocr = OCRReader(use_paddle=False)
        self.hp_det = HPDetector(self.ocr)
        self.mana_det = ManaDetector(self.ocr)
        self.level_det = LevelDetector(self.ocr)
        self.gold_det = GoldDetector(self.ocr)
        self.skills_det = SkillsDetector(self.ocr)
        self._cached_hero_name: str | None = None

        # ── Load hero database ──
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._hero_db: dict[str, dict] = {}
        db_path = os.path.join(base, "assets", "databases", "heroes.json")
        try:
            with open(db_path) as f:
                for hero in json.load(f):
                    self._hero_db[hero["key"]] = hero
            log.info("Loaded %d heroes from database", len(self._hero_db))
        except Exception as e:
            log.warning("Failed to load heroes.json: %s", e)

        # ── Load hero portrait templates (resize ke ukuran portrait di game) ──
        pt_w, pt_h = 110, 100  # ukuran portrait region dari layout
        hero_templates: dict[str, np.ndarray] = {}
        heroes_path = os.path.join(base, "assets", "heroes")
        try:
            for fname in sorted(os.listdir(heroes_path)):
                if fname.endswith(".png"):
                    stem = fname[:-4]
                    img = cv2.imread(os.path.join(heroes_path, fname))
                    if img is not None:
                        img = cv2.resize(img, (pt_w, pt_h), interpolation=cv2.INTER_AREA)
                        hero_templates[stem] = img
            log.info("Loaded %d hero templates (%dx%d)", len(hero_templates), pt_w, pt_h)
        except Exception as e:
            log.warning("Failed to load hero templates: %s", e)

        self._hero_matcher = TemplateMatcher(threshold=0.35, templates=hero_templates)

        # ── Load item database ──
        self._item_db: dict[str, dict] = {}
        item_db_path = os.path.join(base, "assets", "databases", "items.json")
        try:
            with open(item_db_path) as f:
                for item in json.load(f):
                    self._item_db[item["key"]] = item
            log.info("Loaded %d items from database", len(self._item_db))
        except Exception as e:
            log.warning("Failed to load items.json: %s", e)

        # ── Load item templates (resize ke 59x59 = ukuran item slot) ──
        self._item_matcher = None
        item_templates: dict[str, np.ndarray] = {}
        items_path = os.path.join(base, "assets", "items")
        try:
            for fname in sorted(os.listdir(items_path)):
                if fname.endswith(".png"):
                    stem = fname[:-4]
                    img = cv2.imread(os.path.join(items_path, fname))
                    if img is not None:
                        img = cv2.resize(img, (59, 59), interpolation=cv2.INTER_AREA)
                        item_templates[stem] = img
            log.info("Loaded %d item templates (59x59)", len(item_templates))
        except Exception as e:
            log.warning("Failed to load item templates: %s", e)
        if item_templates:
            self._item_matcher = TemplateMatcher(threshold=0.25, templates=item_templates)

        log.info("Detectors initialized")

        # ── Cooldown tracker ──
        self._cd_timers: dict[str, float] = {}  # skill_name -> end_timestamp
        self._cd_data: dict[str, float] = {}  # skill_name -> base_cooldown_seconds
        self._cdr = 0.0  # cooldown reduction (future: dari item)
        self._cd_seen_ready: set[str] = set()  # skill sudah pernah terlihat ready
        self._cd_ref_brightness: dict[str, float] = {}  # reference brightness skill siap

    def detect(self, frame: np.ndarray, video_time: float = 0) -> dict[str, Any]:
        """Run all detectors on a frame and return status dict."""
        status: dict[str, Any] = {}

        # ── Hero name (template matching portrait, dicache) ──
        if self._cached_hero_name is None:
            portrait_img = crop_region(frame, "hero_panel", "portrait")
            if portrait_img is not None and portrait_img.size:
                match = self._hero_matcher.match(portrait_img)
                if match and match.success and match.label:
                    entry = self._hero_db.get(match.label)
                    if entry:
                        self._cached_hero_name = entry.get("name", match.label)
                    else:
                        self._cached_hero_name = match.label.replace("_", " ").title()
                    log.info("Hero matched: %s (%.0f%%)", self._cached_hero_name, match.confidence * 100)
        if self._cached_hero_name:
            status["hero_name"] = self._cached_hero_name

        # ── Level (Tesseract OCR, karena PaddleOCR mati) ──
        lvl_img = crop_region(frame, "hero_panel", "level")
        if lvl_img is not None and lvl_img.size:
            gray = cv2.cvtColor(lvl_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            white_px = cv2.countNonZero(binary)
            if 5 < white_px < binary.size * 0.3 and gray.mean() < 100:
                try:
                    import pytesseract
                    inv = 255 - binary
                    big = cv2.resize(inv, (120, 120), interpolation=cv2.INTER_LINEAR)
                    text = pytesseract.image_to_string(
                        big, config='--psm 10 --oem 3 digits').strip()
                    if text and text.isdigit():
                        val = int(text)
                        if 1 <= val <= 30:
                            status["level"] = val
                            log.debug("Level = %d", val)
                except Exception:
                    pass

        # ── HP ──
        hp_img = crop_region(frame, "hero_panel", "hp_bar")
        if hp_img is not None and hp_img.size:
            result = self.hp_det.run(hp_img)
            if result and result.value is not None:
                status["hp_pct"] = result.value

        # ── Mana ──
        mana_img = crop_region(frame, "hero_panel", "mana_bar")
        if mana_img is not None and mana_img.size:
            result = self.mana_det.run(mana_img)
            if result and result.value is not None:
                status["mana_pct"] = result.value

        # ── KDA (OCR langsung) ──
        kda_img = crop_region(frame, "hero_panel", "kda")
        if kda_img is not None and kda_img.size:
            text = self.ocr.read(kda_img, hint="kda")
            if text:
                status["kda"] = text.strip()

        # ── Gold ──
        gold_img = crop_region(frame, "hero_panel", "gold")
        if gold_img is not None and gold_img.size:
            result = self.gold_det.run(gold_img)
            if result and result.value is not None:
                status["gold"] = result.value

        # ── Skills ──
        skills_status: dict[str, dict] = {}
        for skill_name in ("passive", "skill_1", "skill_2", "skill_3", "battle_spell"):
            skill_img = crop_region(frame, "hero_panel", "skills", skill_name)
            if skill_img is not None and skill_img.size:
                result = self.skills_det.run(skill_img)
                skill_info: dict[str, Any] = {}
                if result and result.value:
                    skill_info = dict(result.value)

                # Fallback kalau SkillsDetector gagal
                if not skill_info:
                    gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)
                    avg = gray.mean()
                    skill_info["ready"] = avg > 140
                    skill_info["cooldown"] = not skill_info["ready"]
                    skill_info["brightness"] = round(float(avg), 1)

                # Update cooldown: visual check (deteksi gelap) + timer kalkulasi
                self._update_skill_cooldown(skill_name, skill_img, skill_info, video_time)

                skills_status[skill_name] = skill_info
        if skills_status:
            status["skills"] = skills_status

        # ── Items (template matching + CDR) ──
        if self._item_matcher is not None:
            item_names: list[str] = []
            total_cdr = 0
            for item_slot in ("item_1", "item_2", "item_3", "item_4", "item_5", "item_6"):
                item_img = crop_region(frame, "hero_panel", "items", item_slot)
                if item_img is not None and item_img.size:
                    gray = cv2.cvtColor(item_img, cv2.COLOR_BGR2GRAY)
                    if gray.mean() < 30 or gray.std() < 28:
                        continue
                    match = self._item_matcher.match(item_img)
                    if match and match.success and match.label:
                        entry = self._item_db.get(match.label)
                        if entry:
                            name = entry.get("name", match.label)
                            total_cdr += entry.get("attributes", {}).get("cooldown_reduction", 0)
                        else:
                            name = match.label.replace("_", " ").title()
                        item_names.append(name)
                        log.debug("Item %s: %s (%.0f%%)", item_slot, name, match.confidence * 100)
            if item_names:
                status["items"] = item_names
                # Cap CDR at 40% (MLBB max dari item)
                self._cdr = min(total_cdr / 100.0, 0.40)
                log.debug("Items: %s | CDR: %.0f%%", ", ".join(item_names), total_cdr)

        return status

    # ── Cooldown helpers ────────────────────────────────────────────────

    _BATTLE_SPELL_CDS = {
        "flicker": 120, "execute": 90, "retribution": 30,
        "sprint": 105, "petrify": 90, "purify": 90,
        "aegis": 90, "inspire": 60, "vengeance": 60,
        "revitalize": 60, "flameshot": 60, "arrival": 90,
        "healing_spell": 90, "interference": 60, "weaken": 60,
        "iron_wall": 75, "track": 20,
    }

    def _check_cooldown_visual(self, skill_name: str, skill_img: np.ndarray) -> bool:
        """Deteksi cooldown: bandingkan brightness vs peak brightness."""
        gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)
        current = gray.mean()
        peak = self._cd_ref_brightness.get(skill_name)

        # Track peak brightness (reference = kondisi paling terang)
        if peak is None or current > peak:
            self._cd_ref_brightness[skill_name] = current
            peak = current

        # Peak terlalu gelap → belum pernah lihat icon terang
        # Fallback: absolute threshold
        if peak < 90:
            return current < 70

        # Peak rendah (<120) → icon naturally agak gelap
        if peak < 120:
            return current / peak < 0.55  # butuh perubahan lebih ekstrem

        # Peak normal (>120) → ratio-based detection
        if current / peak > 0.80:
            return False  # kembali ke brightness normal → ready
        return current / peak < 0.70  # lebih gelap dari 70% peak → cooldown

    def _read_cd_tesseract(self, skill_img: np.ndarray) -> int | None:
        """Baca angka cooldown dari skill icon pakai Tesseract (1x per cd cycle)."""
        if skill_img is None or skill_img.size == 0:
            return None
        try:
            import pytesseract
            h, w = skill_img.shape[:2]
            cx, cy = w // 2, h // 2
            r = 14
            center = skill_img[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
            gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

            # Coba threshold dari tinggi ke rendah
            for th in (240, 220, 200, 180):
                _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
                white = cv2.countNonZero(binary)
                if white < 3 or white > 100:
                    continue
                inv = 255 - binary
                big = cv2.resize(inv, (84, 84), interpolation=cv2.INTER_LINEAR)
                text = pytesseract.image_to_string(
                    big, config='--psm 10 --oem 3 digits').strip()
                cleaned = "".join(c for c in text if c.isdigit())
                if cleaned:
                    val = int(cleaned)
                    if 1 <= val <= 99:
                        log.debug("Tesseract CD: %ds (th=%d, white=%d)", val, th, white)
                        return val
        except Exception:
            pass
        return None

    def _get_base_cooldown(self, skill_name: str) -> float | None:
        """Cari base cooldown dari database hero atau hardcoded battle spell."""
        # Battle spell hardcoded
        if skill_name == "battle_spell":
            return 90  # default, akurat nanti kalau hero udah terdeteksi
        if skill_name in self._BATTLE_SPELL_CDS:
            return self._BATTLE_SPELL_CDS[skill_name]

        # Hero skill → cari di database
        if not self._cached_hero_name:
            return None

        # Map skill_3 → ultimate (di database pake "ultimate")
        db_key = {"skill_3": "ultimate"}.get(skill_name, skill_name)

        hero_key = self._hero_key_from_name(self._cached_hero_name)
        if not hero_key or hero_key not in self._hero_db:
            return None

        hero = self._hero_db[hero_key]
        skill_data = hero.get("skills", {}).get(db_key, {})
        levels = skill_data.get("levels", {})
        cooldowns = levels.get("cooldown", [])
        if cooldowns:
            return cooldowns[0]  # level 1 = cooldown paling panjang
        return skill_data.get("cooldown")

    def _hero_key_from_name(self, hero_name: str) -> str | None:
        """Convert hero display name → database key."""
        for key, hero in self._hero_db.items():
            if hero.get("name") == hero_name:
                return key
        return None

    def _update_skill_cooldown(self, skill_name: str, skill_img: np.ndarray | None,
                               skill_info: dict[str, Any], video_time: float = 0):
        """Update cooldown: visual check + timer kalkulasi dari database."""
        vt = video_time  # alias — waktu video, bukan wall clock
        visual_cd = self._check_cooldown_visual(skill_name, skill_img) if skill_img is not None else False
        timer_end = self._cd_timers.get(skill_name)

        if visual_cd:
            skill_info["cooldown"] = True
            skill_info["ready"] = False

            if timer_end is None or timer_end <= vt:
                # Coba Tesseract dulu (bisa baca meski skill dari awal cooldown)
                cd_sec = self._read_cd_tesseract(skill_img)
                # Fallback database cuma kalau pernah lihat skill ready
                if cd_sec is None and skill_name in self._cd_seen_ready:
                    cd_sec = self._get_base_cooldown(skill_name)
                if cd_sec:
                    self._cd_timers[skill_name] = vt + cd_sec * (1 - self._cdr)
                    log.info("CD start %s: %.0fs (CDR %.0f%%)",
                             skill_name, cd_sec, self._cdr * 100)

            remaining = self._cd_timers.get(skill_name, vt) - vt
            if remaining > 0:
                skill_info["cooldown_seconds"] = int(remaining) + 1
            return

        if timer_end and timer_end > vt:
            skill_info["cooldown"] = True
            skill_info["ready"] = False
            remaining = timer_end - vt
            skill_info["cooldown_seconds"] = int(remaining) + 1
            return

        # Betul-betul ready
        skill_info["ready"] = True
        skill_info["cooldown"] = False
        skill_info.pop("cooldown_seconds", None)
        self._cd_seen_ready.add(skill_name)


# ── Draw helpers ──────────────────────────────────────────────────────

def draw_grid(frame: np.ndarray):
    """Draw grid lines + coordinate numbers (in-place)."""
    h, w = frame.shape[:2]

    # Grid 50px
    for x in range(0, w, 50):
        c = (80, 80, 80) if x % 100 else (120, 120, 120)
        cv2.line(frame, (x, 0), (x, h), c, 1)
    for y in range(0, h, 50):
        c = (80, 80, 80) if y % 100 else (120, 120, 120)
        cv2.line(frame, (0, y), (w, y), c, 1)

    # Garis tebal tiap 500px
    fs = max(0.4, min(0.7, w / 2400))  # font scale responsive
    for x in range(0, w, 500):
        cv2.line(frame, (x, 0), (x, h), (255, 255, 100), 2)
        cv2.putText(frame, str(x), (x + 3, int(20 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
    for y in range(0, h, 500):
        cv2.line(frame, (0, y), (w, y), (255, 255, 100), 2)
        cv2.putText(frame, str(y), (3, int(y + 22 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)

    # Angka grid tiap 100px
    for x in range(100, w, 100):
        if x % 500:
            cv2.putText(frame, str(x), (x + 2, int(14 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (180, 180, 100), 1)
    for y in range(100, h, 100):
        if y % 500:
            cv2.putText(frame, str(y), (2, int(y + 14 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (180, 180, 100), 1)


def draw_region_boxes(frame: np.ndarray):
    """Draw hero_panel bounding box + sub-region boxes (in-place)."""
    hp_box = layout.bbox("hero_panel")
    if hp_box:
        x, y, w2, h2 = hp_box
        cv2.rectangle(frame, (x, y), (x + w2, y + h2), (100, 100, 255), 2)
        cv2.putText(frame, f"hero_panel [{x},{y},{w2},{h2}]", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)

    for path, reg in layout.enumerate_regions():
        if "bbox" not in reg or path == "hero_panel":
            continue
        bx, by, bw, bh = reg["bbox"]
        name = path.split(".")[-1]

        if ".skills." in path or ".items." in path:
            cx, cy = bx + bw // 2, by + bh // 2
            r = max(bw, bh) // 2
            cv2.circle(frame, (cx, cy), r, (0, 200, 200), 2)
            if ".skills." in path and name != "battle_spell":
                cv2.putText(frame, name, (cx - 12, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
            elif ".items." in path:
                cv2.putText(frame, name, (cx - 12, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        else:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 200), 1)
            cv2.putText(frame, f"{name} [{bx},{by},{bw},{bh}]", (bx, by - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)


def draw_status_overlay(frame: np.ndarray, status: dict[str, Any]):
    """Draw detection status overlay on the right side (in-place)."""
    h, w = frame.shape[:2]

    # ── Build status lines ──
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append(("═══ HERO PANEL ═══", (200, 200, 255)))

    if status.get("hero_name"):
        lines.append((f"Hero:  {status['hero_name']}", (255, 255, 255)))
    if status.get("level") is not None:
        lines.append((f"Level: {status['level']}", (255, 255, 100)))
    if status.get("hp_pct") is not None:
        pct = status["hp_pct"]
        color = (0, 255, 0) if pct > 0.5 else (0, 200, 255) if pct > 0.2 else (0, 0, 255)
        lines.append((f"HP:    {pct:.1%}", color))
    if status.get("mana_pct") is not None:
        lines.append((f"Mana:  {status['mana_pct']:.1%}", (255, 200, 100)))
    if status.get("kda"):
        lines.append((f"KDA:   {status['kda']}", (200, 255, 200)))
    if status.get("gold") is not None:
        lines.append((f"Gold:  {status['gold']}", (255, 255, 100)))

    # Skills
    skills = status.get("skills", {})
    if skills:
        lines.append(("── Skills ──", (100, 200, 255)))
        for name in ("passive", "skill_1", "skill_2", "skill_3", "battle_spell"):
            if name in skills:
                s = skills[name]
                label = name.replace("skill_", "S").replace("battle_spell", "SPELL")

                cd_sec = s.get("cooldown_seconds")
                if cd_sec is not None:
                    text = f"  {label}: {cd_sec}s"
                    color = (255, 150, 50)
                elif s.get("ready", False):
                    text = f"  {label}: READY"
                    color = (100, 255, 100)
                elif s.get("cooldown", False):
                    text = f"  {label}: ⏳"
                    color = (255, 100, 100)
                else:
                    text = f"  {label}: ?"
                    color = (200, 200, 200)

                lines.append((text, color))

    # Items
    item_list = status.get("items", [])
    if item_list:
        lines.append(("── Items ──", (100, 255, 200)))
        for name in item_list:
            lines.append((f"  {name}", (200, 255, 200)))

    # ── Draw background panel (kanan) ──
    panel_w = 320
    line_h = 22
    pad = 10
    title_h = 30
    total_h = title_h + len(lines) * line_h + pad * 2
    panel_x = w - panel_w - 10

    # Semi-transparent background (in-place, 1 copy for overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (panel_x + panel_w, total_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (panel_x, 0), (panel_x + panel_w, total_h), (80, 80, 120), 1)

    # ── Draw text ──
    y_pos = pad + title_h - 8
    for text, color in lines:
        cv2.putText(frame, text, (panel_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        y_pos += line_h


# ── Help Overlay ───────────────────────────────────────────────────────
_HELP_LINES = [
    ("── Controls ──", (200, 200, 255)),
    ("Space", "Pause / resume"),
    ("E", "Toggle layout editor"),
    ("S / Shift+S", "Save layout / Screenshot"),
    ("G", "Toggle grid"),
    ("O", "Toggle status overlay"),
    ("H", "Toggle this help"),
    ("L", "Reload layout.yaml"),
    ("R", "Restart video"),
    ("D", "Re-detect (paused)"),
    ("Z", "Debug region sizes"),
    ("Drag handles", "Move / resize regions"),
]

def draw_help_overlay(frame: np.ndarray):
    """Draw key binding help at bottom of frame."""
    h, w = frame.shape[:2]
    line_h = 20
    pad = 8
    total_h = len(_HELP_LINES) * line_h + pad * 2
    x = 10
    y = h - total_h - 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 350, y + total_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 350, y + total_h), (80, 80, 120), 1)

    # Text
    yy = y + pad + line_h - 5
    for line in _HELP_LINES:
        if isinstance(line, tuple):
            if len(line) == 2 and isinstance(line[0], str) and isinstance(line[1], tuple):
                text, color = line
                cv2.putText(frame, text, (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            else:
                key, desc = line
                cv2.putText(frame, f"  {key}:", (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
                tw = cv2.getTextSize(f"  {key}:", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
                cv2.putText(frame, desc, (x + pad + tw + 4, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        yy += line_h


# ── Layout Editor (drag / resize region boxes) ────────────────────────

class _LayoutEditor:
    """Interactive layout region editor — drag, resize, save."""
    def __init__(self):
        self.selected: str | None = None      # region path, e.g. "hero_panel.skills.skill_1"
        self.mode: str | int | None = None    # 'move' or corner index (0-3)
        self.orig_bbox: list[int] | None = None
        self.offset: tuple[int, int] | None = None  # mouse offset inside bbox for move
        self.dirty: bool = False
        self.paused: bool = False  # edit hanya saat paused
        self.edit_mode: bool = False  # toggle dengan E

    def get_region_data(self, path: str) -> dict | None:
        """Get region dict from cached layout by dot-path."""
        data = layout._LAYOUT_CACHE.get(layout._LAYOUT_PATH)
        if data is None:
            data = layout.load()
        parts = path.split(".")
        for p in parts:
            if isinstance(data, dict) and p in data:
                data = data[p]
            else:
                return None
        return data if isinstance(data, dict) else None

    def draw(self, frame: np.ndarray):
        """Draw selection highlight + resize handles."""
        if self.selected is None:
            return
        reg = self.get_region_data(self.selected)
        if reg is None or "bbox" not in reg:
            return
        bx, by, bw, bh = reg["bbox"]
        # Highlight with thick orange border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 3)
        # Resize handles at corners
        handles = [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
        for hx, hy in handles:
            cv2.circle(frame, (hx, hy), 12, (0, 255, 255), -1)
            cv2.circle(frame, (hx, hy), 12, (0, 165, 255), 3)
        # Label
        name = self.selected.split(".")[-1]
        cv2.putText(frame, f"EDIT: {name} [{bx},{by},{bw},{bh}]",
                    (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def _make_mouse_cb(editor: _LayoutEditor, fw: int, fh: int, dw: int, dh: int):
    """Create mouse callback for layout editing."""
    def _cb(event, x, y, flags, param):
        # Convert display coords → frame coords
        fx = int(x * fw / dw) if dw > 0 else x
        fy = int(y * fh / dh) if dh > 0 else y
        fx = max(0, min(fx, fw - 1))
        fy = max(0, min(fy, fh - 1))

        e = param  # _LayoutEditor instance

        if not e.paused or not e.edit_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Coba drag region yang sedang terpilih (kalo klik di dalamnya)
            if e.selected is not None:
                reg = e.get_region_data(e.selected)
                if reg and "bbox" in reg:
                    bx, by, bw, bh = reg["bbox"]
                    on_handle = any(
                        abs(fx - cx) < 14 and abs(fy - cy) < 14
                        for cx, cy in [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
                    )
                    if bx <= fx <= bx + bw and by <= fy <= by + bh:
                        if on_handle:
                            ci = next(i for i, (cx, cy) in
                                      enumerate([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
                                      if abs(fx - cx) < 14 and abs(fy - cy) < 14)
                            e.mode = ci
                        else:
                            e.mode = "move"
                        e.orig_bbox = list(reg["bbox"])
                        e.offset = (fx - bx, fy - by)
                        return
            # Cari region terdalam di bawah klik
            best: tuple[str, dict] | None = None
            for path, reg in reversed(list(layout.enumerate_regions())):
                if "bbox" not in reg:
                    continue
                bx, by, bw, bh = reg["bbox"]
                if not (bx <= fx <= bx + bw and by <= fy <= by + bh):
                    continue
                # Region ini cocok — pilih yang terdalam (nested dots count)
                if best is None or path.count(".") > best[0].count("."):
                    best = (path, reg)
            if best is not None:
                path, reg = best
                # Cek apakah kena handle corner
                bx, by, bw, bh = reg["bbox"]
                corners = [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)]
                for ci, (cx, cy) in enumerate(corners):
                    if abs(fx - cx) < 14 and abs(fy - cy) < 14:
                        e.selected = path
                        e.mode = ci
                        e.orig_bbox = list(reg["bbox"])
                        e.offset = None
                        return
                # Drag body
                e.selected = path
                e.mode = "move"
                e.orig_bbox = list(reg["bbox"])
                e.offset = (fx - bx, fy - by)
                return
            # Klik di luar semua region → deselect
            e.selected = None
            e.mode = None

        elif event == cv2.EVENT_MOUSEMOVE and e.mode is not None and e.selected:
            reg = e.get_region_data(e.selected)
            if reg is None:
                return
            if e.mode == "move":
                dx = fx - e.orig_bbox[0] - e.offset[0]
                dy = fy - e.orig_bbox[1] - e.offset[1]
                reg["bbox"] = [e.orig_bbox[0] + dx, e.orig_bbox[1] + dy,
                               e.orig_bbox[2], e.orig_bbox[3]]
            else:
                # Corner resize — hanya 1 corner yg bergerak
                ox, oy, ow, oh = e.orig_bbox
                r, b = ox + ow, oy + oh  # right, bottom
                if e.mode == 0:      # TL
                    x, y = fx, fy
                    w, h = r - fx, b - fy
                elif e.mode == 1:    # TR
                    x, y = ox, fy
                    w, h = fx - ox, b - fy
                elif e.mode == 2:    # BR
                    x, y = ox, oy
                    w, h = fx - ox, fy - oy
                else:                # BL (mode 3)
                    x, y = fx, oy
                    w, h = r - fx, fy - oy
                if w < 5:
                    w, x = 5, x if e.mode in (0, 3) else r - 5
                if h < 5:
                    h, y = 5, y if e.mode in (0, 1) else b - 5
                reg["bbox"] = [int(x), int(y), int(w), int(h)]
            e.dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if e.selected:
                reg = e.get_region_data(e.selected)
                if reg and "bbox" in reg:
                    bb = reg["bbox"]
                    if bb[2] < 5:
                        bb[2] = 5
                    if bb[3] < 5:
                        bb[3] = 5
            e.mode = None

    return _cb


def _save_layout(editor: _LayoutEditor):
    """Save cached layout back to layout.yaml."""
    if not editor.dirty:
        print("ℹ️ No changes to save")
        return
    path = layout._LAYOUT_PATH
    data = layout._LAYOUT_CACHE.get(path)
    if data is None:
        print("❌ Layout cache empty")
        return
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print(f"💾 Layout saved to {path}")
    editor.dirty = False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--resize", type=float, default=0.7)
    ap.add_argument("--overlay", action="store_true", default=True,
                    help="Tampilkan status overlay (default: True)")
    ap.add_argument("--no-overlay", action="store_false", dest="overlay")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="Speed multiplier (default: 1.5)")
    a = ap.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    vp = a.video
    if not vp:
        vd = os.path.join(BASE, "videos")
        cs = sorted([f for f in os.listdir(vd) if f.endswith(".mp4")])
        if not cs:
            print("No video found in videos/")
            sys.exit(1)
        vp = os.path.join(vd, cs[0])
        print(f"Using: {vp}")

    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = max(1, int(1000 / fps / a.speed))
    print(f"Video: {fps:.1f} fps — {a.speed:.1f}× speed ({frame_delay}ms delay)")
    dw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * a.resize)
    dh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * a.resize)

    # Init detectors
    detector_mgr = DetectorManager()

    paused = False
    show_overlay = a.overlay
    show_grid = False
    show_help = False

    cv2.namedWindow("MLBB Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MLBB Debug", dw, dh)

    # ── Layout Editor ──
    layout_editor = _LayoutEditor()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.setMouseCallback("MLBB Debug", _make_mouse_cb(layout_editor, fw, fh, dw, dh), layout_editor)
    layout_edit_mode = False  # editor mati default, tekan E untuk aktifkan

    detect_every = 10
    frame_count = 0

    clean_frame = None  # snapshot saat pause

    while True:
        if not paused:
            r, fr = cap.read()
            if not r:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # ── Detection ──
        if not paused:
            frame_count += 1
            video_time = frame_count / fps
            if frame_count == 1 or frame_count % detect_every == 0:
                status = detector_mgr.detect(fr, video_time)
                detector_mgr.latest_status = status

        # ── Drawing ── (pakai copy biar ga numpuk kalau paused)
        if paused and clean_frame is not None:
            draw_base = clean_frame.copy()
        else:
            draw_base = fr.copy()
        if show_grid:
            draw_grid(draw_base)
        draw_region_boxes(draw_base)
        if layout_edit_mode:
            layout_editor.draw(draw_base)
        if show_help:
            draw_help_overlay(draw_base)
        if show_overlay:
            draw_status_overlay(draw_base, status)
        if a.resize < 1:
            vis = cv2.resize(draw_base, (dw, dh))
        else:
            vis = draw_base
        cv2.imshow("MLBB Debug", vis)

        # ── Controls ──
        k = cv2.waitKey(frame_delay) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused ^= True
            layout_editor.paused = paused
            if paused:
                clean_frame = fr.copy()
                print("⏸ Paused")
            else:
                layout_editor.selected = None
                layout_editor.mode = None
                print("▶ Resumed")
                status = detector_mgr.detect(fr, frame_count / fps)
        if k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            detector_mgr.latest_status = {}
            detector_mgr._cd_timers.clear()
            detector_mgr._cd_seen_ready.clear()
            detector_mgr._cd_ref_brightness.clear()
            frame_count = 0
        if k == ord("o"):
            show_overlay ^= True
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        if k == ord("d"):
            # Manual re-detect on current frame
            if paused:
                status = detector_mgr.detect(fr, frame_count / fps)
                print(f"🔄 Re-detect → {status.get('hero_name', '?')}")
        if k == ord("g"):
            show_grid ^= True
            print(f"Grid: {'ON' if show_grid else 'OFF'}")
        if k == ord("l"):
            layout._LAYOUT_CACHE.clear()
            print("🔄 Layout reloaded!")
        if k == ord("s"):
            _save_layout(layout_editor)
        if k == ord("S"):  # Shift+S = screenshot
            ts = cv2.getTickCount()
            path = os.path.join(BASE, f"debug_frame_{ts}.png")
            cv2.imwrite(path, vis)
            print(f"📸 Frame saved: {path}")
        if k == ord("e"):
            layout_edit_mode ^= True
            layout_editor.edit_mode = layout_edit_mode
            if not layout_edit_mode:
                layout_editor.selected = None
                layout_editor.mode = None
                print("Layout editor: OFF")
            else:
                print("Layout editor: ON — pause video, lalu klik region")
            print(f"Layout editor: {'ON' if layout_edit_mode else 'OFF'}")
        if k == ord("h"):
            show_help ^= True
            print(f"Help: {'ON' if show_help else 'OFF'}")
        if k == ord("z"):
            # Debug: print crop results for all regions
            for path, reg in layout.enumerate_regions():
                img = crop_region(fr, *path.split("."))
                print(f"  {path}: {'✅' if img is not None and img.size > 0 else '❌'} "
                      f"size={img.shape if img is not None else 'N/A'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
MLBB Vision — Coordinate Grid Tool + Hero Status Overlay + Team Roster
Tampilkan grid, region hero_panel, status deteksi hero, dan team roster (blue/red) di overlay.
"""

from __future__ import annotations
import sys, os, argparse, logging, yaml, time
from typing import Any
import threading, queue
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Batasi penggunaan CPU thread agar laptop tidak kepanasan
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
cv2.setNumThreads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.core import layout
from vision.core.cropper import crop_region
from vision.ocr.reader import OCRReader
from vision.detectors import HPDetector, ManaDetector, LevelDetector, GoldDetector
from vision.detectors import SkillsDetector
from vision.matcher.template import TemplateMatcher
from vision.detectors.team.blue_team import BlueTeamDetector, RedTeamDetector
from vision.detectors.minimap.minimap_hero_tracker import MinimapHeroTracker
from vision.mapper.coordinate_mapper import CoordinateMapper
from vision.trackers.team_hp_tracker import TeamHPTracker, create_team_hp_tracker
from vision.exporters import GameStateExporter
from pathlib import Path
import json

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("debug_vision")
log.setLevel(logging.ERROR)


# ── Detector Manager ──────────────────────────────────────────────────

class DetectorManager:
    """Run all hero_panel detectors on a frame and collect status."""

    def __init__(self):
        self.ocr = OCRReader()
        self.hp_det = HPDetector(self.ocr)
        self.mana_det = ManaDetector(self.ocr)
        self.level_det = LevelDetector(self.ocr)
        self.gold_det = GoldDetector(self.ocr)
        self.skills_det = SkillsDetector(self.ocr)
        self._cached_hero_name: str | None = None
        self._seen_active: set[str] = set()  # skill pernah terlihat cooldown/available → unlocked

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

        # ── Load creeps/jungle database ──
        self._creep_db: dict[str, dict] = {}
        creep_db_path = os.path.join(base, "assets", "databases", "creeps.json")
        try:
            with open(creep_db_path) as f:
                for creep in json.load(f):
                    self._creep_db[creep["key"]] = creep
            log.info("Loaded %d creeps from database", len(self._creep_db))
        except Exception as e:
            log.warning("Failed to load creeps.json: %s", e)

        # ── Load hero portrait templates (resize sesuai ukuran portrait di layout) ──
        pt_region = layout.get_region("hero_panel", "portrait")
        if pt_region and "bbox" in pt_region:
            pt_w, pt_h = pt_region["bbox"][2], pt_region["bbox"][3]  # [x, y, w, h]
        else:
            pt_w, pt_h = 110, 100  # fallback
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

        self._hero_matcher = TemplateMatcher(threshold=0.40, templates=hero_templates)

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

        # ── Load spell database (cooldown data) ──
        self._spell_db: dict[str, dict] = {}
        spell_db_path = os.path.join(base, "assets", "databases", "spells.json")
        try:
            with open(spell_db_path) as f:
                for spell in json.load(f):
                    self._spell_db[spell["key"]] = spell
            log.info("Loaded %d spells from database", len(self._spell_db))
        except Exception as e:
            log.warning("Failed to load spells.json: %s", e)

        # ── Load item templates (resize sesuai ukuran item slot di layout) ──
        self._item_matcher = None
        item_templates: dict[str, np.ndarray] = {}
        item_region = layout.get_region("hero_panel", "items", "item_1")
        if item_region and "bbox" in item_region:
            iw, ih = item_region["bbox"][2], item_region["bbox"][3]
        else:
            iw, ih = 59, 59  # fallback
        items_path = os.path.join(base, "assets", "items")
        try:
            for fname in sorted(os.listdir(items_path)):
                if fname.endswith(".png"):
                    stem = fname[:-4]
                    img = cv2.imread(os.path.join(items_path, fname))
                    if img is not None:
                        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
                        item_templates[stem] = img
            log.info("Loaded %d item templates (%dx%d)", len(item_templates), iw, ih)
        except Exception as e:
            log.warning("Failed to load item templates: %s", e)
        if item_templates:
            self._item_matcher = TemplateMatcher(threshold=0.25, templates=item_templates)

        # ── Load battle spell templates (resize sesuai ukuran battle_spell di layout) ──
        self._spell_matcher = None
        spell_templates: dict[str, np.ndarray] = {}
        spell_region = layout.get_region("hero_panel", "skills", "battle_spell")
        if spell_region and "bbox" in spell_region:
            sw, sh = spell_region["bbox"][2], spell_region["bbox"][3]
        else:
            sw, sh = 59, 59  # fallback
        spells_path = os.path.join(base, "assets", "spells")
        try:
            for fname in sorted(os.listdir(spells_path)):
                if fname.endswith(".png"):
                    stem = fname[:-4]
                    # Skip recall — bukan battle spell
                    if stem == "recall":
                        continue
                    img = cv2.imread(os.path.join(spells_path, fname))
                    if img is not None:
                        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                        spell_templates[stem] = img
            log.info("Loaded %d spell templates (59x59)", len(spell_templates))
        except Exception as e:
            log.warning("Failed to load spell templates: %s", e)
        if spell_templates:
            self._spell_matcher = TemplateMatcher(threshold=0.35, templates=spell_templates)

        # ── Team Detectors (5 hero portraits from scoreboard) ──
        self.blue_team_detector = BlueTeamDetector(confidence_threshold=0.25)
        self.red_team_detector = RedTeamDetector(confidence_threshold=0.25)

        # ── Minimap Hero Tracker ──
        self.minimap_hero_tracker = MinimapHeroTracker(
            yolo_model_path="models/hero_tracker.onnx",
            max_miss_frames=30,  # dot bertahan ~2 detik tanpa deteksi
        )
        self._minimap_coord_mapper = CoordinateMapper.from_layout()

        # ── Minimap bbox (dari layout.yaml) ──
        mm_bbox = layout.bbox("map")
        if mm_bbox:
            self._mm_x, self._mm_y, self._mm_w, self._mm_h = mm_bbox
        else:
            self._mm_x, self._mm_y, self._mm_w, self._mm_h = (80, 0, 350, 340)

        # ── Cached identified battle spell (periodic retry, bukan one-shot) ──
        self._cached_spell_key: str | None = None
        self._cached_spell_cd: float | None = None
        self._last_spell_identify_time: float = -999.0
        self._spell_identify_interval: float = 5.0  # retry identifikasi setiap 5 detik

        self._spell_cd_end: float = 0.0  # timer spell (independent dari _cd_timers)

        # ── Cooldown tracker ──
        self._cd_timers: dict[str, float] = {}  # skill_name -> end_timestamp
        self._cdr = 0.0  # cooldown reduction (dari item)
        self._cd_seen_ready: set[str] = set()  # skill sudah pernah terlihat ready
        self._cd_confirm_count: dict[str, int] = {}  # skill_name -> consecutive cooldown frames
        self._cd_ready_count: dict[str, int] = {}    # skill_name -> consecutive ready frames
        self._cd_confirm_threshold: int = 3  # frames hysteresis

        # ── Background thread untuk OCR CD ──
        self._cd_ocr_executor = ThreadPoolExecutor(max_workers=2)  # dibatasi untuk hemat CPU
        self._cd_ocr_pending: dict[str, bool] = {}
        self._cd_ocr_last_time: dict[str, float] = {}
        self._cd_ocr_results: dict[str, tuple[float, int]] = {}  # skill_name -> (video_time, remaining_sec)
        self._CD_OCR_INTERVAL = 1.0

        log.info("Detectors initialized")

    def detect(self, frame: np.ndarray, video_time: float = 0, frame_idx: int | None = None) -> dict[str, Any]:
        """Run all detectors on a frame and return status dict."""
        status: dict[str, Any] = {}
        # Gunakan frame_idx dari caller, atau estimasi dari video_time
        if frame_idx is None:
            frame_idx = int(video_time * 30)

        # ── Hero name (template matching portrait) ──
        if self._cached_hero_name:
            status["hero_name"] = self._cached_hero_name
        else:
            # Coba match portrait hero panel
            portrait_img = crop_region(frame, "hero_panel", "portrait")
            if portrait_img is not None and portrait_img.size > 0:
                # Resize ke ukuran template (82x75 dari layout)
                pt_region = layout.get_region("hero_panel", "portrait")
                if pt_region and "bbox" in pt_region:
                    pt_w, pt_h = pt_region["bbox"][2], pt_region["bbox"][3]
                    if portrait_img.shape[1] != pt_w or portrait_img.shape[0] != pt_h:
                        portrait_img = cv2.resize(portrait_img, (pt_w, pt_h),
                                                  interpolation=cv2.INTER_AREA)
                match = self._hero_matcher.match(portrait_img)
                if match and match.success:
                    self._cached_hero_name = match.label
                    status["hero_name"] = match.label
                    log.info("✅ Hero identified: %s (conf=%.2f)", match.label, match.confidence)
                else:
                    status["hero_name"] = "...scanning"
                    last_log = getattr(self, '_last_scan_log', 0)
                    if video_time > 0 and video_time - last_log >= 3:
                        log.info("⏳ Hero masih scanning... (%.0fs)", video_time)
                        self._last_scan_log = video_time
            else:
                status["hero_name"] = "...scanning"

        # ── Simpan sample portrait ke .tmp/ (setelah hero teridentifikasi) ──
        if not getattr(self, '_portrait_saved', False) and self._cached_hero_name:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tmp_dir = os.path.join(base_dir, ".tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            portrait_img = crop_region(frame, "hero_panel", "portrait")
            if portrait_img is not None and portrait_img.size > 0:
                gray = cv2.cvtColor(portrait_img, cv2.COLOR_BGR2GRAY)
                if gray.mean() > 40:
                    fname = f"portrait_{self._cached_hero_name}.png"
                    cv2.imwrite(os.path.join(tmp_dir, fname), portrait_img)
                    log.info("📸 Portrait saved: .tmp/%s", fname)

                    # Simpan template pembanding dari hero yang terdeteksi
                    hero_path = os.path.join(base_dir, "assets", "heroes", f"{self._cached_hero_name}.png")
                    if os.path.exists(hero_path):
                        tmpl = cv2.imread(hero_path)
                        if tmpl is not None:
                            pt_region = layout.get_region("hero_panel", "portrait")
                            if pt_region and "bbox" in pt_region:
                                pw, ph = pt_region["bbox"][2], pt_region["bbox"][3]
                            else:
                                pw, ph = 110, 100
                            cv2.imwrite(os.path.join(tmp_dir, f"template_{self._cached_hero_name}.png"),
                                        cv2.resize(tmpl, (pw, ph)))

                    self._portrait_saved = True

        # ── Level (Tesseract OCR, cached setiap 2 detik) ──
        if not hasattr(self, '_last_lvl_ocr'):
            self._last_lvl_ocr = 0.0
            self._cached_level = None

        if self._cached_level is not None:
            status["level"] = self._cached_level

        if video_time - self._last_lvl_ocr >= 2.0 or self._cached_level is None:
            lvl_img = crop_region(frame, "hero_panel", "level")
            if lvl_img is not None and lvl_img.size:
                gray = cv2.cvtColor(lvl_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                white_px = cv2.countNonZero(binary)
                if 5 < white_px < binary.size * 0.3 and gray.mean() < 100:
                    try:
                        import pytesseract
                        self._last_lvl_ocr = video_time
                        inv = 255 - binary
                        big = cv2.resize(inv, (120, 120), interpolation=cv2.INTER_LINEAR)
                        text = pytesseract.image_to_string(
                            big, config='--psm 10 --oem 3 digits').strip()
                        if text and text.isdigit():
                            val = int(text)
                            if 1 <= val <= 30:
                                self._cached_level = val
                                status["level"] = val
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

        # ── Blue Team Detection (scoreboard portraits) ──
        # Scan every 3 seconds until all 5 heroes found
        if not hasattr(self, '_blue_team_scanned'):
            self._blue_team_scanned = False
            self._blue_team_last_scan = 0.0
            self._blue_team_heroes = []  # list of {"name": str, "slot": int, "confidence": float}
            log.info("🔍 Blue team scanner initialized (scan every 3s video time)")

        if not self._blue_team_scanned and video_time - self._blue_team_last_scan >= 3.0:
            self._blue_team_last_scan = video_time
            log.info("🔍 Blue team scan at video_time=%.1fs (frame=%d)", video_time, int(video_time * 30))
            # Run blue team detector on full frame
            blue_result = self.blue_team_detector.detect(frame)
            if blue_result and blue_result.value:
                heroes = blue_result.value.get("heroes", [])
                detected = [h for h in heroes if h.get("hero_name")]
                log.info("  Blue team raw result: %d heroes total, %d matched",
                         len(heroes), len(detected))
                for h in heroes:
                    if h.get("hero_name"):
                        log.info("  ✅ Slot %d: %s (conf=%.2f)", h["slot"], h["hero_name"], h["confidence"])
                    else:
                        log.info("  ❌ Slot %d: NO MATCH (conf=%.2f)", h["slot"], h["confidence"])
                for h in detected:
                    # Merge with existing (keep highest confidence per slot)
                    existing = next((x for x in self._blue_team_heroes if x["slot"] == h["slot"]), None)
                    if not existing or h["confidence"] > existing["confidence"]:
                        if existing:
                            self._blue_team_heroes.remove(existing)
                        self._blue_team_heroes.append({
                            "name": h["hero_name"],
                            "slot": h["slot"],
                            "confidence": h["confidence"],
                        })
                log.info("Blue team scan result: %d/5 heroes found", len(self._blue_team_heroes))
                if len(self._blue_team_heroes) >= 5:
                    self._blue_team_scanned = True
                    log.info("✅ Blue team complete: %s",
                             ", ".join([f"{h['name']}(slot{h['slot']})" for h in self._blue_team_heroes]))
            else:
                log.warning("  Blue team detector returned no result (image size=%s)", frame.shape[:2] if frame is not None else "None")

        # Add blue team heroes to status for overlay
        status["blue_team_heroes"] = self._blue_team_heroes
        status["blue_team_complete"] = self._blue_team_scanned

        # ── Red Team Detection ──
        if not hasattr(self, '_red_team_scanned'):
            self._red_team_scanned = False
            self._red_team_last_scan = 0.0
            self._red_team_heroes = []
            log.info("🔍 Red team scanner initialized")

        if not self._red_team_scanned and video_time - self._red_team_last_scan >= 3.0:
            self._red_team_last_scan = video_time
            log.info("🔍 Red team scan at video_time=%.1fs", video_time)
            red_result = self.red_team_detector.detect(frame)
            if red_result and red_result.value:
                heroes = red_result.value.get("heroes", [])
                detected = [h for h in heroes if h.get("hero_name")]
                for h in heroes:
                    if h.get("hero_name"):
                        log.info("  🟥 Slot %d: %s (conf=%.2f)", h["slot"], h["hero_name"], h["confidence"])
                    else:
                        log.info("  🟥 Slot %d: NO MATCH (conf=%.2f)", h["slot"], h["confidence"])
                for h in detected:
                    existing = next((x for x in self._red_team_heroes if x["slot"] == h["slot"]), None)
                    if not existing or h["confidence"] > existing["confidence"]:
                        if existing:
                            self._red_team_heroes.remove(existing)
                        self._red_team_heroes.append({
                            "name": h["hero_name"],
                            "slot": h["slot"],
                            "confidence": h["confidence"],
                        })
                log.info("Red team scan result: %d/5 heroes found", len(self._red_team_heroes))
                if len(self._red_team_heroes) >= 5:
                    self._red_team_scanned = True
                    log.info("✅ Red team complete: %s",
                             ", ".join([f"{h['name']}" for h in self._red_team_heroes]))

        status["red_team_heroes"] = self._red_team_heroes
        status["red_team_complete"] = self._red_team_scanned

        # ── Minimap Hero Tracking ──
        # Pass roster ke tracker jika kedua team sudah complete
        if self._blue_team_scanned and self._red_team_scanned:
            if not getattr(self, '_minimap_roster_set', False):
                blue_names = [h["name"] for h in self._blue_team_heroes if h.get("name")]
                red_names = [h["name"] for h in self._red_team_heroes if h.get("name")]
                if len(blue_names) == 5 and len(red_names) == 5:
                    self.minimap_hero_tracker.set_roster(blue_names, red_names)
                    self._minimap_roster_set = True
                    log.info("✅ Minimap tracker roster set: %d blue + %d red heroes",
                             len(blue_names), len(red_names))

                    # ── Crop hero portraits dari scoreboard sebagai template ──
                    portraits = {}
                    for team_key in ("blue_team", "red_team"):
                        for i in range(1, 6):
                            pt_key = f"portrait_hero_{i}"
                            pt_cfg = layout.get_region(team_key, pt_key)
                            if pt_cfg and "bbox" in pt_cfg:
                                bx, by, bw, bh = pt_cfg["bbox"]
                                if 0 <= by < frame.shape[0] and 0 <= bx < frame.shape[1]:
                                    crop = frame[by:by+bh, bx:bx+bw]
                                    if crop.size > 0:
                                        # Map to hero name
                                        hero_list = self._blue_team_heroes if team_key == "blue_team" else self._red_team_heroes
                                        hero_entry = next((h for h in hero_list if h["slot"] == i), None)
                                        if hero_entry and hero_entry.get("name"):
                                            portraits[hero_entry["name"]] = crop
                    if portraits:
                        self.minimap_hero_tracker.set_portrait_crops(portraits)
                        log.info("✅ Set %d hero portrait crops as templates", len(portraits))

        # ── Items + CDR (SEBELUM skills, biar _cdr up-to-date) ──
        self._cdr = 0.0
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
            if item_names:
                status["items"] = item_names
                self._cdr = min(total_cdr / 100.0, 0.40)
                if total_cdr > 0:
                    log.info("Items: %s | CDR: %d%%", ", ".join(item_names[:4]), round(self._cdr * 100))
        status["cdr_pct"] = round(self._cdr * 100, 0)

        # ── Skills ──
        skills_status: dict[str, dict] = {}
        for skill_name in ("passive", "skill_1", "skill_2", "skill_3", "skill_4", "battle_spell"):
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

                # ── Battle spell identification (periodic retry, bukan one-shot) ──
                if skill_name == "battle_spell" and self._spell_matcher is not None:
                    should_identify = (
                        self._cached_spell_key is None or
                        video_time - self._last_spell_identify_time >= self._spell_identify_interval
                    )
                    if should_identify:
                        gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)
                        if gray.mean() > 30:  # skip if completely dark
                            match = self._spell_matcher.match(skill_img)
                            if match and match.success and match.label:
                                target_label = match.label
                                self._cached_spell_key = target_label
                                # Cari CD dari spell database dulu, fallback ke hardcoded dict
                                cd = None
                                spell_entry = self._spell_db.get(target_label)
                                if spell_entry and spell_entry.get("cooldown"):
                                    cd = spell_entry["cooldown"]
                                if cd is None:
                                    cd = self._BATTLE_SPELL_CDS.get(target_label)
                                if cd:
                                    self._cached_spell_cd = float(cd)
                                self._last_spell_identify_time = video_time
                                log.info("Battle spell: %s (CD=%ss, conf=%.0f%%)",
                                         target_label, cd or "?", match.confidence * 100)
                    if self._cached_spell_key:
                        skill_info["spell_name"] = self._cached_spell_key

                # ═══ Skills dikelola oleh fast detection di main thread ═══
                pass
        # skills_status tidak diset ke status — fast detection yang handle

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

    def _read_cd_tesseract(self, skill_img: np.ndarray, max_cd: int = 120) -> int | None:
        """Baca angka cooldown dari skill icon pakai Tesseract.

        Full image 59x59 langsung — tanpa crop, tanpa shifting.
        Hanya mencoba jika mean brightness < 80 (skill cooldown).
        """
        if skill_img is None or skill_img.size == 0:
            return None

        gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)
        if gray.mean() > 80:
            return None

        try:
            import pytesseract
            for th in (200, 180, 160, 140, 130, 120, 110, 100):
                _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
                wpx = cv2.countNonZero(binary)
                if wpx < 5 or wpx > gray.size * 0.2:
                    continue
                big = cv2.resize(binary, (140, 140), interpolation=cv2.INTER_LINEAR)
                text = pytesseract.image_to_string(
                    big, config='--psm 10 --oem 3 digits').strip()
                cleaned = "".join(c for c in text if c.isdigit())
                if cleaned:
                    val = int(cleaned)
                    if 1 <= val <= max_cd:
                        return val
        except Exception:
            pass
        return None

    
    
    def _hero_key_from_name(self, hero_name: str) -> str | None:
        """Convert hero display name → database key."""
        for key, hero in self._hero_db.items():
            if hero.get("name") == hero_name:
                return key
        return None

    
    def _update_skill_cooldown(self, skill_name: str, skill_img: np.ndarray | None,
                               skill_info: dict[str, Any], video_time: float = 0):
        """
        Update skill state — cuma hysteresis anti flicker. Gak ada level gate / timer / CDR.

        Hanya hysteresis anti flicker. Model CNN sumber kebenaran.
        """
        # ══ Track skill yg pernah dipake ══
        if skill_info.get("cooldown", False):
            self._seen_active.add(skill_name)

        # ══ UNAVAILABLE flag: skill blm pernah cooldown ══
        hero_lvl = getattr(self, '_cached_level', None) or 1
        skill_info["unavailable"] = (skill_name not in self._seen_active 
                                     and hero_lvl < 4 
                                     and skill_name != "battle_spell")

        # ══ Model output langsung — gak perlu hysteresis ══
        is_cd = skill_info.get("cooldown", False)
        is_ready = skill_info.get("available", False) or skill_info.get("ready", False)
        if is_ready:
            skill_info["ready"] = True
            skill_info["cooldown"] = False
        elif is_cd:
            skill_info["cooldown"] = True
            skill_info["ready"] = False
        skill_info.pop("remaining_cd", None)


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


# ── Global debug toggle (toggled by 'M' key) ──
_show_mm_debug = False


def draw_minimap_debug(frame: np.ndarray, status: dict[str, Any],
                        mm_x: int = 80, mm_y: int = 0,
                        mm_w: int = 350, mm_h: int = 340):
    """
    Draw template matching debug overlay on the minimap.

    Shows:
      - Blue/red mask overlay for detected positions
      - Hero names and confidence at match locations
      - Match confidence score
    """
    debug = status.get("_mm_debug")
    if not debug:
        return

    blue_mask = debug.get("blue_mask")
    red_mask = debug.get("red_mask")
    blue_dots = debug.get("blue_dots", [])
    red_dots = debug.get("red_dots", [])
    all_circles = debug.get("blue_all_cnt", [])  # (cx, cy, r) from HoughCircles

    # ── Draw ALL HoughCircles (yellow outline) ──
    for c in all_circles:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            dx, dy, r = c[:3]
            px, py = mm_x + dx, mm_y + dy
            cv2.circle(frame, (px, py), int(r), (0, 255, 255), 1)

    # ── Overlay blue mask ──
    if blue_mask is not None:
        blue_overlay = np.zeros_like(frame, dtype=np.uint8)
        mask_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (mm_w, mm_h))
        blue_overlay[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w] = (255, 100, 50) & mask_bgr
        cv2.addWeighted(blue_overlay, 0.3, frame, 0.7, 0, frame)

    # ── Overlay red mask ──
    if red_mask is not None:
        red_overlay = np.zeros_like(frame, dtype=np.uint8)
        mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (mm_w, mm_h))
        red_overlay[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w] = (30, 30, 200) & mask_bgr
        cv2.addWeighted(red_overlay, 0.3, frame, 0.7, 0, frame)

    # ── Draw confirmed hero dots ──
    for dx, dy in blue_dots:
        px, py = mm_x + dx, mm_y + dy
        cv2.circle(frame, (px, py), 5, (255, 200, 0), -1)
        cv2.circle(frame, (px, py), 9, (255, 200, 0), 2)
    for dx, dy in red_dots:
        px, py = mm_x + dx, mm_y + dy
        cv2.circle(frame, (px, py), 5, (0, 100, 255), -1)
        cv2.circle(frame, (px, py), 9, (0, 100, 255), 2)

    # ── Draw jungle dots (green) ──
    jungle_dots = debug.get("jungle_dots", [])
    for dx, dy in jungle_dots:
        px, py = mm_x + dx, mm_y + dy
        cv2.circle(frame, (px, py), 5, (0, 200, 0), -1)
        cv2.circle(frame, (px, py), 9, (0, 200, 0), 2)

    # Label
    cv2.putText(frame, "HOUGH CIRCLES", (mm_x + 2, mm_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"B:{len(blue_dots)} R:{len(red_dots)} J:{len(jungle_dots)} C:{len(all_circles)}",
                (mm_x + 2, mm_y + mm_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_minimap_heroes(frame: np.ndarray, status: dict[str, Any],
                         mm_x: int = 80, mm_y: int = 0,
                         mm_w: int = 350, mm_h: int = 340):
    """Draw hero dots + names on the minimap overlay (alive only)."""
    heroes = status.get("minimap_heroes", [])
    if not heroes:
        return

    fs = 0.55
    thick_circle = 1
    thick_text = 1
    dot_r = 6
    glow_r = 11

    for entry in heroes:
        px = mm_x + entry.get("pixel_x", 0)
        py = mm_y + entry.get("pixel_y", 0)
        team = entry.get("team", "blue")
        name = entry.get("name", "?")
        conf = entry.get("confidence", 0)

        if entry.get("pixel_x", 0) == 0 and entry.get("pixel_y", 0) == 0:
            continue
        if px < mm_x or px > mm_x + mm_w or py < mm_y or py > mm_y + mm_h:
            continue
        if team in ("blue", "hero"):
            outer_color = (255, 140, 60)
            fill_color = (220, 80, 30)
        elif team == "red":
            outer_color = (60, 60, 255)
            fill_color = (30, 30, 200)
        elif team == "jungle":
            outer_color = (60, 220, 60)
            fill_color = (30, 180, 30)
        else:
            outer_color = (180, 180, 180)
            fill_color = (120, 120, 120)

        # Dot + glow (selalu digambar meskipun status unknown)
        cv2.circle(frame, (px, py), glow_r, outer_color, thick_circle)
        cv2.circle(frame, (px, py), dot_r, fill_color, -1)
        cv2.circle(frame, (px, py), max(2, dot_r // 2), (255, 255, 255), -1)

        # Label teks — tampilkan nama hero atau "?" jika unknown
        if team == "jungle":
            label = name if (name and not name.startswith("jungle")) else "jungle"
        else:
            label = "?" if (not name or name.startswith("unknown")) else name[:12]
        extra = f" ({conf:.0%})" if conf < 0.8 else ""
        text = f"{label}{extra}"

        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick_text)
        lx = px + dot_r + 5
        ly = py + th // 3
        pad = 5

        cv2.rectangle(frame, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                      outer_color, 1)
        cv2.putText(frame, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), thick_text, cv2.LINE_AA)


def draw_minimap_hero_overlay(frame: np.ndarray, status: dict[str, Any]):
    """
    Draw minimap hero tracking info panel at bottom-left of video.
    Always shown when overlay is on.
    """
    fh, fw = frame.shape[:2]
    mm_heroes = status.get("minimap_heroes", [])

    # Build lines
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append(("🗺 MINIMAP HEROES", (100, 255, 200)))

    # Always show roster status
    blue_complete = status.get("blue_team_complete", False)
    red_complete = status.get("red_team_complete", False)
    roster_status = f"roster: B{'✓' if blue_complete else '...'}/R{'✓' if red_complete else '...'}"
    lines.append((f"  {roster_status}", (180, 180, 180)))

    if mm_heroes:
        for team_label, team_color in [("BLUE", (100, 200, 255)), ("RED", (100, 100, 255))]:
            team_heroes = [e for e in mm_heroes if e["team"] == team_label.lower()]
            if team_heroes:
                lines.append((f"  [{team_label}] {len(team_heroes)} visible", team_color))
                for entry in sorted(team_heroes, key=lambda x: x.get("name") or ""):
                    name = entry.get("name") or "?"
                    gx, gy = entry.get("game_x"), entry.get("game_y")
                    # Use region from regions.json (RegionMapper)
                    region = entry.get("region") or ""
                    conf = entry.get("confidence", 0)

                    conf_color = (100, 255, 100) if conf > 0.8 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)

                    if gx is not None and gy is not None:
                        text = f"  {name}: ({gx:.0f}, {gy:.0f})"
                    else:
                        nxy = f"({entry['norm_x']:.2f}, {entry['norm_y']:.2f})"
                        text = f"  {name}: {nxy}"

                    if region:
                        text += f" @{region}"

                    lines.append((text, conf_color))

        # Jungle objectives
        jungle_heroes = [e for e in mm_heroes if e["team"] == "jungle"]
        if jungle_heroes:
            lines.append((f"  [JUNGLE] {len(jungle_heroes)} visible", (100, 255, 100)))
            for entry in jungle_heroes:
                c_name = entry.get("name") or "jungle"
                if c_name.startswith("jungle_"):
                    c_name = "jungle"
                gx, gy = entry.get("game_x"), entry.get("game_y")
                region = entry.get("region") or ""
                conf = entry.get("confidence", 0)
                if gx is not None and gy is not None:
                    text = f"  {c_name}: ({gx:.0f}, {gy:.0f})"
                else:
                    text = f"  {c_name}: ({entry['norm_x']:.2f}, {entry['norm_y']:.2f})"
                if region:
                    text += f" @{region}"
                lines.append((text, (100, 255, 100)))
    else:
        lines.append(("  (waiting for detection)", (180, 180, 180)))

    # ── Shortcut bar ──
    lines.append(("─── Shortcuts ───────", (100, 100, 120)))
    lines.append(("Space:Pause  G:Grid  O:Overlay  M:Minimap", (130, 130, 130)))
    lines.append(("E:Editor  S:Save  R:Restart  P/p:Export  H:Help  Q:Quit", (130, 130, 130)))
    lines.append(("c:CropSkills  C:Auto  l:Locked  L:LockedAuto  -:Slower  =:Faster", (130, 130, 130)))

    # ── Draw panel ──
    line_h = 24
    pad = 10
    max_tw = 0
    for text, _ in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if tw > max_tw:
            max_tw = tw
    panel_w = max(440, max_tw + pad * 2 + 20)
    total_h = len(lines) * line_h + pad * 2

    panel_x = 10
    panel_y = fh - total_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + total_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + total_h), (80, 80, 120), 1)

    y_pos = panel_y + pad + line_h - 5
    for text, color in lines:
        cv2.putText(frame, text, (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_pos += line_h


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

    # Speed indicator
    spd = status.get("_speed", 1.0)
    spd_color = (100, 255, 100) if spd >= 1.0 else (200, 200, 100)
    lines.append((f"⏵ Speed: {spd:.2f}×", spd_color))

    lines.append(("----- HERO PANEL -----", (200, 200, 255)))

    # Export status indicator
    if status.get("_export_enabled"):
        cnt = status.get("_export_count", 0)
        lines.append((f"📊 PARQUET: ON ({cnt} frames)", (100, 255, 100)))
    elif status.get("_export_count", 0) > 0:
        lines.append((f"📊 PARQUET: OFF (buffered)", (200, 200, 100)))
    else:
        lines.append(("📊 PARQUET: OFF", (150, 150, 150)))

    # Skill dataset collection status
    sc = status.get("_skill_collect", False)
    sc_cnt = status.get("_skill_collect_count", 0)
    locked = status.get("_skill_collect_locked", False)
    if locked:
        lines.append((f"🔒 LOCKED AUTO: ON ({sc_cnt} frames)", (255, 200, 100)))
    elif sc:
        lines.append((f"💾 SKILL DATA: ON ({sc_cnt} frames)", (100, 255, 100)))
    elif sc_cnt > 0:
        lines.append((f"💾 SKILL DATA: OFF ({sc_cnt} saved)", (200, 200, 100)))
    else:
        lines.append(("💾 SKILL DATA: OFF", (150, 150, 150)))

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
    cdr = status.get("cdr_pct")
    if cdr is not None and cdr > 0:
        lines.append((f"CDR:   {cdr:.0f}%", (100, 200, 255)))

    # Skills
    skills = status.get("skills", {})
    if skills:
        lines.append(("____ Skills ____", (100, 200, 255)))
        for name in ("passive", "skill_1", "skill_2", "skill_3", "skill_4", "battle_spell"):
            if name not in skills:
                continue
            s = skills[name]
            if name == "battle_spell" and s.get("spell_name"):
                label = f"SPELL({s['spell_name']})"
            else:
                label = name.replace("skill_", "S").replace("battle_spell", "SPELL")

            if s.get("locked", False):
                text = f"  {label}: LOCKED"
                color = (100, 100, 100)
            elif s.get("unavailable", False):
                text = f"  {label}: UNAVAILABLE"
                color = (150, 150, 150)
            elif s.get("ready", False):
                text = f"  {label}: READY"
                color = (100, 255, 100)
            elif s.get("cooldown", False):
                remaining = s.get("remaining_cd")
                if remaining is not None and remaining > 0:
                    text = f"  {label}: COOLDOWN({remaining:.0f}s)"
                else:
                    text = f"  {label}: COOLDOWN"
                color = (255, 100, 100)
            else:
                text = f"  {label}: --"
                color = (200, 200, 200)

            lines.append((text, color))

    # Items
    item_list = status.get("items", [])
    if item_list:
        lines.append(("____ Items ____", (100, 255, 200)))
        for name in item_list:
            lines.append((f"  {name}", (200, 255, 200)))

    # ── Blue Team (Scoreboard) ──
    blue_heroes = status.get("blue_team_heroes", [])
    blue_complete = status.get("blue_team_complete", False)
    if blue_heroes:
        lines.append(("____ TIM BIRU (ALLY) ____", (100, 200, 255)))
        for h in sorted(blue_heroes, key=lambda x: x["slot"]):
            hp_text = ""
            hp = h.get("hp_pct")
            if hp is not None:
                hp_text = f" HP:{hp:.0%}"
                hp_color = (0, 255, 0) if hp > 0.5 else (0, 200, 255) if hp > 0.2 else (0, 0, 255)
            else:
                hp_color = (180, 180, 180)
            conf_color = (100, 255, 100) if h["confidence"] > 0.6 else (0, 255, 255)
            lines.append((f"  [{h['slot']}] {h['name']}{hp_text}", conf_color if not hp_text else hp_color))
    elif not blue_complete:
        lines.append(("____ TIM BIRU (ALLY) ____", (100, 200, 255)))
        lines.append((f"  Scanning... ({len(blue_heroes)}/5)", (180, 180, 180)))

    # ── Red Team (Scoreboard) ──
    red_heroes = status.get("red_team_heroes", [])
    red_complete = status.get("red_team_complete", False)
    if red_heroes:
        lines.append(("____ TIM MERAH (ENEMY) ____", (100, 100, 255)))
        for h in sorted(red_heroes, key=lambda x: x["slot"]):
            hp_text = ""
            hp = h.get("hp_pct")
            if hp is not None:
                hp_text = f" HP:{hp:.0%}"
            conf_color = (100, 255, 100) if h["confidence"] > 0.6 else (0, 255, 255)
            hp_color = (0, 255, 0) if (hp or 0) > 0.5 else (0, 200, 255) if (hp or 0) > 0.2 else (0, 0, 255)
            lines.append((f"  [{h['slot']}] {h['name']}{hp_text}", conf_color if not hp_text else hp_color))
    elif not red_complete:
        lines.append(("____ TIM MERAH (ENEMY) ____", (100, 100, 255)))
        lines.append((f"  Scanning... ({len(red_heroes)}/5)", (180, 180, 180)))

    # ── Draw background panel (kanan) ──
    panel_w = 380
    line_h = 28
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
                    0.6, color, 2, cv2.LINE_AA)
        y_pos += line_h


# ── Help Overlay ───────────────────────────────────────────────────────
_HELP_LINES = [
    ("____ Controls ____", (200, 200, 255)),
    ("Space", "Pause / resume"),
    ("E", "Toggle layout editor"),
    ("S / Shift+S", "Save layout / Screenshot"),
    ("G", "Toggle grid"),
    ("O", "Toggle status overlay"),
    ("M", "Toggle minimap debug (HSV mask)"),
    ("H", "Toggle this help"),
    ("L", "Reload layout.yaml"),
    ("R", "Restart video"),
    ("P/p", "Toggle Parquet export"),
    ("D", "Re-detect (paused)"),
    ("Z", "Debug region sizes"),
    ("c / C", "Save skill crops / Auto-collect skill data"),
    ("l / L", "Save locked / Auto-collect locked (stun/CC)"),
    ("t", "Toggle skill mode (CNN Model / CV Legacy)"),
    ("- / =", "Slow down / Speed up (0.5× step)"),
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


# ── Skill Dataset Collection (dipanggil dari key binding) ─────────────

_auto_collect_count = 0
_auto_collect_skills = False
_auto_collect_locked = False
_locked_end_time = 0.0
_last_skills: dict = {}

def _collect_skill_samples(frame: np.ndarray, detector: DetectorManager,
                            frame_idx: int, video_time: float,
                            force_label: str | None = None):
    """Save cropped skill icons from current frame (per-hero folder).

    Labeling pake model CNN. Kalau force_label diisi, semua crop disimpan
    ke folder label tersebut (override hasil model).
    """
    global _auto_collect_count
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "trainings", "hero_skills", "dataset")

    hero_name = getattr(detector, '_cached_hero_name', None) or "unknown"
    skill_slots = ("skill_1", "skill_2", "skill_3", "battle_spell")

    saved = 0
    for skill_name in skill_slots:
        skill_img = crop_region(frame, "hero_panel", "skills", skill_name)
        if skill_img is None or skill_img.size == 0:
            continue

        if force_label:
            label = force_label
        else:
            # Dapatkan label dari detector
            result = detector.skills_det.run(skill_img)
            label = "unknown"
            if result and result.value:
                v = result.value if isinstance(result.value, dict) else {}
                if v.get("cooldown"):
                    label = "cooldown"
                elif v.get("available"):
                    label = "available"
                elif v.get("ready"):
                    label = "ready"
                elif v.get("locked"):
                    label = "locked"

        # Path: trainings/hero_skills/dataset/<hero>/<slot>/<label>/
        save_dir = os.path.join(base_dir, hero_name, skill_name, label)
        os.makedirs(save_dir, exist_ok=True)
        existing = len(os.listdir(save_dir))
        fname = f"debug_f{frame_idx:06d}_{existing:03d}.png"
        cv2.imwrite(os.path.join(save_dir, fname), skill_img)
        saved += 1

    if saved:
        _auto_collect_count += 1
        hero_info = f" ({hero_name})" if hero_name != "unknown" else ""
        label_info = f" [{force_label}]" if force_label else ""
        print(f"💾 Saved {saved} skill crops{hero_info}{label_info} (frame {frame_idx}, t={video_time:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global _auto_collect_skills, _auto_collect_count, _auto_collect_locked, _locked_end_time, _last_skills
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--overlay", action="store_true", default=True,
                    help="Tampilkan status overlay (default: True)")
    ap.add_argument("--no-overlay", action="store_false", dest="overlay")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="Speed multiplier (default: 1.5)")
    a = ap.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Reset .tmp/ di setiap startup ──
    tmp_dir = os.path.join(BASE, ".tmp")
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
        print("🧹 Cleared .tmp/")
    os.makedirs(tmp_dir, exist_ok=True)

    vd = os.path.join(BASE, "videos")
    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")
    if os.path.exists(vd):
        cs = sorted([f for f in os.listdir(vd) if f.lower().endswith(video_exts)])
    else:
        cs = []

    vp = a.video
    if vp:
        if os.path.isfile(vp):
            pass
        elif vp.isdigit() and cs and 1 <= int(vp) <= len(cs):
            vp = os.path.join(vd, cs[int(vp) - 1])
        elif os.path.isfile(os.path.join(vd, vp)):
            vp = os.path.join(vd, vp)
        else:
            print(f"❌ Video file not found: {vp}")
            sys.exit(1)
    else:
        if not cs:
            print(f"❌ No video found in {vd}")
            sys.exit(1)

        print("\n🎬 Pilih video untuk di-debug:")
        for idx, filename in enumerate(cs, 1):
            filepath = os.path.join(vd, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [{idx}] {filename} ({size_mb:.1f} MB)")

        while True:
            try:
                choice = input(f"\nMasukkan nomor video (1-{len(cs)}) [default: 1]: ").strip()
                if not choice:
                    selected_idx = 0
                else:
                    selected_idx = int(choice) - 1

                if 0 <= selected_idx < len(cs):
                    vp = os.path.join(vd, cs[selected_idx])
                    break
                else:
                    print(f"⚠️  Pilihan tidak valid. Harap masukkan nomor antara 1 dan {len(cs)}.")
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\n❌ Input dibatalkan.")
                sys.exit(1)

    print(f"\n📹 Menggunakan video: {os.path.basename(vp)}\n")

    

    # ── Layout: pilih 3 skill atau 4 skill ──

    layout_file = os.path.join(BASE, "vision", "layout.yaml")

    try:

        inp = input("🎮 Layout skills [3/4] (default: 3): ").strip()

        if inp == "4":

            src = os.path.join(BASE, "vision", "layout_4_skill.yaml")

            if os.path.exists(src):

                import shutil

                shutil.copy2(src, layout_file)

                layout._LAYOUT_CACHE.clear()

                print("   ✅ Layout: 4 skill (ultimate di skill_4)")

            else:

                print("   ⚠️  layout_4_skill.yaml tidak ditemukan, pakai 3 skill")

        else:

            src = os.path.join(BASE, "vision", "layout_3_skill.yaml")

            if os.path.exists(src):

                import shutil

                shutil.copy2(src, layout_file)

                layout._LAYOUT_CACHE.clear()

                print("   ✅ Layout: 3 skill (default)")

    except (EOFError, KeyboardInterrupt):

        print("   ✅ Layout: 3 skill (default)")

    

    # ── Parquet Exporter ──
    exporter = GameStateExporter(os.path.join(BASE, "data", "game_states"), flush_every=0)
    export_enabled = False
    match_id = Path(vp).stem  # "alpha_1", "franco_1", dll

    # Gunakan CAP_AVFOUNDATION di macOS untuk Hardware Media Engine (Apple Silicon VideoToolbox)
    cap = cv2.VideoCapture(vp, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    speed_mult = a.speed  # bisa diubah via [ (slower) / ] (faster)

    def _calc_delay(mult: float) -> int:
        return max(1, int(1000 / fps / mult))

    frame_delay = _calc_delay(speed_mult)
    print(f"Video: {fps:.1f} fps — {speed_mult:.1f}× speed ({frame_delay}ms delay) [Hardware Decoded]")

    # Window size: lebar full screen, height mengikut ratio video
    import tkinter as _tk
    _root = _tk.Tk()
    _root.withdraw()
    screen_w = _root.winfo_screenwidth()
    screen_h = _root.winfo_screenheight()
    _root.destroy()
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale to full screen width, height mengikut ratio video
    dw = screen_w
    dh = int(screen_w * video_h / video_w)

    # Init detectors
    detector_mgr = DetectorManager()

    paused = False
    show_overlay = a.overlay
    show_grid = False
    show_help = False

    # Window lebar penuh tanpa fullscreen (biar gak nutup terminal)
    cv2.namedWindow("MLBB Debug", cv2.WINDOW_NORMAL)
    # Show dummy frame dulu — di macOS, resize cuma jalan setelah imshow
    cv2.imshow("MLBB Debug", np.zeros((1, 1, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.resizeWindow("MLBB Debug", dw, dh)
    cv2.moveWindow("MLBB Debug", 0, 0)

    # ── Layout Editor ──
    layout_editor = _LayoutEditor()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.setMouseCallback("MLBB Debug", _make_mouse_cb(layout_editor, fw, fh, dw, dh), layout_editor)
    layout_edit_mode = False  # editor mati default, tekan E untuk aktifkan

    detect_every = 45        # deteksi hero panel tiap 45 frame (~1.5s pada 30fps)
    minimap_every = 1         # minimap tracking tiap frame (smooth; thread handle backpressure via queue maxsize=2)
    frame_count = 0

    clean_frame = None  # snapshot saat pause

    # ── Vision Engine (thread terpisah) ──
    vision_queue: queue.Queue = queue.Queue(maxsize=2)
    vision_status: dict[str, Any] = {}
    vision_status_lock = threading.Lock()
    vision_running = True

    def _vision_worker():
        nonlocal vision_status
        while vision_running:
            try:
                frame, fc, vt = vision_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = detector_mgr.detect(frame, vt, frame_idx=fc)
                with vision_status_lock:
                    vision_status = result
            except Exception as e:
                log.warning("Vision engine error: %s", e)

    vision_thread = threading.Thread(target=_vision_worker, daemon=True)
    vision_thread.start()

    # ── Team HP Tracker (proses sinkron di main loop tanpa thread/reader kedua) ──
    hp_tracker = create_team_hp_tracker()

    # ── Overlay: digabung ke main loop (tidak perlu thread terpisah) ──
    # Rendering overlay ringan, tidak butuh thread sendiri
    def _draw_overlay(frame, status, grid_on, edit_mode, help_on, ov_on, editor):
        try:
            if grid_on:
                draw_grid(frame)
            draw_region_boxes(frame)
            if ov_on and status:
                draw_minimap_heroes(frame, status)
                draw_minimap_hero_overlay(frame, status)
            if _show_mm_debug and status:
                draw_minimap_debug(frame, status)
            if edit_mode:
                editor.draw(frame)
            if help_on:
                draw_help_overlay(frame)
            if ov_on and status:
                draw_status_overlay(frame, status)
        except Exception as e:
            log.warning("Overlay error: %s", e)
        return frame

    # ── Minimap Hero Tracker Thread (thread terpisah untuk tracking hero) ──
    minimap_queue: queue.Queue = queue.Queue(maxsize=2)
    minimap_result: dict = {}
    minimap_result_lock = threading.Lock()
    minimap_running = True

    def _minimap_worker():
        nonlocal minimap_result
        while minimap_running:
            try:
                mm_img, fc = minimap_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                mm_heroes = detector_mgr.minimap_hero_tracker.update(mm_img, fc)
                jungle_status = detector_mgr.minimap_hero_tracker.get_jungle_status()
                heroes_data = [
                    {"name": h.name, "team": h.team,
                     "norm_x": round(h.norm_x, 3), "norm_y": round(h.norm_y, 3),
                     "pixel_x": h.pixel_x, "pixel_y": h.pixel_y,
                     "confidence": round(h.confidence, 3),
                     "game_x": round(h.game_pos.x, 1) if h.game_pos else None,
                     "game_y": round(h.game_pos.y, 1) if h.game_pos else None,
                     "lane": h.game_pos.lane if h.game_pos else None,
                     "nearest": h.game_pos.nearest_landmark if h.game_pos else None,
                     "region": h.region}
                    for h in mm_heroes
                ]
                debug_data = detector_mgr.minimap_hero_tracker.get_last_debug_data()
                with minimap_result_lock:
                    minimap_result = {
                        "heroes": heroes_data,
                        "jungle_status": jungle_status,
                        "debug": debug_data,
                    }
            except Exception as e:
                log.warning("Minimap tracker error: %s", e)

    minimap_thread = threading.Thread(target=_minimap_worker, daemon=True)
    minimap_thread.start()

    while True:
        if not paused:
            r, fr = cap.read()
            if not r:
                # Flush remaining data sebelum quit
                if export_enabled and exporter.count > 0:
                    exporter.flush_and_reset(match_id or f"match_{int(time.time())}")
                print("\n🎥 Video selesai — tekan tombol apa saja untuk keluar")
                # Tampilkan layar hitam dengan pesan "Video ended"
                end_frame = np.zeros((fh, fw, 3), dtype=np.uint8)
                cv2.putText(end_frame, "VIDEO ENDED — Press any key to exit",
                            (fw // 4, fh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 3, cv2.LINE_AA)
                vis_frame = cv2.resize(end_frame, (dw, dh))
                cv2.imshow("MLBB Debug", vis_frame)
                cv2.waitKey(0)
                break

            # Frame skipping untuk speed tinggi
            skip_frames = int(speed_mult / 2) if speed_mult >= 4 else 0
            for _ in range(skip_frames):
                r_skip = cap.grab()
                if not r_skip:
                    break
                frame_count += 1

        # ── Detection (submit ke vision thread) ──
        if not paused:
            frame_count += 1
            video_time = frame_count / fps
            if frame_count == 1 or frame_count % detect_every == 0:
                try:
                    vision_queue.put_nowait((fr.copy(), frame_count, video_time))
                except queue.Full:
                    pass
        # Baca status terbaru dari vision thread
        with vision_status_lock:
            current_status = dict(vision_status) if vision_status else {}
        # Restore skills dari fast detection (vision thread gak set skills)
        if "skills" not in current_status and _last_skills:
            current_status["skills"] = _last_skills

# ── Skills detection cepat di main thread (tiap ~0.17s, update overlay) ──
        if not paused and frame_count % 2 == 0:
            try:
                skills_status = {}
                for sn in ("skill_1", "skill_2", "skill_3", "skill_4", "battle_spell"):
                    si = crop_region(fr, "hero_panel", "skills", sn)
                    if si is not None and si.size:
                        r = detector_mgr.skills_det.run(si)
                        if r and r.value:
                            info = dict(r.value)
                            detector_mgr._update_skill_cooldown(sn, si, info, video_time)
                            if sn == "battle_spell" and detector_mgr._cached_spell_key:
                                info["spell_name"] = detector_mgr._cached_spell_key
                            skills_status[sn] = info
                if skills_status:
                    current_status["skills"] = skills_status
                    _last_skills = skills_status
            except Exception:
                pass

        # ── Minimap tracking (submit ke minimap thread, tiap minimap_every frame) ──
        if not paused and frame_count % minimap_every == 0:
            mm_img = crop_region(fr, "map")
            if mm_img is not None and mm_img.size > 0:
                try:
                    minimap_queue.put_nowait((mm_img.copy(), frame_count))
                except queue.Full:
                    pass

        # Baca hasil terbaru dari minimap thread
        with minimap_result_lock:
            mm_data = dict(minimap_result) if minimap_result else {}
        if mm_data:
            current_status["minimap_heroes"] = mm_data.get("heroes", [])
            current_status["jungle_status"] = mm_data.get("jungle_status", {})
            current_status["minimap_tracking_active"] = True
            dd = mm_data.get("debug")
            if dd:
                current_status["_mm_debug"] = {"blue_mask": dd.get("blue_mask"),
                    "red_mask": dd.get("red_mask"), "blue_dots": dd.get("blue_dots", []),
                    "red_dots": dd.get("red_dots", []),
                    "jungle_dots": dd.get("jungle_dots", [])}

        # ── Sync HP tracker → extract HP tiap 15 frame (~0.5s) ──
        if not paused and frame_count % 15 == 0:
            hp_tracker.process_frame(fr, frame_count, video_time)
        hp_data = hp_tracker.get_all_hp()
        if hp_data:
            for team, status_key in [("blue", "blue_team_heroes"), ("red", "red_team_heroes")]:
                heroes = current_status.get(status_key, [])
                if not heroes:
                    continue
                team_hp = {h["slot"]: h["hp_pct"] for h in hp_data if h["team"] == team}
                for hero in heroes:
                    slot = hero.get("slot")
                    if slot is not None and slot in team_hp and team_hp[slot] is not None:
                        hero["hp_pct"] = team_hp[slot]

        # ── Export ke Parquet (tiap frame, hanya saat recording dan tidak pause) ──
        if export_enabled and not paused:
            exporter.append(frame_count, video_time, current_status)

        # Inject export + skill dataset status for overlay display
        current_status["_export_enabled"] = export_enabled
        current_status["_export_count"] = exporter.count
        current_status["_export_total"] = exporter.total_exported
        current_status["_skill_collect"] = _auto_collect_skills or _auto_collect_locked
        current_status["_skill_collect_count"] = _auto_collect_count
        current_status["_skill_collect_locked"] = _auto_collect_locked
        current_status["_speed"] = speed_mult

        # ── Auto-collect skill dataset (jika aktif, setiap ~1 detik) ──
        # Auto-stop locked setelah 1 detik
        if _auto_collect_locked and video_time >= _locked_end_time:
            _auto_collect_locked = False
            print(f"🔒 Auto-collect locked: OFF (1s selesai)")

        if not paused and (_auto_collect_skills or _auto_collect_locked):
            _last_ac = getattr(_collect_skill_samples, '_last_time', -999.0)
            if video_time - _last_ac >= 0.5:
                _collect_skill_samples._last_time = video_time
                if _auto_collect_locked:
                    _collect_skill_samples(fr, detector_mgr, frame_count, video_time, force_label="locked")
                else:
                    _collect_skill_samples(fr, detector_mgr, frame_count, video_time)

        # ── Drawing langsung di main loop ──
        if paused and clean_frame is not None:
            draw_src = clean_frame
        else:
            draw_src = fr
        vis_frame = _draw_overlay(draw_src.copy(), current_status, show_grid, layout_edit_mode,
                                   show_help, show_overlay, layout_editor)
        cv2.imshow("MLBB Debug", vis_frame)

        # ── Controls ──
        k = cv2.waitKey(frame_delay) & 0xFF
        if k in (ord("q"), 27):
            if export_enabled and exporter.count > 0:
                exporter.flush_and_reset(match_id or f"match_{int(time.time())}")
                print(f"📊 Exported {exporter.total_exported} frames total")
            break
        if k == ord(" "):
            paused ^= True
            layout_editor.paused = paused
            if paused:
                clean_frame = fr.copy()
                print("⏸ Paused")
                # Flush parquet saat pause supaya buffer gak numpuk terus
                if export_enabled and exporter.count > 0:
                    exporter.flush(f"{match_id}_pause_{int(time.time())}")
            else:
                layout_editor.selected = None
                layout_editor.mode = None
                print("▶ Resumed")
                vision_queue.put_nowait((fr.copy(), frame_count, frame_count / fps))
        if k == ord("r"):
            # Flush parquet sebelum restart
            if export_enabled and exporter.count > 0:
                exporter.flush_and_reset(match_id or f"match_{int(time.time())}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            detector_mgr._cd_timers.clear()
            detector_mgr._cd_seen_ready.clear()
            detector_mgr._cd_confirm_count.clear()
            detector_mgr._cd_ready_count.clear()
            detector_mgr._cd_ocr_results.clear()
            detector_mgr._seen_active.clear()
            if hasattr(detector_mgr, '_cd_ref_images'):
                detector_mgr._cd_ref_images.clear()
                detector_mgr._cd_ref_brightness.clear()
            detector_mgr._cached_spell_key = None
            detector_mgr._cached_spell_cd = None
            detector_mgr._last_spell_identify_time = -999.0
            detector_mgr._spell_cd_end = 0.0
            detector_mgr._cached_hero_name = None
            # Reset team scanners
            detector_mgr._blue_team_scanned = False
            detector_mgr._blue_team_last_scan = 0.0
            detector_mgr._blue_team_heroes = []
            detector_mgr._red_team_scanned = False
            detector_mgr._red_team_last_scan = 0.0
            detector_mgr._red_team_heroes = []
            # Reset minimap tracker
            detector_mgr.minimap_hero_tracker.reset()
            detector_mgr._minimap_roster_set = False
            detector_mgr._minimap_heroes = []
            with vision_status_lock:
                vision_status.clear()
            frame_count = 0
        if k == ord("o"):
            show_overlay ^= True
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        if k == ord("d"):
            if paused:
                vision_queue.put_nowait((fr.copy(), frame_count, frame_count / fps))
                print(f"🔄 Re-detect")
        if k == ord("g"):
            show_grid ^= True
            print(f"Grid: {'ON' if show_grid else 'OFF'}")
        if k == ord("m"):
            global _show_mm_debug
            _show_mm_debug ^= True
            print(f"Minimap debug: {'ON' if _show_mm_debug else 'OFF'}")
        if k == ord("l"):
            layout._LAYOUT_CACHE.clear()
            print("🔄 Layout reloaded!")
        if k == ord("c"):
            # Save skill crops dataset (collect samples from current frame)
            _collect_skill_samples(fr, detector_mgr, frame_count, video_time)
        if k == ord("C"):
            # Toggle auto-collect mode
            _auto_collect_skills = not _auto_collect_skills
            print(f"💾 Auto-collect skills: {'ON' if _auto_collect_skills else 'OFF'}")
        if k == ord("l"):
            # Save skill crops as LOCKED (force label, bypass model)
            _collect_skill_samples(fr, detector_mgr, frame_count, video_time, force_label="locked")
        if k == ord("L"):
            # Auto-collect locked mode: 1 detik ke depan
            _auto_collect_locked = True
            _auto_collect_skills = False
            _locked_end_time = video_time + 1.0
            print(f"🔒 Auto-collect locked: ON (until t={video_time+1.0:.1f}s)")
        if k == ord("s"):
            _save_layout(layout_editor)
        if k == ord("S"):  # Shift+S = screenshot
            ts = cv2.getTickCount()
            path = os.path.join(BASE, f"debug_frame_{ts}.png")
            cv2.imwrite(path, vis)
            print(f"📸 Frame saved: {path}")
        if k in (ord("P"), ord("p")):  # P / p — toggle parquet export
            if export_enabled:
                exporter.flush_and_reset(match_id or f"match_{int(time.time())}")
            export_enabled ^= True
            print(f"📊 Parquet export: {'ON' if export_enabled else 'OFF'}"
                  f" ({exporter.count} frames in buffer)")
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
        if k == ord("="):  # Speed up (+0.5x)
            speed_mult = min(10.0, speed_mult + 0.5)
            frame_delay = _calc_delay(speed_mult)
            print(f"⏩ Speed: {speed_mult:.2f}x ({frame_delay}ms delay)")
        if k == ord("-"):  # Slow down (-0.5x)
            speed_mult = max(0.25, speed_mult - 0.5)
            frame_delay = _calc_delay(speed_mult)
            print(f"⏪ Speed: {speed_mult:.2f}x ({frame_delay}ms delay)")
        if k == ord("z"):
            # Debug: print crop results for all regions
            for path, reg in layout.enumerate_regions():
                img = crop_region(fr, *path.split("."))
                print(f"  {path}: {'✅' if img is not None and img.size > 0 else '❌'} "
                      f"size={img.shape if img is not None else 'N/A'}")

    # Cleanup
    vision_running = False
    minimap_running = False
    hp_tracker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

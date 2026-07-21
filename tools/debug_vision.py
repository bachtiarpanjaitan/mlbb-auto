"""
MLBB Vision — Coordinate Grid Tool + Hero Status Overlay
Tampilkan grid, region hero_panel, dan status deteksi hero di overlay kiri.
"""

from __future__ import annotations
import sys, os, argparse, logging
from typing import Any
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.core import layout
from vision.core.cropper import crop_region
from vision.ocr.reader import OCRReader
from vision.detectors import HPDetector, ManaDetector, LevelDetector, GoldDetector
from vision.detectors import SkillsDetector

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("debug_vision")


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
        log.info("Detectors initialized (OCR fallback mode)")

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Run all detectors on a frame and return status dict."""
        status: dict[str, Any] = {}

        # ── Hero name (OCR langsung) ──
        hero_name_img = crop_region(frame, "hero_panel", "hero_name")
        if hero_name_img is not None and hero_name_img.size:
            text = self.ocr.read(hero_name_img, hint="text")
            if text:
                status["hero_name"] = text.strip()

        # ── Level ──
        lvl_img = crop_region(frame, "hero_panel", "level")
        if lvl_img is not None and lvl_img.size:
            result = self.level_det.run(lvl_img)
            if result and result.value:
                status["level"] = result.value

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
        for skill in ("passive", "skill_1", "skill_2", "skill_3", "battle_spell"):
            skill_img = crop_region(frame, "hero_panel", "skills", skill)
            if skill_img is not None and skill_img.size:
                result = self.skills_det.run(skill_img)
                skill_info: dict[str, Any] = {}
                if result and result.value:
                    skill_info = dict(result.value)

                # OCR untuk baca angka cooldown (teks putih di atas gelap)
                cd_seconds = self._read_cooldown_number(skill_img)
                if cd_seconds is not None and cd_seconds > 0:
                    skill_info["cooldown_seconds"] = cd_seconds
                    skill_info["ready"] = False
                    skill_info["cooldown"] = True

                skills_status[skill] = skill_info
        if skills_status:
            status["skills"] = skills_status

        return status

    def _read_cooldown_number(self, skill_img: np.ndarray) -> int | None:
        """Baca angka cooldown (detik) dari skill icon via contour + template matching."""
        if skill_img is None or skill_img.size == 0:
            return None

        gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold — lebih toleran ke variasi cahaya
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -8,
        )
        # Ambil hanya piksel sangat terang (angka cooldown putih bersih)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        # Gabung: piksel yang terang DI threshold adaptif & bright
        combined = cv2.bitwise_and(binary, bright)

        bright_ratio = cv2.countNonZero(combined) / combined.size
        if bright_ratio < 0.005 or bright_ratio > 0.50:
            return None

        # Crop bagian tengah (angka cooldown di tengah icon)
        h, w = combined.shape
        margin_x, margin_y = int(w * 0.1), int(h * 0.12)
        center = combined[margin_y:h - margin_y, margin_x:w - margin_x]

        # Cari contour putih (digit = putih di background hitam)
        contours, _ = cv2.findContours(center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ch, cw = center.shape[:2]
        cx, cy = cw // 2, ch // 2
        candidates = []
        for cnt in contours:
            x, y, cw_cnt, ch_cnt = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            area_ratio = area / (cw_cnt * ch_cnt) if cw_cnt * ch_cnt > 0 else 0

            # Filter digit: ukuran wajar, area solid, di area tengah
            if (ch_cnt > 6 and ch_cnt < ch * 0.85 and
                    cw_cnt > 3 and cw_cnt < cw * 0.55 and
                    area_ratio > 0.15 and
                    abs(y + ch_cnt // 2 - cy) < ch * 0.40):
                candidates.append((x, y, cw_cnt, ch_cnt, area))

        if not candidates:
            # Fallback: coba OCR di upscale
            return self._ocr_fallback(center)

        # Urut kiri ke kanan
        candidates.sort(key=lambda c: c[0])

        # Match tiap digit
        digits = []
        for x, y, cw_cnt, ch_cnt, area in candidates:
            pad = max(2, int(min(cw_cnt, ch_cnt) * 0.2))
            y1 = max(0, y - pad)
            y2 = min(center.shape[0], y + ch_cnt + pad)
            x1 = max(0, x - pad)
            x2 = min(center.shape[1], x + cw_cnt + pad)
            roi = center[y1:y2, x1:x2]

            d = self._match_digit(roi)
            if d is not None:
                digits.append(d)

        if digits:
            num_str = "".join(str(d) for d in digits)
            try:
                val = int(num_str)
                if 1 <= val <= 300:
                    return val
            except ValueError:
                pass

        return self._ocr_fallback(center)

    def _ocr_fallback(self, center: np.ndarray) -> int | None:
        """Fallback: upscale + OCR langsung."""
        up = cv2.resize(center, (center.shape[1] * 4, center.shape[0] * 4),
                        interpolation=cv2.INTER_LINEAR)
        text = self.ocr.read(up, hint="number")
        if text:
            cleaned = "".join(ch for ch in text if ch.isdigit())
            if cleaned:
                try:
                    val = int(cleaned)
                    if 1 <= val <= 999:
                        return val
                except ValueError:
                    pass
        return None

    _DIGIT_CACHE: dict[int, np.ndarray] | None = None

    def _match_digit(self, roi: np.ndarray) -> int | None:
        """Match digit ROI terhadap template 0-9 pakai contour features."""
        if roi is None or roi.size < 20:
            return None

        # Resize ke canonical size
        roi = cv2.resize(roi, (20, 28), interpolation=cv2.INTER_NEAREST)

        # Init templates sekali
        if self._DIGIT_CACHE is None:
            self.__class__._DIGIT_CACHE = {}
            for d in range(10):
                templ = np.zeros((28, 20), dtype=np.uint8)
                cv2.putText(templ, str(d), (2, 22), cv2.FONT_HERSHEY_DUPLEX,
                            0.9, 255, 2, cv2.LINE_AA)
                self._DIGIT_CACHE[d] = templ

        # Cari template dengan match terbaik
        best_d = None
        best_score = -1
        for d, templ in self._DIGIT_CACHE.items():
            result = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_d = d

        # Threshold rendah karena font game beda dengan template
        if best_score > 0.15:
            return best_d

        # Estimasi contour features sebagai fallback
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            x, y, wr, hr = cv2.boundingRect(contours[0])
            aspect = wr / max(hr, 1)
            area_r = cv2.contourArea(contours[0]) / (wr * hr) if wr * hr > 0 else 0
            if aspect < 0.35 and area_r > 0.3:
                return 1  # digit "1" sangat ramping
        return None


# ── Draw helpers ──────────────────────────────────────────────────────

def draw_grid(frame: np.ndarray) -> np.ndarray:
    """Draw grid lines + coordinate numbers."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Grid 50px
    for x in range(0, w, 50):
        c = (80, 80, 80) if x % 100 else (120, 120, 120)
        cv2.line(out, (x, 0), (x, h), c, 1)
    for y in range(0, h, 50):
        c = (80, 80, 80) if y % 100 else (120, 120, 120)
        cv2.line(out, (0, y), (w, y), c, 1)

    # Garis tebal tiap 500px
    for x in range(0, w, 500):
        cv2.line(out, (x, 0), (x, h), (255, 255, 100), 2)
        cv2.putText(out, str(x), (x + 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for y in range(0, h, 500):
        cv2.line(out, (0, y), (w, y), (255, 255, 100), 2)
        cv2.putText(out, str(y), (5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Angka grid tiap 100px
    for x in range(100, w, 100):
        if x % 500:
            cv2.putText(out, str(x), (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 100), 1)
    for y in range(100, h, 100):
        if y % 500:
            cv2.putText(out, str(y), (3, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 100), 1)

    return out


def draw_region_boxes(frame: np.ndarray) -> np.ndarray:
    """Draw hero_panel bounding box + sub-region boxes."""
    out = frame.copy()

    hp_box = layout.bbox("hero_panel")
    if hp_box:
        x, y, w2, h2 = hp_box
        cv2.rectangle(out, (x, y), (x + w2, y + h2), (100, 100, 255), 2)
        cv2.putText(out, f"hero_panel [{x},{y},{w2},{h2}]", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)

    for path, reg in layout.enumerate_regions():
        if "bbox" not in reg or path == "hero_panel":
            continue
        bx, by, bw, bh = reg["bbox"]
        name = path.split(".")[-1]

        # Skill slots — lingkaran, label nama (kecuali battle_spell)
        if ".skills." in path:
            cx, cy = bx + bw // 2, by + bh // 2
            r = max(bw, bh) // 2
            cv2.circle(out, (cx, cy), r, (0, 200, 200), 2)
            if name != "battle_spell":
                cv2.putText(out, name, (cx - 12, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        else:
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 200, 200), 1)
            cv2.putText(out, f"{name} [{bx},{by},{bw},{bh}]", (bx, by - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)

    return out


def draw_status_overlay(frame: np.ndarray, status: dict[str, Any]) -> np.ndarray:
    """Draw detection status as text overlay on the left side."""
    out = frame.copy()
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

    # ── Draw background panel ──
    panel_w = 320
    line_h = 22
    pad = 10
    title_h = 30
    total_h = title_h + len(lines) * line_h + pad * 2

    # Semi-transparent background
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, total_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    # Border
    cv2.rectangle(out, (0, 0), (panel_w, total_h), (80, 80, 120), 1)

    # ── Draw text ──
    y_pos = pad + title_h - 8
    for text, color in lines:
        cv2.putText(out, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        y_pos += line_h

    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--resize", type=float, default=0.7)
    ap.add_argument("--overlay", action="store_true", default=True,
                    help="Tampilkan status overlay (default: True)")
    ap.add_argument("--no-overlay", action="store_false", dest="overlay")
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
    dw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * a.resize)
    dh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * a.resize)

    # Init detectors
    detector_mgr = DetectorManager()
    status: dict[str, Any] = {}

    paused = False
    show_overlay = a.overlay
    show_grid = True

    cv2.namedWindow("MLBB Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MLBB Debug", dw, dh)

    while True:
        if not paused:
            r, fr = cap.read()
            if not r:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # ── Detection ──
        if not paused:
            status = detector_mgr.detect(fr)

        # ── Drawing ──
        vis = fr.copy()
        if show_grid:
            vis = draw_grid(vis)
        vis = draw_region_boxes(vis)
        if show_overlay:
            vis = draw_status_overlay(vis, status)

        if a.resize < 1:
            vis = cv2.resize(vis, (dw, dh))

        cv2.imshow("MLBB Debug", vis)

        # ── Controls ──
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused ^= True
            if paused:
                print("⏸ Paused — running detection on current frame")
                status = detector_mgr.detect(fr)
        if k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status = {}
        if k == ord("o"):
            show_overlay ^= True
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        if k == ord("d"):
            # Manual re-detect on current frame
            if paused:
                status = detector_mgr.detect(fr)
                print(f"🔄 Re-detect → {status.get('hero_name', '?')}")
        if k == ord("g"):
            show_grid ^= True
            print(f"Grid: {'ON' if show_grid else 'OFF'}")
        if k == ord("l"):
            layout._LAYOUT_CACHE.clear()
            print("🔄 Layout reloaded!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

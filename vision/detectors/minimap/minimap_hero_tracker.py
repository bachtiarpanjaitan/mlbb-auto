"""
Minimap Hero Tracker — Melacak posisi masing-masing hero di minimap.

Flow:
  1. Deteksi roster hero dari scoreboard (blue_team + red_team)
  2. Load hero portrait dari assets/heroes/ → circular crop + resize
  3. Per-frame: template match portrait ke minimap → dapat (name, x, y)
  4. Update tracked heroes dengan temporal smoothing
  5. Konversi ke game coordinates via CoordinateMapper

Detection method:
  Menggunakan hero portrait templates yang di-crop circular dan
  di-resize ke ukuran minimap icon (~18-24px diameter).
  Multi-scale cv2.matchTemplate pada minimap → langsung mendapatkan
  identity hero + posisi sekaligus (bukan HSV color detection).

Fallback:
  HoughCircles + HSV ring color tetap tersedia sebagai fallback
  jika portrait templates belum di-load atau confidence rendah.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...core import layout as layout_mod
from ...mapper.coordinate_mapper import CoordinateMapper, MapPosition
from ..base import BaseDetector, Detection

logger = logging.getLogger("mlbb.vision.minimap_hero")

# ── Default assets path ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HEROES_ASSET_DIR = _PROJECT_ROOT / "assets" / "heroes"

# ── Default HSV ranges untuk fallback blue/red team dots ──
BLUE_HUE_CENTER = 100
BLUE_HUE_RANGE = 15
RED_HUE_CENTER = 170
RED_HUE_RANGE = 20       # perlu lebih wide karena red wraps around 0/180
SAT_MIN = 80
VAL_MIN = 50
AREA_MIN = 12
AREA_MAX = 250

# ── Template matching config ──
# Ukuran icon hero di minimap (diameter dalam pixel)
# Multi-scale untuk handle variasi resolusi
ICON_SIZES = [16, 18, 20, 22, 24]  # tambah scale untuk finer matching
# Confidence threshold untuk template match dianggap valid
MATCH_THRESHOLD = 0.35  # lebih rendah untuk recall, diverifikasi via border color
# NMS overlap threshold (IoU)
NMS_IOU_THRESHOLD = 0.3
# Jarak minimum (pixel) antar deteksi untuk NMS
NMS_DISTANCE_THRESHOLD = 12
# Bobot kombinasi score
SCORE_GRAY_WEIGHT = 0.55
SCORE_EDGE_WEIGHT = 0.30
SCORE_BORDER_WEIGHT = 0.15


@dataclass
class MinimapHero:
    """Satu hero yang terlacak di minimap."""
    name: str | None            # hero name dari roster (None jika unknown)
    team: str                   # "blue" | "red"
    norm_x: float               # posisi x normalized 0-1 di minimap
    norm_y: float               # posisi y normalized 0-1 di minimap
    confidence: float           # seberapa yakin detection ini
    last_seen_frame: int = 0    # frame terakhir terlihat
    first_seen_frame: int = 0   # frame pertama terlihat
    frames_alive: int = 0       # berapa frame berturut-turut terlihat
    pixel_x: int = 0            # posisi x dalam pixel minimap
    pixel_y: int = 0            # posisi y dalam pixel minimap
    game_pos: MapPosition | None = None  # posisi dalam game coordinates
    is_dead: bool = False       # apakah hero mati (dot hilang dari minimap)


@dataclass
class TrackedMinimapHero:
    """Internal tracked state untuk satu hero."""
    name: str | None
    team: str
    # Smoothed position (EMA)
    norm_x: float
    norm_y: float
    last_seen_frame: int
    first_seen_frame: int
    frames_alive: int
    confidence: float
    # Smoothing factor (0-1), lower = lebih responsive (kurang smoothing)
    smooth_alpha: float = 0.3  # lebih cepat mengikuti pergerakan
    # Konter consecutive miss sebelum dianggap dead
    miss_count: int = 0
    max_miss_before_dead: int = 5
    # Velocity (normalized per frame) untuk position gating
    vel_x: float = 0.0
    vel_y: float = 0.0
    # Maximum movement per frame (normalized) — lebih besar = lebih responsif
    max_move: float = 0.08  # ~27px pada 340px minimap (sebelumnya 0.03)

    def smooth_position(self, raw_x: float, raw_y: float):
        """
        Velocity-gated position update.

        1. Predict position: pred = current + velocity
        2. Check if raw observation is within max_move of prediction
        3. If YES: update with velocity-aware smoothing + update velocity
        4. If NO:  fall back to prediction (reject outlier detection)
        """
        # Predict
        pred_x = self.norm_x + self.vel_x
        pred_y = self.norm_y + self.vel_y

        # Gate check
        dx = raw_x - pred_x
        dy = raw_y - pred_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist <= self.max_move:
            # Normal update with velocity-aware smoothing
            self.norm_x = self.smooth_alpha * pred_x + (1 - self.smooth_alpha) * raw_x
            self.norm_y = self.smooth_alpha * pred_y + (1 - self.smooth_alpha) * raw_y
            # Update velocity
            self.vel_x = self.norm_x - pred_x
            self.vel_y = self.norm_y - pred_y
        else:
            # Reject outlier — coast with prediction (small damping)
            self.norm_x = pred_x * 0.9 + self.norm_x * 0.1
            self.norm_y = pred_y * 0.9 + self.norm_y * 0.1
            self.vel_x *= 0.9  # velocity decay
            self.vel_y *= 0.9

    def to_minimap_hero(self, frame_idx: int, mapper: CoordinateMapper | None,
                        minimap_w: int, minimap_h: int) -> MinimapHero:
        """Convert to public MinimapHero dataclass."""
        px = int(self.norm_x * minimap_w)
        py = int(self.norm_y * minimap_h)
        game_pos = mapper.minimap_to_game(self.norm_x, self.norm_y) if mapper else None
        return MinimapHero(
            name=self.name,
            team=self.team,
            norm_x=self.norm_x,
            norm_y=self.norm_y,
            confidence=self.confidence,
            last_seen_frame=frame_idx,
            first_seen_frame=self.first_seen_frame,
            frames_alive=self.frames_alive,
            pixel_x=px,
            pixel_y=py,
            game_pos=game_pos,
            is_dead=False,
        )


@dataclass
class _PortraitMatch:
    """Hasil satu portrait match di minimap."""
    name: str
    team: str
    cx: int          # center x dalam pixel minimap
    cy: int          # center y dalam pixel minimap
    confidence: float
    scale_idx: int   # index skala yang cocok
    bbox: tuple[int, int, int, int]  # (x, y, w, h) match region


class MinimapHeroTracker:
    """
    Melacak posisi hero di minimap per-hero.

    Menggunakan circular portrait template matching untuk menemukan
    posisi tiap hero di minimap. Portrait dari assets/heroes/ di-crop
    circular dan di-resize ke ukuran minimap icon, lalu di-match
    ke minimap image setiap frame.

    Args:
        max_miss_frames: Berapa frame hero hilang sebelum dianggap dead.
        match_distance: Jarak max (normalized 0-1) untuk match dot ke tracked hero.
        smooth_alpha: Smoothing factor EMA (0-1) — lebih besar = lebih smooth.
        use_coordinate_mapper: Apakah konversi ke game coordinates.
        match_threshold: Min confidence untuk portrait match (0-1).
        icon_sizes: List ukuran diameter icon (pixel) untuk multi-scale.
        heroes_asset_dir: Path ke folder hero portraits.
    """

    def __init__(
        self,
        max_miss_frames: int = 5,
        match_distance: float = 0.06,
        smooth_alpha: float = 0.6,
        use_coordinate_mapper: bool = True,
        minimap_bbox: tuple[int, int, int, int] | None = None,
        minimap_size: int | None = None,
        match_threshold: float = MATCH_THRESHOLD,
        icon_sizes: list[int] | None = None,
        heroes_asset_dir: str | Path | None = None,
        yolo_model_path: str | Path | None = None,
    ):
        self.max_miss_frames = max_miss_frames
        self.match_distance = match_distance
        self.smooth_alpha = smooth_alpha
        self.match_threshold = match_threshold
        self.icon_sizes = icon_sizes or list(ICON_SIZES)
        self._yolo_model_path = yolo_model_path
        self._yolo_detector = None

        # Tracked heroes: index by hero name
        self._tracked: dict[str, TrackedMinimapHero] = {}  # name -> hero
        self._unknown_count: int = 0

        # Roster dari TeamDetector
        self._roster: dict[str, str] = {}  # hero_name -> team
        self._roster_unassigned: set[str] = set()

        # ── Portrait templates ──
        # Raw portrait (original dari asset/scoreboard)
        self._portraits_raw: dict[str, np.ndarray] = {}
        # Circular templates per scale: {name: {'gray': [...], 'edge': [...]}}
        self._circle_templates: dict[str, dict] = {}
        # Circular masks per scale (shared): [mask_s1, mask_s2, ...]
        self._circle_masks: list[np.ndarray] = []

        # Pre-compute circular masks untuk setiap scale
        for size in self.icon_sizes:
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
            self._circle_masks.append(mask)

        # Load asset portraits
        self._heroes_asset_dir = Path(heroes_asset_dir) if heroes_asset_dir else _HEROES_ASSET_DIR
        self._load_asset_portraits()

        # Coordinate mapper
        self._mapper: CoordinateMapper | None = None
        if use_coordinate_mapper:
            mm_bbox = minimap_bbox or layout_mod.bbox("map") or (80, 0, 350, 340)
            _, _, mm_w, mm_h = mm_bbox
            mm_size = minimap_size or max(mm_w, mm_h)
            self._mapper = CoordinateMapper(minimap_size=mm_size)

        # Layout info (untuk pixel coordinate)
        self._minimap_bbox = minimap_bbox or layout_mod.bbox("map") or (80, 0, 350, 340)
        _, _, self._mm_w, self._mm_h = self._minimap_bbox

        # Debug data (last frame's raw detection for visualization)
        self._last_debug: dict | None = None
        self._last_circles: list[tuple[int, int, int]] = []  # HoughCircles output for debug overlay
        # Position history for variance-based motion filter
        self._pos_history: dict[tuple[int, int], list[tuple[int, int]]] = {}  # (x//16,y//16) -> [(x,y),...]
        self._pos_history_frames: int = 0
        self._motion_blend: float = 0.7  # blending factor for mask history

        # Frame counter
        self._current_frame = 0

        logger.info(
            "MinimapHeroTracker initialized: %d asset portraits, "
            "%d circle templates, icon_sizes=%s, "
            "match_thresh=%.2f, smooth=%.2f, miss_thresh=%d",
            len(self._portraits_raw),
            len(self._circle_templates),
            self.icon_sizes,
            match_threshold, smooth_alpha, max_miss_frames,
        )

        # ── YOLO detector (optional) ──
        if self._yolo_model_path is not None:
            try:
                from .yolo_detector import YOLOMinimapDetector
                self._yolo_detector = YOLOMinimapDetector(self._yolo_model_path)
                logger.info("YOLO detector loaded from %s", self._yolo_model_path)
            except Exception as e:
                logger.warning("Failed to load YOLO detector: %s", e)
                self._yolo_detector = None

    # ── Portrait Template Preparation ─────────────────────────────────

    def _load_asset_portraits(self):
        """
        Load hero portrait PNG dari assets/heroes/ directory.

        Setiap file {hero_key}.png di-load, di-crop circular, dan
        di-resize ke setiap ukuran di self.icon_sizes.
        """
        if not self._heroes_asset_dir.is_dir():
            logger.warning("Heroes asset dir not found: %s", self._heroes_asset_dir)
            return

        count = 0
        for fpath in sorted(self._heroes_asset_dir.glob("*.png")):
            name = fpath.stem  # e.g. "aamon", "chou", "lapu_lapu"
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to load portrait: %s", fpath)
                continue

            self._portraits_raw[name] = img
            self._circle_templates[name] = self._prepare_circular_templates(img)
            count += 1

        logger.info(
            "Loaded %d hero portraits from %s", count, self._heroes_asset_dir,
        )

    def _prepare_circular_templates(
        self, portrait: np.ndarray,
    ) -> dict:
        """
        Buat circular templates dari satu portrait image — FIXED version.

        Perbaikan dari versi sebelumnya:
        - Background diisi dengan neutral gray (128), bukan 0 (black)
          → mencegah NaN di TM_CCOEFF_NORMED dan false positive di dark area
        - Template terdiri dari GRAY + EDGE version
          → EDGE menangkap struktur wajah yang lebih diskriminatif di ukuran kecil
        - EDGE: Laplacian setelah GaussianBlur untuk noise reduction

        Args:
            portrait: Full portrait image (BGR, biasanya 128x128).

        Returns:
            Dict with keys 'gray' dan 'edge', masing-masing list per scale.
        """
        h, w = portrait.shape[:2]

        # Crop center 65% — fokus ke wajah
        margin = 0.175
        y1 = int(h * margin)
        y2 = int(h * (1 - margin))
        x1 = int(w * margin)
        x2 = int(w * (1 - margin))
        face = portrait[y1:y2, x1:x2]

        if face.size == 0:
            face = portrait

        # Convert ke grayscale
        if face.ndim == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray = face

        gray_templates = []
        edge_templates = []

        for i, size in enumerate(self.icon_sizes):
            mask = self._circle_masks[i]

            # Resize ke target icon size
            resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

            # Equalize histogram untuk robustness
            eq = cv2.equalizeHist(resized)

            # ── FIX: Gunakan NETRAL BACKGROUND (128) bukan black (0) ──
            # Background = 128, circle content = equalized face
            # Ini mencegah false positive di dark minimap area karena
            # TM_CCOEFF_NORMED akan membandingkan pola, bukan brightness absolut
            tmpl_gray = np.full((size, size), 128, dtype=np.uint8)
            np.copyto(tmpl_gray, eq, where=mask.astype(bool))
            gray_templates.append(tmpl_gray)

            # ── EDGE template untuk structural matching ──
            # Gaussian blur dulu sebelum Laplacian untuk mengurangi noise
            blurred = cv2.GaussianBlur(eq, (3, 3), 0)
            edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
            # Clamp edges untuk gradien yang lebih bersih
            edges = np.clip(edges, 0, 255).astype(np.uint8)

            # Edge template juga dengan neutral background
            tmpl_edge = np.full((size, size), 0, dtype=np.uint8)  # edge bg = 0 (hitam, natural untuk edge map)
            np.copyto(tmpl_edge, edges, where=mask.astype(bool))
            edge_templates.append(tmpl_edge)

        return {
            'gray': gray_templates,
            'edge': edge_templates,
        }

    # ── Roster Management ─────────────────────────────────────────────

    def set_roster(self, blue_heroes: list[str], red_heroes: list[str]):
        """
        Set team roster dari TeamDetector.

        Jika hero name ada di loaded portraits, circle templates sudah
        siap dipakai. Jika tidak, hero tetap bisa di-track via fallback.

        Args:
            blue_heroes: List 5 hero name tim biru.
            red_heroes: List 5 hero name tim merah.
        """
        new_roster: dict[str, str] = {}
        for name in blue_heroes:
            new_roster[name] = "blue"
        for name in red_heroes:
            new_roster[name] = "red"

        self._roster = new_roster

        # Rebuild unassigned set (exclude already tracked heroes)
        tracked_names = {h.name for h in self._tracked.values() if h.name}
        self._roster_unassigned = {
            name for name in new_roster if name not in tracked_names
        }

        # Log template availability
        has_template = sum(1 for n in new_roster if n in self._circle_templates)
        logger.info(
            "Roster updated: %d blue + %d red = %d heroes, "
            "%d unassigned, %d have portrait templates",
            len(blue_heroes), len(red_heroes),
            len(new_roster), len(self._roster_unassigned),
            has_template,
        )

    def get_roster(self) -> dict[str, str]:
        """Return current roster {hero_name: team}."""
        return dict(self._roster)

    def is_roster_complete(self) -> bool:
        """Cek apakah roster sudah lengkap (10 hero)."""
        return len(self._roster) >= 10

    # ── Hero Portrait Templates (dari scoreboard) ────────────────────

    def set_portrait_crops(self, portraits: dict[str, np.ndarray]):
        """
        Set hero portrait crops sebagai template untuk minimap matching.

        Portraits di-crop langsung dari scoreboard bar (blue_team/red_team).
        Akan di-convert ke circular templates dan OVERRIDE asset portrait
        jika sudah ada (scoreboard crop lebih match dengan minimap skin).

        Args:
            portraits: Dict {hero_name: cropped_BGR_image}.
        """
        for name, img in portraits.items():
            self._portraits_raw[name] = img
            # Generate circular templates dari scoreboard crop
            self._circle_templates[name] = self._prepare_circular_templates(img)

        logger.info(
            "Set %d portrait crops as circular templates (override assets)",
            len(portraits),
        )

    # ── Portrait-Based Hero Detection ─────────────────────────────────
    #
    # Deteksi hero di minimap menggunakan circular portrait template
    # matching dengan combined scoring:
    #   - Gray template match (TM_CCOEFF_NORMED)
    #   - Edge template match (TM_CCOEFF_NORMED dengan Laplacian edges)
    #   - Border color verification (blue/red ring check)
    # Score final = GRAY_WEIGHT * gray_score + EDGE_WEIGHT * edge_score
    #              + BORDER_WEIGHT * border_bonus

    def _compute_gray_match(
        self, gray_img: np.ndarray, tmpl_gray: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        """Compute grayscale template match score."""
        result = cv2.matchTemplate(
            gray_img, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=mask,
        )
        # Handle NaN — jika semua nilai NaN, return 0
        if np.all(np.isnan(result)):
            return 0.0, (0, 0)
        # Find best non-NaN result
        valid = ~np.isnan(result)
        if not np.any(valid):
            return 0.0, (0, 0)
        # Cari max hanya dari non-NaN values
        flat = result[valid]
        max_val = float(np.max(flat))
        max_flat_idx = int(np.argmax(flat))
        # Convert flat index back to 2D
        valid_indices = np.where(valid)
        max_idx = (valid_indices[0][max_flat_idx], valid_indices[1][max_flat_idx])
        return max_val, max_idx

    def _compute_edge_match(
        self, edge_img: np.ndarray, tmpl_edge: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute edge template match score (TM_CCORR_NORMED — no NaN issues with edges)."""
        result = cv2.matchTemplate(
            edge_img, tmpl_edge, cv2.TM_CCORR_NORMED, mask=mask,
        )
        if result.size == 0:
            return 0.0
        return float(np.max(result))

    def _verify_border_color(
        self, bgr_img: np.ndarray, cx: int, cy: int,
        expected_team: str,
    ) -> float:
        """
        Verifikasi warna border (ring) hero di minimap.

        Hero di minimap MLBB punya ring berwarna:
          - Blue team → ring biru terang
          - Red team → ring merah

        Sampling pixel di radius sekitar circle hero dan cek dominasi warna.
        Returns bonus score (0.0 atau BORDER_BONUS_VALUE).

        Args:
            bgr_img: Minimap image (BGR).
            cx, cy: Center of detected hero.
            expected_team: "blue" atau "red".

        Returns:
            Bonus score: SCORE_BORDER_WEIGHT jika cocok, 0 jika tidak.
        """
        h, w = bgr_img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return 0.0

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # Sampling di radius ring (di luar face, di border circle)
        # Gunakan 3 radius berbeda untuk robustness
        radii = [8, 10, 12]  # pixel radius sampling
        pixels = []
        for r in radii:
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                sx = int(cx + r * np.cos(angle))
                sy = int(cy + r * np.sin(angle))
                if 0 <= sx < w and 0 <= sy < h:
                    pixels.append(hsv[sy, sx])

        if not pixels:
            return 0.0

        pixels = np.array(pixels, dtype=np.int32)
        hues = pixels[:, 0]
        sats = pixels[:, 1]
        vals = pixels[:, 2]

        # Cek dominasi warna
        blue_count = 0
        red_count = 0
        for i in range(len(hues)):
            hd_blue = min(abs(hues[i] - BLUE_HUE_CENTER),
                          180 - abs(hues[i] - BLUE_HUE_CENTER))
            hd_red = min(abs(hues[i] - RED_HUE_CENTER),
                         180 - abs(hues[i] - RED_HUE_CENTER))
            in_blue = hd_blue <= BLUE_HUE_RANGE and sats[i] >= 25 and vals[i] >= 40
            in_red = hd_red <= RED_HUE_RANGE and sats[i] >= 25 and vals[i] >= 40
            if in_blue and not in_red:
                blue_count += 1
            elif in_red and not in_blue:
                red_count += 1

        total = len(pixels)
        blue_ratio = blue_count / max(total, 1)
        red_ratio = red_count / max(total, 1)

        # Threshold: minimal 20% pixel cocok dengan dominasi 2:1
        if expected_team == "blue" and blue_ratio >= 0.20 and blue_ratio >= red_ratio * 2:
            return SCORE_BORDER_WEIGHT
        elif expected_team == "red" and red_ratio >= 0.20 and red_ratio >= blue_ratio * 2:
            return SCORE_BORDER_WEIGHT
        elif expected_team == "blue" and blue_ratio >= 0.12 and blue_ratio >= red_ratio:
            return SCORE_BORDER_WEIGHT * 0.5  # partial match
        elif expected_team == "red" and red_ratio >= 0.12 and red_ratio >= blue_ratio:
            return SCORE_BORDER_WEIGHT * 0.5
        return 0.0

    def _detect_circles_fast(
        self, minimap_img: np.ndarray,
    ) -> list[tuple[str, int, int, int]]:
        """Detect via YOLO (async cache) or HSV (fallback)."""
        if minimap_img is None or minimap_img.size == 0:
            return []
        # YOLO: pakai hasil async + trigger inference berikutnya
        if self._yolo_detector is not None:
            dets = self._yolo_detector.last_result
            self._yolo_detector.detect_async(minimap_img)
            result = [(team, cx, cy, r) for team, cx, cy, r, _ in dets]
            result.sort(key=lambda x: -x[3])
            final, used = [], []
            for team, cx, cy, r in result:
                if len(final) >= 14: break
                if all((cx-ux)**2+(cy-uy)**2>144 for ux,uy in used):
                    final.append((team, cx, cy, r)); used.append((cx, cy))
            return final
        # HSV fallback
        import numpy as np, cv2
        h, w = minimap_img.shape[:2]
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        k = np.ones((3,3), np.uint8)
        bm = cv2.morphologyEx(cv2.inRange(hsv,(85,80,80),(115,255,255)), cv2.MORPH_OPEN, k)
        r1 = cv2.morphologyEx(cv2.inRange(hsv,(150,60,80),(180,255,255)), cv2.MORPH_OPEN, k)
        r2 = cv2.morphologyEx(cv2.inRange(hsv,(0,60,80),(10,255,255)), cv2.MORPH_OPEN, k)
        for m in [bm, r1|r2]:
            cv2.rectangle(m,(0,0),(w,8),0,-1); cv2.rectangle(m,(0,h-8),(w,h),0,-1)
        def ff(mask, mc):
            dots = []
            for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                a = cv2.contourArea(cnt)
                if a < 30 or a > 400: continue
                p = cv2.arcLength(cnt, True)
                if p == 0 or 4*np.pi*a/(p*p) < mc: continue
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                face = gray[max(0,cy-3):cy+3, max(0,cx-3):cx+3]
                if face.size > 0 and face.mean() < 35: continue
                dots.append((cx, cy, max(6,int((a/np.pi)**0.5))))
            return dots
        blue = ff(bm, 0.55); red = ff(r1|r2, 0.40)
        result = [("blue", *d) for d in blue] + [("red", *d) for d in red]
        result.sort(key=lambda x: -x[3])
        final, used = [], []
        for team, cx, cy, r in result:
            if len(final) >= 14: break
            if all((cx-ux)**2+(cy-uy)**2>144 for ux,uy in used):
                final.append((team, cx, cy, r)); used.append((cx, cy))
        return final
    def detect_heroes(
        self,
        minimap_img: np.ndarray,
    ) -> list[_PortraitMatch]:
        """
        Detect hero positions — OPTIMIZED version.

        Flow:
          1. _detect_circles_fast() → HoughCircles + border color (fast: ~1ms)
          2. Untuk setiap circle → crop patch → gray template match (2 scales only)
          3. Score = gray_score + border_bonus (tanpa edge, tanpa NaN handling)
          4. NMS position-based

        Optimasi dari versi sebelumnya:
          - 5 scales → 2 scales (18, 20 px) — 60% lebih sedikit matchTemplate
          - Edge matching dihapus — gray + border sudah cukup
          - NaN handling dihapus — neutral bg (128) jarang produce NaN
          - Border color panggil _check_circle_color sekali (sudah dari _detect_circles_fast)

        Args:
            minimap_img: Cropped minimap image (BGR).

        Returns:
            List of _PortraitMatch.
        """
        h, w = minimap_img.shape[:2]

        # ── Preprocess (gray → CLAHE) ──
        gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # ── HoughCircles + border color ──
        raw_circles = self._detect_circles_fast(minimap_img)
        if not raw_circles:
            self._last_circles = []
            return []

        self._last_circles = [(team, cx, cy, r) for team, cx, cy, r in raw_circles]

        raw_matches: list[_PortraitMatch] = []

        for team, cx, cy, r in raw_circles:
            # ── Team heroes lookup ──
            team_heroes = [n for n, t in self._roster.items() if t == team]
            if not team_heroes:
                continue

            # ── Crop patch (24+4=28px, fixed size) ──
            half = 14  # 28//2
            px1 = max(0, cx - half)
            py1 = max(0, cy - half)
            px2 = min(w, cx + half)
            py2 = min(h, cy + half)

            patch = gray_eq[py1:py2, px1:px2]
            ph, pw = patch.shape[:2]
            if ph < 6 or pw < 6:
                continue

            best_name: str | None = None
            best_score = 0.0
            best_scale = 0

            # ── Template matching: 2 scales only (18, 20) ──
            _FAST_SCALES = [1, 2]  # index di icon_sizes untuk scale 18px dan 20px

            for hero_name in team_heroes:
                templates = self._circle_templates.get(hero_name)
                if not templates:
                    continue
                gray_tmpls = templates.get('gray', [])

                for scale_idx in _FAST_SCALES:
                    if scale_idx >= len(gray_tmpls) or scale_idx >= len(self._circle_masks):
                        continue
                    tg = gray_tmpls[scale_idx]
                    mask = self._circle_masks[scale_idx]
                    th, tw = tg.shape[:2]
                    if ph < th or pw < tw:
                        continue

                    # Fast gray match (tanpa NaN check)
                    result = cv2.matchTemplate(patch, tg, cv2.TM_CCOEFF_NORMED, mask=mask)
                    max_val = float(np.max(result)) if result.size > 0 else 0.0

                    if max_val > 0.01 and max_val > best_score:
                        best_score = max_val
                        best_name = hero_name
                        best_scale = scale_idx

            # ── Boundary check & identity ──
            border_bonus = self._verify_border_color(minimap_img, cx, cy, team)
            final_score = best_score + border_bonus if best_name else border_bonus

            if final_score >= self.match_threshold:
                raw_matches.append(_PortraitMatch(
                    name=best_name, team=team,
                    cx=cx, cy=cy,
                    confidence=round(min(1.0, final_score), 4),
                    scale_idx=best_scale,
                    bbox=(cx - r, cy - r, r * 2, r * 2),
                ))

        # ── NMS ──
        raw_matches.sort(key=lambda m: m.confidence, reverse=True)
        final_matches = self._nms_portrait_matches(raw_matches)

        # Store circles for debug
        self._last_circles = [(cx, cy, r) for _, cx, cy, r in raw_circles]

        return final_matches

    @staticmethod
    def _nms_portrait_matches(
        matches: list[_PortraitMatch],
        distance_threshold: int = NMS_DISTANCE_THRESHOLD,
    ) -> list[_PortraitMatch]:
        """
        Non-Maximum Suppression — POSITION-BASED (bukan IoU).

        Dua deteksi dianggap overlap jika jarak euclidean center-nya
        < distance_threshold. Yang confidence lebih tinggi dipertahankan.

        Ini lebih cocok untuk circular objects di minimap daripada IoU
        yang sensitif terhadap perbedaan scale.

        Args:
            matches: List sorted by confidence descending.
            distance_threshold: Jarak pixel max untuk dianggap overlap.

        Returns:
            Filtered list of matches (max 1 per position).
        """
        if len(matches) <= 1:
            return matches

        kept: list[_PortraitMatch] = []
        suppressed: set[int] = set()

        for i in range(len(matches)):
            if i in suppressed:
                continue
            kept.append(matches[i])

            for j in range(i + 1, len(matches)):
                if j in suppressed:
                    continue
                # Euclidean distance between centers
                dx = matches[i].cx - matches[j].cx
                dy = matches[i].cy - matches[j].cy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < distance_threshold:
                    suppressed.add(j)

        return kept

    @staticmethod
    def _compute_iou(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union antara dua bbox (x, y, w, h)."""
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        # Intersection
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1)

    # ── Fallback: HoughCircles + HSV (backward compat) ────────────────
    #
    # Tetap tersedia untuk dipakai jika portrait templates belum di-load
    # atau sebagai fallback ketika template matching gagal.

    @staticmethod
    def _check_circle_color(
        bgr: np.ndarray, cx: int, cy: int, r: int,
    ) -> str | None:
        """
        Cek warna ring + face brightness hero di minimap.

        DUAL verification:
          1. Face brightness: center area harus lebih terang dari background
             (hero face/icon, bukan lingkaran kosong)
          2. Ring color: sampling di beberapa radius untuk cari biru/merah
             (mengcover ketidaktepatan radius HoughCircles)

        Returns: "blue", "red", atau None.
        """
        h, w = bgr.shape[:2]

        # ── Edge rejection ──
        if cx < 8 or cx > w - 8 or cy < 8 or cy > h - 8:
            return None

        # ── Face brightness check ──
        # Hero face area (radius ~4-5 dari center) harus lebih terang
        # dari background minimap (~25-40). Ini reject false positives
        # seperti bush markers, minion dots di pinggir, dll.
        face_r = max(3, min(5, r - 2))
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(face_mask, (cx, cy), face_r, 255, -1)
        face_region = bgr[face_mask == 255]
        if len(face_region) > 0:
            face_brightness = cv2.cvtColor(face_region[None], cv2.COLOR_BGR2GRAY).mean()
        else:
            face_brightness = 0
        if face_brightness < 35:  # terlalu gelap → bukan hero face
            return None

        # ── Ring color check (multi-radius) ──
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        pixels = []
        radii = [r + dr for dr in range(-2, 3) if r + dr >= 3]
        for pr in radii:
            for i in range(8):  # 8 angles × 5 radii = 40 samples
                angle = 2 * np.pi * i / 8
                sx = int(cx + pr * np.cos(angle))
                sy = int(cy + pr * np.sin(angle))
                if 0 <= sx < w and 0 <= sy < h:
                    pixels.append(hsv[sy, sx])

        if not pixels:
            return None

        pixels = np.array(pixels, dtype=np.int32)
        hues = pixels[:, 0]
        sats = pixels[:, 1]
        vals = pixels[:, 2]

        blue_count = 0
        red_count = 0
        for i in range(len(hues)):
            hd_blue = min(abs(hues[i] - BLUE_HUE_CENTER),
                          180 - abs(hues[i] - BLUE_HUE_CENTER))
            hd_red = min(abs(hues[i] - RED_HUE_CENTER),
                         180 - abs(hues[i] - RED_HUE_CENTER))
            in_blue = hd_blue <= BLUE_HUE_RANGE and sats[i] >= 25 and vals[i] >= 40
            in_red = hd_red <= RED_HUE_RANGE and sats[i] >= 25 and vals[i] >= 40
            if in_blue and not in_red:
                blue_count += 1
            elif in_red and not in_blue:
                red_count += 1

        total = len(pixels)
        if blue_count >= total * 0.15 and blue_count >= red_count * 2:
            return "blue"
        if red_count >= total * 0.15 and red_count >= blue_count * 2:
            return "red"
        return None

    def detect_dots(
        self, minimap_img: np.ndarray,
        return_debug: bool = False,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | tuple:
        """
        Fallback: Detect hero circles via HoughCircles + ring color check.

        Digunakan sebagai fallback jika portrait templates belum di-load
        atau untuk mendeteksi hero yang belum ada di roster.

        Args:
            minimap_img: Cropped minimap image (BGR).
            return_debug: If True, return extra debug data.

        Returns:
            Normal: (blue_dots, red_dots).
            Debug: (blue_dots, red_dots, blue_mask, red_mask,
                    match_details, []).
        """
        if minimap_img is None or minimap_img.size == 0:
            if return_debug:
                return [], [], None, None, [], []
            return [], []

        gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # CLAHE untuk enhance kontras
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # HoughCircles: cari lingkaran
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=max(18, min(w, h) // 12),
            param1=50,
            param2=14,
            minRadius=6,   # hero icon ~12-18px diameter → 6-9px radius
            maxRadius=11,  # maks radius hero
        )

        blue_dots: list = []
        red_dots: list = []
        all_circles: list = []

        if circles is not None:
            circles = np.round(circles[0]).astype("int")
            for (cx, cy, r) in circles:
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    continue
                all_circles.append((cx, cy, r))
                # Cek warna ring → biru, merah, atau bukan hero
                team = self._check_circle_color(minimap_img, cx, cy, r)
                if team == "blue":
                    blue_dots.append((cx, cy))
                elif team == "red":
                    red_dots.append((cx, cy))

        if return_debug:
            blue_mask = np.zeros((h, w), dtype=np.uint8)
            red_mask = np.zeros((h, w), dtype=np.uint8)
            for cx, cy in blue_dots:
                cv2.circle(blue_mask, (cx, cy), 8, 255, -1)
            for cx, cy in red_dots:
                cv2.circle(red_mask, (cx, cy), 8, 255, -1)
            return blue_dots, red_dots, blue_mask, red_mask, all_circles, []

        return blue_dots, red_dots

    # ── Core Tracking Logic ───────────────────────────────────────────

    def _normalize(self, px: int, py: int) -> tuple[float, float]:
        """Convert pixel coords to normalized 0-1."""
        nx = px / max(1, self._mm_w)
        ny = py / max(1, self._mm_h)
        return (nx, ny)

    def _nearest_neighbor_match(
        self,
        dots: list[tuple[int, int]],
        tracked_list: list[TrackedMinimapHero],
        max_dist_tight: float,
        max_dist_wide: float | None = None,
    ) -> dict[str, tuple[int, int]]:
        """
        Match dots ke tracked heroes via nearest-neighbor, two-pass.

        Pass 1 — tight threshold (max_dist_tight):
          Untuk hero yang aktif (miss_count=0). Mereka tidak bergerak jauh
          antar frame, jadi threshold ketat mencegah false match.

        Pass 2 — wide threshold (max_dist_wide):
          Untuk hero yang baru reappear setelah fog/kematian (miss_count>0).
          Posisi bisa berubah drastis (respawn di base).

        Args:
            dots: List of (px, py) dot positions.
            tracked_list: List of tracked heroes to match against.
            max_dist_tight: Threshold untuk active tracking.
            max_dist_wide: Threshold untuk reconnection (default: 3× tight).

        Returns:
            Dict {tracked_name: (px, py)} matched pairs.
        """
        if not dots or not tracked_list:
            return {}

        max_dist_wide = max_dist_wide or (max_dist_tight * 3)
        dot_norm = [self._normalize(px, py) for px, py in dots]
        matched: dict[str, tuple[int, int]] = {}
        used_dots: set[int] = set()

        # Sort tracked: active (miss_count=0) first, then recently missed
        # Dalam tiap grup, urutkan frames_alive descending
        def _sort_key(h: TrackedMinimapHero):
            return (0 if h.miss_count == 0 else 1, -h.frames_alive)

        sorted_tracked = sorted(tracked_list, key=_sort_key)

        for hero in sorted_tracked:
            threshold = max_dist_tight if hero.miss_count == 0 else max_dist_wide
            best_dist = threshold
            best_idx = -1
            for i, (dnx, dny) in enumerate(dot_norm):
                if i in used_dots:
                    continue
                dist = ((dnx - hero.norm_x) ** 2 + (dny - hero.norm_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                key = hero.name or str(hero.first_seen_frame)
                matched[key] = dots[best_idx]
                used_dots.add(best_idx)

        return matched

    def update(
        self,
        minimap_img: np.ndarray,
        frame_idx: int,
    ) -> list[MinimapHero]:
        """
        Update tracking — FAST PATH.

        Flow optimasi:
          1. _detect_circles_fast() → HoughCircles + border color (0.8ms)
          2. Stable heroes (frames_alive=10, conf=0.7, miss=0):
             -> NN match ke nearest same-team circle -> skip template match
          3. Sisa circles -> template matching (2 scales, gray only) untuk identity
          4. Named matches -> update tracking langsung
          5. Unmatched dots -> NN ke unstable tracked heroes
          6. Increment miss_count, cleanup dead heroes

        Fast path menghemat ~80% template matching karena stable heroes
        tidak perlu di-template-match tiap frame.

        Returns:
            List MinimapHero untuk hero yang visible.
        """
        self._current_frame = frame_idx

        # ── 1. Fast: HoughCircles + border color (geometric only) ──
        all_dots = self._detect_circles_fast(minimap_img)  # [(team, cx, cy, r), ...]
        self._last_circles = [(cx, cy, r) for _, cx, cy, r in all_dots]

        # ── Motion filter: reject static terrain via position variance ──
        self._pos_history_frames += 1
        for _, cx, cy, _ in all_dots:
            key = (cx//16*16, cy//16*16)
            if key not in self._pos_history:
                self._pos_history[key] = []
            self._pos_history[key].append((cx, cy))
            # Keep only last 60 entries
            if len(self._pos_history[key]) > 60:
                self._pos_history[key] = self._pos_history[key][-60:]

        # After 30 frames of data, filter by variance
        if self._pos_history_frames >= 30:
            # Decay: remove very old history entries (>60 frames ago)
            for key in list(self._pos_history.keys()):
                if len(self._pos_history[key]) > 50:
                    self._pos_history[key] = self._pos_history[key][-40:]
                # Check variance
                if len(self._pos_history[key]) >= 10:
                    xs = [p[0] for p in self._pos_history[key]]
                    ys = [p[1] for p in self._pos_history[key]]
                    var = np.var(xs) + np.var(ys)
                    if var < 15:  # static terrain/tower
                        del self._pos_history[key]

            # Filter current dots: remove if position is in static history
            filtered = []
            for item in all_dots:
                team, cx, cy, r = item
                key = (cx//16*16, cy//16*16)
                if key in self._pos_history:
                    filtered.append(item)  # has motion history = moving hero
                # Keep if near a tracked hero (for NN matching)
                elif any((cx-int(h.norm_x*self._mm_w))**2+(cy-int(h.norm_y*self._mm_h))**2 < 400
                         for h in self._tracked.values() if h.miss_count < self.max_miss_frames):
                    filtered.append(item)
            if filtered:
                all_dots = filtered

        matched_names: set[str] = set()
        used_dots: set[int] = set()  # indices of used dots in all_dots

        # ── 2. Fast path: stable heroes -> NN match ──
        stable_heroes = [
            h for h in self._tracked.values()
            if h.name and h.frames_alive >= 10
            and h.confidence >= 0.7 and h.miss_count == 0
        ]

        for hero in stable_heroes:
            best_dist = self.match_distance
            best_idx = -1
            for i, (team, cx, cy, r) in enumerate(all_dots):
                if i in used_dots or team != hero.team:
                    continue
                nx, ny = cx / max(1, self._mm_w), cy / max(1, self._mm_h)
                dist = ((nx - hero.norm_x) ** 2 + (ny - hero.norm_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                _, cx, cy, r = all_dots[best_idx]
                used_dots.add(best_idx)
                matched_names.add(hero.name)

                nx, ny = self._normalize(cx, cy)
                hero.smooth_position(nx, ny)
                hero.last_seen_frame = frame_idx
                hero.frames_alive += 1
                hero.miss_count = 0
                hero.confidence = min(1.0, hero.confidence + 0.01)

        # ── 3. Sisa circles -> template matching untuk identity ──
        remaining_dots = [
            (team, cx, cy, r) for i, (team, cx, cy, r) in enumerate(all_dots)
            if i not in used_dots
        ]

        if remaining_dots:
            h, w = minimap_img.shape[:2]
            gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)

            for team, cx, cy, r in remaining_dots:
                team_heroes = [n for n, t in self._roster.items() if t == team]
                if not team_heroes:
                    continue

                half = 14; px1, py1 = max(0, cx-half), max(0, cy-half)
                px2, py2 = min(w, cx+half), min(h, cy+half)
                patch = gray_eq[py1:py2, px1:px2]; ph, pw = patch.shape[:2]
                if ph < 6 or pw < 6:
                    continue

                best_name = None; best_score = 0.0

                for hero_name in team_heroes:
                    templates = self._circle_templates.get(hero_name)
                    if not templates: continue
                    gray_tmpls = templates.get('gray', [])
                    for scale_idx in (1, 2):  # scales 18, 20
                        if scale_idx >= len(gray_tmpls) or scale_idx >= len(self._circle_masks):
                            continue
                        tg, mask = gray_tmpls[scale_idx], self._circle_masks[scale_idx]
                        th, tw = tg.shape[:2]
                        if ph < th or pw < tw: continue
                        result = cv2.matchTemplate(patch, tg, cv2.TM_CCOEFF_NORMED, mask=mask)
                        max_val = float(np.max(result)) if result.size > 0 else 0.0
                        if max_val > 0.01 and max_val > best_score:
                            best_score, best_name = max_val, hero_name

                border_bonus = self._verify_border_color(minimap_img, cx, cy, team)
                final_score = best_score + border_bonus if best_name else border_bonus
                if final_score < self.match_threshold:
                    continue

                if best_name:
                    name = best_name
                    if name in self._tracked:
                        hero = self._tracked[name]
                        nx, ny = self._normalize(cx, cy)
                        hero.smooth_position(nx, ny)
                        hero.last_seen_frame = frame_idx
                        hero.frames_alive += 1
                        hero.miss_count = 0
                        hero.confidence = min(1.0, final_score * 0.7 + min(0.25, hero.frames_alive * 0.01) + 0.15)
                    else:
                        nx, ny = self._normalize(cx, cy)
                        t = self._roster.get(name, team)
                        hero = TrackedMinimapHero(name=name, team=t, norm_x=nx, norm_y=ny,
                            last_seen_frame=frame_idx, first_seen_frame=frame_idx,
                            frames_alive=1, confidence=final_score, smooth_alpha=self.smooth_alpha)
                        self._tracked[name] = hero
                        self._roster_unassigned.discard(name)
                    matched_names.add(name)

        # ── 4. NN fallback untuk hero yang belum matched ──
        unmatched_tracked = [
            h for h in self._tracked.values()
            if h.name and h.name not in matched_names and h.miss_count < self.max_miss_frames
        ]
        if unmatched_tracked:
            for team in ("blue", "red"):
                avail = [(cx, cy) for t, cx, cy, r in all_dots if t == team
                         and not any(abs(cx - int(mh.norm_x*self._mm_w)) < NMS_DISTANCE_THRESHOLD
                                     and abs(cy - int(mh.norm_y*self._mm_h)) < NMS_DISTANCE_THRESHOLD
                                     for mn in matched_names for mh in [self._tracked.get(mn)] if mh)]
                if not avail:
                    continue
                nn = self._nearest_neighbor_match(avail, [h for h in unmatched_tracked if h.team == team],
                                                  self.match_distance, self.match_distance * 3)
                for key, (px, py) in nn.items():
                    h = self._tracked.get(key)
                    if h is None: continue
                    matched_names.add(h.name or "")
                    nx, ny = self._normalize(px, py)
                    h.smooth_position(nx, ny); h.last_seen_frame = frame_idx
                    h.frames_alive += 1; h.miss_count = 0
                    h.confidence = min(1.0, h.frames_alive * 0.04 + 0.5)

        # ── 5. Sisa dots: assign ke roster unassigned ──
        for team, cx, cy, r in all_dots:
            if any(i in used_dots for i, (t, cxc, cyc, rc) in enumerate(all_dots) if (cxc, cyc) == (cx, cy)):
                continue
            if matched_names and any(abs(cx - int(mh.norm_x*self._mm_w)) < NMS_DISTANCE_THRESHOLD
                                     and abs(cy - int(mh.norm_y*self._mm_h)) < NMS_DISTANCE_THRESHOLD
                                     for mn in matched_names for mh in [self._tracked.get(mn)] if mh):
                continue
            candidates = [n for n in self._roster_unassigned if self._roster.get(n) == team]
            if candidates:
                name = candidates[0]
                nx, ny = self._normalize(cx, cy)
                h = TrackedMinimapHero(name=name, team=team, norm_x=nx, norm_y=ny,
                    last_seen_frame=frame_idx, first_seen_frame=frame_idx,
                    frames_alive=1, confidence=0.5, smooth_alpha=self.smooth_alpha)
                self._tracked[name] = h; self._roster_unassigned.discard(name)
                matched_names.add(name)

        # ── 6. Coasting: gerakin hero yang gak terdeteksi sesuai velocity ──
        for h in self._tracked.values():
            if h.name and h.name not in matched_names:
                h.norm_x += h.vel_x * 0.5  # half-speed coast
                h.norm_y += h.vel_y * 0.5
                h.vel_x *= 0.95
                h.vel_y *= 0.95
                h.miss_count += 1

        # ── 7. Cleanup dead ──
        dead = [n for n, h in self._tracked.items() if h.miss_count >= self.max_miss_frames]
        for n in dead:
            del self._tracked[n]
            if n in self._roster: self._roster_unassigned.add(n)

        # ── 8. Debug ──
        hh, ww = self._mm_h, self._mm_w
        if minimap_img is not None and minimap_img.size > 0:
            hh, ww = minimap_img.shape[:2]

        blue_dots = [(cx, cy) for t, cx, cy, r in all_dots if t == "blue"]
        red_dots = [(cx, cy) for t, cx, cy, r in all_dots if t == "red"]

        for h in self._tracked.values():
            if h.miss_count < self.max_miss_frames:
                px, py = int(h.norm_x * ww), int(h.norm_y * hh)
                if h.team == "blue" and (px, py) not in blue_dots:
                    blue_dots.append((px, py))
                elif h.team == "red" and (px, py) not in red_dots:
                    red_dots.append((px, py))

        bm = np.zeros((hh, ww), dtype=np.uint8)
        rm = np.zeros((hh, ww), dtype=np.uint8)
        for cx, cy in blue_dots: cv2.circle(bm, (cx, cy), 8, 255, -1)
        for cx, cy in red_dots: cv2.circle(rm, (cx, cy), 8, 255, -1)

        self._last_debug = dict(minimap_img=minimap_img, portrait_matches=[], matched_names=list(matched_names),
            tracked_count=len(self._tracked), dead_heroes=dead, blue_dots=blue_dots, red_dots=red_dots,
            blue_mask=bm, red_mask=rm, blue_all_cnt=self._last_circles)

        # ── 9. Result ──
        return [h.to_minimap_hero(frame_idx, self._mapper, self._mm_w, self._mm_h)
                for h in self._tracked.values() if h.miss_count < self.max_miss_frames]

    # ── Query Methods ────────────────────────────────────────────────

    def get_hero_position(self, hero_name: str) -> MinimapHero | None:
        """Get current position of a specific hero."""
        tracked = self._tracked.get(hero_name)
        if tracked is None or tracked.miss_count >= self.max_miss_frames:
            return None
        return tracked.to_minimap_hero(
            self._current_frame, self._mapper, self._mm_w, self._mm_h,
        )

    def get_all_positions(self) -> list[MinimapHero]:
        """Get positions of all currently visible heroes."""
        return [
            h.to_minimap_hero(
                self._current_frame, self._mapper, self._mm_w, self._mm_h,
            )
            for h in self._tracked.values()
            if h.miss_count < self.max_miss_frames
        ]

    def get_team_positions(self, team: str) -> list[MinimapHero]:
        """Get positions of all visible heroes for a team."""
        return [
            h.to_minimap_hero(
                self._current_frame, self._mapper, self._mm_w, self._mm_h,
            )
            for h in self._tracked.values()
            if h.team == team and h.miss_count < self.max_miss_frames
        ]

    def get_last_debug_data(self) -> dict | None:
        """Return debug data from last frame."""
        return self._last_debug

    def get_template_names(self) -> list[str]:
        """Return list of hero names that have loaded templates."""
        return list(self._circle_templates.keys())

    def has_template(self, hero_name: str) -> bool:
        """Check if a specific hero has a loaded template."""
        return hero_name in self._circle_templates

    def set_hsv_params(self, blue_hue: int = 100, blue_range: int = 15,
                       red_hue: int = 170, red_range: int = 15,
                       sat_min: int = 40, val_min: int = 60,
                       area_min: int = 4, area_max: int = 250):
        """Override HSV detection parameters (for fallback detection)."""
        global BLUE_HUE_CENTER, BLUE_HUE_RANGE, RED_HUE_CENTER, RED_HUE_RANGE
        global SAT_MIN, VAL_MIN, AREA_MIN, AREA_MAX
        BLUE_HUE_CENTER = blue_hue
        BLUE_HUE_RANGE = blue_range
        RED_HUE_CENTER = red_hue
        RED_HUE_RANGE = red_range
        SAT_MIN = sat_min
        VAL_MIN = val_min
        AREA_MIN = area_min
        AREA_MAX = area_max
        logger.info("HSV params: blue=%d±%d red=%d±%d S≥%d V≥%d area=[%d,%d]",
                    blue_hue, blue_range, red_hue, red_range,
                    sat_min, val_min, area_min, area_max)

    def reset(self):
        """Reset semua tracking state."""
        self._tracked.clear()
        self._unknown_count = 0
        self._roster_unassigned = set(self._roster.keys())
        self._last_debug = None
        self._last_circles = []
        self._bg_blue = None
        self._bg_red = None
        self._pos_history.clear()
        self._pos_history_frames = 0
        logger.info("MinimapHeroTracker reset")


# ── Pipeline-compatible Wrapper ────────────────────────────────────────

class MinimapHeroDetector(BaseDetector):
    """
    Pipeline-compatible detector untuk hero positions di minimap.

    Membungkus MinimapHeroTracker agar bisa dipakai di pipeline vision.

    Args:
        tracker: Instance MinimapHeroTracker (dibuat otomatis jika None).
        yolo_model_path: Path ke YOLO model. Jika None, pake HSV fallback.
    """

    def __init__(self, ocr=None, tracker: MinimapHeroTracker | None = None,
                 yolo_model_path: str | None = None):
        super().__init__(ocr)
        if tracker is None:
            tracker = MinimapHeroTracker(yolo_model_path=yolo_model_path)
        self.tracker = tracker
        self.load_config("minimap")

    def set_roster(self, blue_heroes: list[str], red_heroes: list[str]):
        """Forward roster ke internal tracker."""
        self.tracker.set_roster(blue_heroes, red_heroes)

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        Detect all hero positions on minimap.

        Args:
            image: Cropped minimap region.

        Returns:
            Detection with hero position data.
        """
        if image is None or image.size == 0:
            return None

        heroes = self.tracker.update(image, getattr(self, '_frame_idx', 0))

        # Count visible per team
        blue_visible = sum(1 for h in heroes if h.team == "blue")
        red_visible = sum(1 for h in heroes if h.team == "red")

        return Detection(
            value={
                "heroes": [
                    {
                        "name": h.name,
                        "team": h.team,
                        "norm_x": h.norm_x,
                        "norm_y": h.norm_y,
                        "pixel_x": h.pixel_x,
                        "pixel_y": h.pixel_y,
                        "confidence": round(h.confidence, 3),
                        "game_x": h.game_pos.x if h.game_pos else None,
                        "game_y": h.game_pos.y if h.game_pos else None,
                        "lane": h.game_pos.lane if h.game_pos else None,
                        "nearest_landmark": h.game_pos.nearest_landmark if h.game_pos else None,
                    }
                    for h in heroes
                ],
                "blue_visible": blue_visible,
                "red_visible": red_visible,
                "total_visible": len(heroes),
                "roster_complete": self.tracker.is_roster_complete(),
            },
            confidence=0.85 if heroes else 0.5,
            label="minimap_heroes",
            meta={
                "blue_visible": blue_visible,
                "red_visible": red_visible,
            },
        )

    def set_frame_idx(self, idx: int):
        """Set frame index for tracking."""
        self._frame_idx = idx

"""
Auto Labeling Tool — MLBB minimap hero dot auto-detection.

Deteksi hero dots berdasarkan kombinasi HSV color mask + circularity check.
Menggunakan contour detection dari warna ring (biru/merah), dengan filter ketat:
  - Face brightness (patch tengah harus terang)
  - Circularity (bentuk harus bulat)
  - Area (ukuran sesuai hero dot)
  - Ring contrast (ring harus lebih jenuh dari background)

Hasil disimpan dalam format YOLO dengan prefix "auto_" untuk membedakan data manual.

Pipeline:
  Frame → crop minimap → HSV mask per team → contour → filter (area/circularity/brightness) → class assignment

Usage:
  python tools/auto_label_minimap.py --video alpha_1.mp4
  python tools/auto_label_minimap.py --all-videos --preview
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.core import layout
from vision.core.cropper import crop_region

# ── Constants ──
DATASET_DIR = Path("trainings/hero_detector")
VIDEO_DIR = Path("videos")
IMG_W, IMG_H = 350, 340
HERO_DOT_SIZE = 34  # YOLO bounding box size (sama dengan label_minimap.py)

# ── HSV ranges untuk hero rings ──
# Blue team ring: cyan/light-blue (H ~90-120)
BLUE_H_LOW, BLUE_H_HIGH = 90, 120
BLUE_S_MIN = 100    # Harus jenuh
BLUE_V_MIN = 100    # Harus terang

# Red team ring: pink/red (wraps around H 0-180 boundary)
RED_RANGES = [(150, 180), (0, 20)]
RED_S_MIN = 80
RED_V_MIN = 80

# ── Filter parameters ──
AREA_MIN = 30       # Min contour area pixels (hero dot ring ~30-120px)
AREA_MAX = 200      # Max contour area
CIRCULARITY_MIN = 0.5   # Circularity threshold (semakin tinggi semakin bulat)

# Face brightness: mean gray value of center patch (radius ~7px)
FACE_RADIUS = 7
FACE_BRIGHTNESS_MIN = 90

# Ring strength: mean saturation of ring area
RING_SAT_MIN = 100

# Overlap dedup
OVERLAP_MIN_DIST = 16

# Frame skip
FRAME_SKIP = 30

# File prefix untuk auto-labeled
AUTO_PREFIX = "auto_"


def _hsv_mask(hsv, h_low, h_high, s_min, v_min):
    """Buat HSV mask untuk satu range hue."""
    return cv2.inRange(hsv, (h_low, s_min, v_min), (h_high, 255, 255))


def detect_hero_dots(hsv, gray):
    """
    Deteksi hero dots menggunakan HSV color mask + contour + multi-filter.

    Returns:
        list of (class_id, cx, cy)
    """
    height, width = hsv.shape[:2]

    # ── Blue mask ──
    mask_blue = _hsv_mask(hsv, BLUE_H_LOW, BLUE_H_HIGH, BLUE_S_MIN, BLUE_V_MIN)

    # ── Red mask (dual range untuk wraparound) ──
    mask_red = np.zeros((height, width), dtype=np.uint8)
    for hl, hh in RED_RANGES:
        part = _hsv_mask(hsv, hl, hh, RED_S_MIN, RED_V_MIN)
        mask_red = cv2.bitwise_or(mask_red, part)

    results = []

    for class_id, mask in [(0, mask_blue), (1, mask_red)]:
        # Morphological close untuk sambung ring yang putus
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # ── Filter 1: Area ──
            if area < AREA_MIN or area > AREA_MAX:
                continue

            # ── Filter 2: Circularity ──
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circ = 4.0 * np.pi * area / (perimeter * perimeter)
            if circ < CIRCULARITY_MIN:
                continue

            # ── Centroid ──
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))

            # Bounds check
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue

            # ── Filter 3: Face brightness ──
            # Ambil circular patch di tengah
            face_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(face_mask, (cx, cy), FACE_RADIUS, 255, -1)
            face_pixels = gray[face_mask == 255]
            if len(face_pixels) < 10:
                continue
            face_mean = np.mean(face_pixels)
            if face_mean < FACE_BRIGHTNESS_MIN:
                continue

            # ── Filter 4: Ring saturation (mean S of ring area) ──
            ring_mask = np.zeros((height, width), dtype=np.uint8)
            outer_r = FACE_RADIUS + 6
            cv2.circle(ring_mask, (cx, cy), outer_r, 255, -1)
            cv2.circle(ring_mask, (cx, cy), FACE_RADIUS, 0, -1)
            ring_s_pixels = hsv[:, :, 1][ring_mask == 255]
            if len(ring_s_pixels) < 10:
                continue
            ring_s_mean = np.mean(ring_s_pixels)
            if ring_s_mean < RING_SAT_MIN:
                continue

            # ── Filter 5: Ring harus lebih jenuh dari face (hero ring vs background) ──
            face_s_pixels = hsv[:, :, 1][face_mask == 255]
            face_s_mean = np.mean(face_s_pixels) if len(face_s_pixels) > 0 else 0
            if ring_s_mean < face_s_mean + 20:
                # Ring harus significantly lebih jenuh dari face
                continue

            results.append((class_id, cx, cy))

    return results


def deduplicate(results, min_dist=OVERLAP_MIN_DIST):
    """Hapus deteksi duplikat yang terlalu berdekatan (satu hero terdeteksi 2x)."""
    if len(results) <= 1:
        return results
    kept = []
    # Sort by class
    for cls in [0, 1]:
        cls_results = [(c, x, y) for c, x, y in results if c == cls]
        for r in cls_results:
            _, cx, cy = r
            too_close = False
            for _, kx, ky in kept:
                if np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2) < min_dist:
                    too_close = True
                    break
            if not too_close:
                kept.append(r)
    return kept


def save_labels(img_name: str, boxes: list, split: str = "train"):
    """Simpan YOLO format labels."""
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
    lines = []
    for cls, cx, cy in boxes:
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = HERO_DOT_SIZE / IMG_W
        nh = HERO_DOT_SIZE / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def draw_preview(mm, results):
    """Buat visual preview deteksi."""
    vis = mm.copy()
    vis = cv2.resize(vis, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
    colors = {0: (255, 180, 30), 1: (30, 30, 255)}
    labels = {0: "blue_hero", 1: "red_hero"}
    for cls, cx, cy in results:
        color = colors[cls]
        sx, sy = int(cx * 2), int(cy * 2)
        r = HERO_DOT_SIZE // 2 * 2
        cv2.circle(vis, (sx, sy), r, color, 3)
        cv2.circle(vis, (sx, sy), 3, (255, 255, 255), -1)
        cv2.putText(vis, labels[cls], (sx - 35, sy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def get_existing_auto_labels(video_stem):
    """Cari frame auto-label yang sudah ada."""
    existing = set()
    pattern = f"{AUTO_PREFIX}{video_stem}_frame_"
    label_dir = DATASET_DIR / "labels" / "train"
    if not label_dir.exists():
        return existing
    for f in label_dir.glob(f"{pattern}*.txt"):
        try:
            idx_str = f.stem.replace(pattern, "")
            existing.add(int(idx_str))
        except ValueError:
            continue
    return existing


def main():
    ap = argparse.ArgumentParser(
        description="Auto-label MLBB minimap hero dots via HSV + contour + multi-filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python tools/auto_label_minimap.py --video alpha_1.mp4\n  python tools/auto_label_minimap.py --all-videos",
    )
    ap.add_argument("--video", "-v", type=str, default=None, help="Video file name")
    ap.add_argument("--all-videos", action="store_true", help="Process all .mp4 videos")
    ap.add_argument("--preview", "-p", action="store_true", help="Show preview window")
    ap.add_argument("--dry-run", action="store_true", help="Don't save files")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="Skip existing auto-labels")
    ap.add_argument("--max-frames", type=int, default=0, help="Max frames to save per video")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help=f"Process every N frames (default: {FRAME_SKIP})")
    args = ap.parse_args()

    # Setup output dirs
    os.makedirs(DATASET_DIR / "images" / "train", exist_ok=True)
    os.makedirs(DATASET_DIR / "labels" / "train", exist_ok=True)

    # Dapatkan daftar video
    if args.video:
        video_path = VIDEO_DIR / args.video
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        video_files = [video_path]
    elif args.all_videos:
        video_files = sorted(VIDEO_DIR.glob("*.mp4"))
        if not video_files:
            print(f"❌ No .mp4 videos found in {VIDEO_DIR.absolute()}")
            return
    else:
        ap.print_help()
        print("\n\nSpecify --video or --all-videos")
        return

    print("=" * 60)
    print("🎯 Auto Label Minimap — HSV + Contour + Multi-Filter")
    print("=" * 60)
    print(f"  Output:     trainings/hero_detector/ (train split)")
    print(f"  Prefix:     {AUTO_PREFIX}")
    print(f"  Classes:    0 = blue_hero, 1 = red_hero")
    print(f"  Filters:    area=[{AREA_MIN}-{AREA_MAX}], circularity>={CIRCULARITY_MIN}")
    print(f"              faceBrightness>={FACE_BRIGHTNESS_MIN}, ringSat>={RING_SAT_MIN}")
    print(f"              ringSat > faceSat + 20 (ring contrast)")
    print(f"  Preview:    {'ON' if args.preview else 'OFF'}")
    print(f"  Dry-run:    {'YES' if args.dry_run else 'NO'}")
    print(f"  Frame skip: every {args.frame_skip} frame(s)")
    print(f"  Videos:     {len(video_files)} file(s)")
    print("-" * 60)

    total_frames_saved = 0
    total_dots = 0

    for vid_idx, vid in enumerate(video_files):
        print(f"\n📹 [{vid_idx + 1}/{len(video_files)}] Processing: {vid.name}")

        existing = get_existing_auto_labels(vid.stem) if args.skip_existing else set()
        if existing:
            print(f"   📋 Existing: {len(existing)} frames")

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"   ❌ Cannot open {vid.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   Total frames: {total_frames}, FPS: {fps:.1f}")

        skip_frames = int(fps * 5) if fps > 0 else 150
        frame_idx = 0
        saved = 0
        empty = 0
        skipped = 0
        frame_dots = 0
        max_frames = args.max_frames or float("inf")

        # Skip intro
        while frame_idx < skip_frames and frame_idx < total_frames - 1:
            cap.read()
            frame_idx += 1

        if args.preview:
            cv2.namedWindow("Auto Label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Auto Label", IMG_W * 2 + 20, IMG_H * 2 + 60)

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in existing:
                skipped += 1
                frame_idx += 1
                continue
            if frame_idx % args.frame_skip != 0:
                frame_idx += 1
                continue

            mm = crop_region(frame, "map")
            if mm is None or mm.shape[:2] != (IMG_H, IMG_W):
                frame_idx += 1
                continue

            hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

            results = detect_hero_dots(hsv, gray)
            results = deduplicate(results)

            img_name = f"{AUTO_PREFIX}{vid.stem}_frame_{frame_idx}.png"

            if not results:
                empty += 1
                if args.preview:
                    vis = draw_preview(mm, results)
                    cv2.putText(vis, f"[AUTO] Frame {frame_idx} — No heroes", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.imshow("Auto Label", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            counts = Counter(c for c, _, _ in results)

            if not args.dry_run:
                cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                save_labels(img_name, [(c, x, y) for c, x, y in results])

            saved += 1
            frame_dots += len(results)
            total_dots += len(results)

            if args.preview:
                vis = draw_preview(mm, results)
                info = " | ".join(f"{'blue' if c==0 else 'red'}={n}" for c, n in sorted(counts.items()))
                cv2.putText(vis, f"[AUTO] Frame {frame_idx} — {info}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Auto Label", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if saved >= max_frames:
                break

        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

        print(f"   ✅ {saved} saved, {empty} empty, {skipped} skipped, {frame_dots} dots")
        total_frames_saved += saved

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  Total frames:  {total_frames_saved}")
    print(f"  Total dots:    {total_dots}")
    if total_frames_saved:
        print(f"  Avg/frame:     {total_dots / total_frames_saved:.1f}")
    if not args.dry_run:
        auto_count = len(list((DATASET_DIR / "images" / "train").glob(f"{AUTO_PREFIX}*.png")))
        total_imgs = len(list((DATASET_DIR / "images" / "train").glob("*.png")))
        print(f"  Dataset: {total_imgs} images ({total_imgs - auto_count} manual, {auto_count} auto)")
    print("=" * 60)


if __name__ == "__main__":
    main()

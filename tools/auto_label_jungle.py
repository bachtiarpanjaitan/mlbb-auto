"""
Auto Labeling Tool — Jungle Camps untuk MLBB minimap.

Jungle camp punya posisi STATIS di minimap, jadi auto-labelnya simpel:
  Untuk tiap frame → cek warna dot di tiap fixed position → label YOLO

Ini bisa di-run SEMUA frame (tidak perlu skip tiap 30 frame) karena:
  - Cuma 14 posisi yang dicek per frame
  - Sangat cepat (tidak perlu model inference)

Usage:
  python tools/auto_label_jungle.py --video alpha_1.mp4
  python tools/auto_label_jungle.py --all-videos
  python tools/auto_label_jungle.py --all-videos --preview
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
BOX_SIZE = 34

AUTO_PREFIX = "auto_jngl_"

# Template matching threshold
TEMPLATE_MATCH_THRESHOLD = 0.18
TEMPLATE_THRESHOLD_OVERRIDES = {
    2: 0.30,  # lord — threshold lebih tinggi: hindari countdown timer
}

# ── Load Templates ──
# Map class_id ke template image
TEMPLATE_DIR = Path("assets/creeps_minimap")
TEMPLATES = {}
for t_path in sorted(TEMPLATE_DIR.glob("*.png")):
    name = t_path.stem
    if name == "crab":
        TEMPLATES[7] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "fire_beetle":
        TEMPLATES[9] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "horned_lizard":
        TEMPLATES[10] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lava_golem":
        TEMPLATES[8] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lithowanderer":
        TEMPLATES[6] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lord":
        TEMPLATES[2] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "molten_fiend":
        TEMPLATES[5] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "thunder_fenrir":
        TEMPLATES[4] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)

# ── Camp Positions (PIXEL) — Derived from manual data ──
JUNGLE_CAMPS = [
    # Lord (class 2) — 2 pit (dari manual labels)
    {"id": "lord_top", "cls": 2, "px": 113, "py": 96},
    {"id": "lord_bot", "cls": 2, "px": 241, "py": 241},
    # Turtle (class 3) — tanpa template, skip auto-label
    # Thunder Fenrir / Blue Buff (class 4)
    {"id": "thunder_fenrir_a", "cls": 4, "px": 91, "py": 175},
    {"id": "thunder_fenrir_b", "cls": 4, "px": 263, "py": 162},
    # Molten Fiend / Red Buff (class 5)
    {"id": "molten_fiend_a", "cls": 5, "px": 187, "py": 72},
    {"id": "molten_fiend_b", "cls": 5, "px": 165, "py": 264},
    # Lithowanderer (class 6) — 2 posisi (sungai tengah atas & bawah)
    {"id": "lithowanderer_a", "cls": 6, "px": 139, "py": 132},
    {"id": "lithowanderer_b", "cls": 6, "px": 211, "py": 208},
    # Crab (class 7) — 2 sisi
    {"id": "crab_a", "cls": 7, "px": 69, "py": 88},
    {"id": "crab_b", "cls": 7, "px": 282, "py": 249},
    # Lava Golem (class 8) — 2 sisi
    {"id": "lava_golem_a", "cls": 8, "px": 174, "py": 89},
    {"id": "lava_golem_b", "cls": 8, "px": 179, "py": 247},
    # Fire Beetle (class 9) — 2 sisi
    {"id": "fire_beetle_a", "cls": 9, "px": 134, "py": 61},
    {"id": "fire_beetle_b", "cls": 9, "px": 218, "py": 276},
    # Horned Lizard (class 10) — 2 sisi
    {"id": "horned_lizard_a", "cls": 10, "px": 62, "py": 144},
    {"id": "horned_lizard_b", "cls": 10, "px": 290, "py": 192},
]


def detect_jungle_camps(img_bgr):
    """
    Deteksi jungle camps pakai template matching.

    Untuk tiap camp:
      1. Crop patch 34x34 di posisi camp
      2. Template match dengan template camp yang sesuai
      3. Jika match score > threshold, camp hidup

    Returns:
        list of (class_id, cx, cy)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    results = []

    for camp in JUNGLE_CAMPS:
        cx = camp["px"]
        cy = camp["py"]
        cls_id = camp["cls"]

        half = BOX_SIZE // 2
        if not (half < cx < w - half and half < cy < h - half):
            continue

        # Crop patch di sekitar camp (grayscale)
        patch = gray[cy - half:cy + half, cx - half:cx + half]
        if patch.shape != (BOX_SIZE, BOX_SIZE):
            continue

        # Template matching — hanya camp dengan template yang dideteksi
        if cls_id not in TEMPLATES or TEMPLATES[cls_id] is None:
            continue  # Skip camp tanpa template (false positive risk)

        template = TEMPLATES[cls_id]
        t_h, t_w = template.shape[:2]

        # Per-class threshold (override untuk camp tertentu)
        threshold = TEMPLATE_THRESHOLD_OVERRIDES.get(cls_id, TEMPLATE_MATCH_THRESHOLD)

        if t_h <= BOX_SIZE and t_w <= BOX_SIZE:
            result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            center_sat = np.mean(hsv[cy - 3:cy + 3, cx - 3:cx + 3, 1])
            if max_val >= threshold and center_sat >= 100:
                results.append((cls_id, cx, cy))
        else:
            result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0]
            center_sat = np.mean(hsv[cy - 3:cy + 3, cx - 3:cx + 3, 1])
            if score >= threshold and center_sat >= 100:
                results.append((cls_id, cx, cy))

    return results


def save_labels(img_name: str, boxes: list, split: str = "train"):
    """Simpan YOLO format labels. Append ke file kalo sudah ada."""
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
    lines = []
    for cls, cx, cy in boxes:
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = BOX_SIZE / IMG_W
        nh = BOX_SIZE / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def merge_into_existing_labels(img_name: str, new_boxes: list, split: str = "train"):
    """
    Merge jungle labels ke file label yang sudah ada (hero labels),
    tanpa menghapus class lain.
    """
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"

    # Baca existing labels
    existing_lines = []
    if label_path.exists():
        existing_lines = [l.strip() for l in label_path.read_text().strip().split("\n") if l.strip()]

    # Parse existing class IDs
    existing_classes = set()
    for line in existing_lines:
        parts = line.split()
        if parts:
            existing_classes.add(int(parts[0]))

    # Tambah yang baru (hanya jungle class yang belum ada)
    new_lines = []
    for cls, cx, cy in new_boxes:
        if cls not in existing_classes:
            nx = cx / IMG_W
            ny = cy / IMG_H
            nw = BOX_SIZE / IMG_W
            nh = BOX_SIZE / IMG_H
            new_lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

    if not new_lines:
        return False  # Tidak ada yang baru

    combined = existing_lines + new_lines
    label_path.write_text("\n".join(combined) + "\n")
    return True


def draw_preview(mm, results):
    """Visual preview overlay."""
    vis = cv2.resize(mm.copy(), (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)

    # CLASS_COLORS dari label_minimap.py
    colors = {
        2: (255, 0, 255),    # Lord (Magenta)
        3: (0, 255, 0),      # Turtle (Green)
        4: (255, 255, 0),    # Thunder Fenrir (Cyan)
        5: (0, 140, 255),    # Molten Fiend (Orange)
        6: (200, 255, 0),    # Lithowanderer
        7: (0, 215, 255),    # Crab (Gold)
        8: (180, 0, 180),    # Lava Golem (Purple)
        9: (0, 100, 255),    # Fire Beetle
        10: (255, 100, 200), # Horned Lizard (Pink)
    }
    names = {
        2: "lord", 3: "turtle", 4: "thunder_fenrir", 5: "molten_fiend",
        6: "lithowanderer", 7: "crab", 8: "lava_golem", 9: "fire_beetle", 10: "horned_lizard",
    }

    for cls, cx, cy in results:
        color = colors.get(cls, (200, 200, 200))
        name = names.get(cls, f"cls_{cls}")
        sx, sy = int(cx * 2), int(cy * 2)
        r = BOX_SIZE // 2 * 2
        cv2.circle(vis, (sx, sy), r, color, 3)
        cv2.circle(vis, (sx, sy), 3, (255, 255, 255), -1)
        cv2.putText(vis, name, (sx - 30, sy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    return vis


def main():
    ap = argparse.ArgumentParser(
        description="Auto-label jungle camps on minimap (fixed position + HSV check)",
    )
    ap.add_argument("--video", "-v", type=str, default=None, help="Video file name")
    ap.add_argument("--all-videos", action="store_true", help="Process all .mp4 videos")
    ap.add_argument("--preview", "-p", action="store_true", help="Show preview window")
    ap.add_argument("--dry-run", action="store_true", help="Don't save files")
    ap.add_argument("--merge", action="store_true", default=True,
                    help="Merge dengan label hero yang sudah ada (default: True)")
    ap.add_argument("--frame-skip", type=int, default=15,
                    help="Process every N frames (default: 15 ≈ 2fps)")
    args = ap.parse_args()

    os.makedirs(DATASET_DIR / "images" / "train", exist_ok=True)
    os.makedirs(DATASET_DIR / "labels" / "train", exist_ok=True)

    # Video list
    if args.video:
        video_path = VIDEO_DIR / args.video
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        video_files = [video_path]
    elif args.all_videos:
        video_files = sorted(VIDEO_DIR.glob("*.mp4"))
        if not video_files:
            print(f"❌ No .mp4 videos found")
            return
    else:
        ap.print_help()
        print("\nSpecify --video or --all-videos")
        return

    print("=" * 60)
    print("🌿 Auto Label Jungle — Fixed Position + HSV Color Check")
    print("=" * 60)
    print(f"  Camps:  {len(JUNGLE_CAMPS)} positions")
    print(f"  Classes: 2-10 (lord, turtle, buffs, creeps)")
    print(f"  Merge:  {'Yes (add to existing hero labels)' if args.merge else 'No (separate files)'}")
    print(f"  Videos: {len(video_files)} file(s)")
    print("-" * 60)

    total_camps_detected = 0
    total_frames_with_jungle = 0
    class_total = Counter()

    for vid_idx, vid in enumerate(video_files):
        print(f"\n📹 [{vid_idx + 1}/{len(video_files)}] Processing: {vid.name}")

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"   ❌ Cannot open {vid.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = args.frame_skip
        print(f"   Frames: {total_frames}, FPS: {fps:.1f}")

        # Skip intro ~5 detik
        skip_frames = int(fps * 5) if fps > 0 else 150

        frame_idx = 0
        frame_checked = 0
        frame_with_camps = 0
        total_camps = 0

        # Skip intro
        while frame_idx < skip_frames and frame_idx < total_frames - 1:
            cap.read()
            frame_idx += 1

        if args.preview:
            cv2.namedWindow("Jungle Auto-Label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Jungle Auto-Label", IMG_W * 2 + 20, IMG_H * 2 + 60)

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            mm = crop_region(frame, "map")
            if mm is None or mm.shape[:2] != (IMG_H, IMG_W):
                frame_idx += 1
                continue

            hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

            results = detect_jungle_camps(mm)

            frame_checked += 1

            if results:
                frame_with_camps += 1
                total_camps += len(results)

                counts = Counter(c for c, _, _ in results)
                class_total.update(counts)

                img_name = f"{vid.stem}_frame_{frame_idx}.png"

                if not args.dry_run:
                    if args.merge:
                        # Merge ke file label yang sudah ada
                        merge_into_existing_labels(img_name, results)
                        # Image juga harus ada (hero auto-label seharusnya sudah simpan)
                        if not (DATASET_DIR / "images" / "train" / img_name).exists():
                            cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                    else:
                        # Simpan terpisah dengan prefix
                        jngl_name = f"{AUTO_PREFIX}{img_name}"
                        cv2.imwrite(str(DATASET_DIR / "images" / "train" / jngl_name), mm)
                        save_labels(jngl_name, [(c, x, y) for c, x, y in results])

                if args.preview:
                    vis = draw_preview(mm, results)
                    info = " | ".join(f"{n}={counts[c]}" for c, n in
                                      [(2, "lord"), (3, "turtle"), (4, "b_buff"), (5, "r_buff"),
                                       (6, "litho"), (7, "crab"), (8, "golem"), (9, "beetle"), (10, "lizard")] if c in counts)
                    cv2.putText(vis, f"[JNGL] Frame {frame_idx} — {info}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Jungle Auto-Label", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_idx += 1

        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

        print(f"   Checked {frame_checked} frames, {frame_with_camps} with camps, {total_camps} total detections")

        total_camps_detected += total_camps
        total_frames_with_jungle += frame_with_camps

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  Frames checked: {total_frames_with_jungle} (with jungle camps)")
    print(f"  Total camp detections: {total_camps_detected}")
    print(f"  Per class:")
    for cls_id in sorted(class_total.keys()):
        names = {2: "lord", 3: "turtle", 4: "thunder_fenrir", 5: "molten_fiend",
                 6: "lithowanderer", 7: "crab", 8: "lava_golem", 9: "fire_beetle", 10: "horned_lizard"}
        print(f"    {cls_id}: {names.get(cls_id, '?')}: {class_total[cls_id]}")
    print(f"\n  Mode: {'Merged into existing labels' if args.merge else 'Separate auto_ files'}")
    if not args.dry_run:
        print(f"  Next: python tools/train_yolo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

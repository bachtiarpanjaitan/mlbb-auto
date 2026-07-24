"""
Auto Label Full — Hero + Jungle auto-labeling dalam satu tool.

Menggabungkan:
  - Hero detection: YOLO model (blue_hero class 0, red_hero class 1)
  - Jungle detection: Template matching di fixed position (class 2-10)

Output disimpan dengan prefix "full_" — tidak menyentuh file existing.

Usage:
  python tools/auto_label_full.py --video alpha_1.mp4
  python tools/auto_label_full.py --all-videos
  python tools/auto_label_full.py --all-videos --conf 0.85
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

# Model paths
MODEL_ONNX = Path("models/hero_tracker.onnx")
MODEL_PT = Path("trainings/hero_detector") / "yolo11n_minimap" / "weights" / "best.pt"

# YOLO confidence threshold
YOLO_CONF = 0.85

# Template matching threshold
TEMPLATE_MATCH_THRESHOLD = 0.18

# Threshold override per class
TEMPLATE_THRESHOLD_OVERRIDES = {
    2: 0.30,  # lord: hindari countdown timer
}

# Frame skip
FRAME_SKIP = 30

# Output prefix — tidak bentrok dengan file existing
OUTPUT_PREFIX = "full_"

# ── Jungle Camp Templates ──
TEMPLATE_DIR = Path("assets/creeps_minimap")
TEMPLATES = {}
for t_path in sorted(TEMPLATE_DIR.glob("*.png")):
    name = t_path.stem
    if name == "crab":      TEMPLATES[7] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "fire_beetle":   TEMPLATES[9] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "horned_lizard": TEMPLATES[10] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lava_golem":    TEMPLATES[8] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lithowanderer": TEMPLATES[6] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "lord":     TEMPLATES[2] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "molten_fiend":  TEMPLATES[5] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    elif name == "thunder_fenrir": TEMPLATES[4] = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)

# ── Jungle Camp Fixed Positions (from manual data) ──
JUNGLE_CAMPS = [
    # Lord (class 2) — 2 pit
    {"id": "lord_top", "cls": 2, "px": 113, "py": 96},
    {"id": "lord_bot", "cls": 2, "px": 241, "py": 241},
    # Thunder Fenrir (class 4) — 2 sisi
    {"id": "thunder_fenrir_a", "cls": 4, "px": 91, "py": 175},
    {"id": "thunder_fenrir_b", "cls": 4, "px": 263, "py": 162},
    # Molten Fiend (class 5) — 2 sisi
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


# ═══════════════════════════════════════════
#  HERO DETECTION (YOLO)
# ═══════════════════════════════════════════

def load_yolo():
    """Load YOLO model — prefer ONNX."""
    if MODEL_ONNX.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(MODEL_ONNX))
            input_name = session.get_inputs()[0].name
            _, _, model_h, model_w = session.get_inputs()[0].shape
            return ("onnx", session, input_name, model_w, model_h)
        except Exception:
            pass
    if MODEL_PT.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(MODEL_PT))
            return ("pt", model, None, None, None)
        except Exception:
            pass
    return None


def detect_heroes_yolo(model_info, mm):
    """
    YOLO inference untuk hero (class 0 dan 1).

    Returns:
        list of (class_id, cx, cy)
    """
    backend = model_info[0]

    if backend == "onnx":
        session, input_name, model_w, model_h = model_info[1], model_info[2], model_info[3], model_info[4]

        img = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (model_w, model_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]

        out = session.run(None, {input_name: img})[0][0]
        scale_x = IMG_W / model_w
        scale_y = IMG_H / model_h

        boxes = []
        for i in range(out.shape[1]):
            scores = out[4:, i]
            max_cls = int(np.argmax(scores))
            ms = float(scores[max_cls])
            if max_cls not in {0, 1} or ms < 0.1:
                continue
            cx, cy = float(out[0, i]), float(out[1, i])
            bw, bh = float(out[2, i]), float(out[3, i])
            if bw < 3 or bh < 3:
                continue
            if not (0 <= cx < model_w and 0 <= cy < model_h):
                continue
            boxes.append((max_cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, ms))
    else:
        # PyTorch fallback
        from ultralytics import YOLO
        model = model_info[1]
        results_pt = model(mm, verbose=False, conf=0.01, iou=0.5)
        boxes = []
        for r in results_pt:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                if cls_id not in {0, 1}:
                    continue
                cx, cy = (r.boxes.xyxy[i][0].item() + r.boxes.xyxy[i][2].item()) / 2, \
                         (r.boxes.xyxy[i][1].item() + r.boxes.xyxy[i][3].item()) / 2
                bw, bh = r.boxes.xyxy[i][2].item() - r.boxes.xyxy[i][0].item(), \
                         r.boxes.xyxy[i][3].item() - r.boxes.xyxy[i][1].item()
                if bw >= 3 and bh >= 3:
                    boxes.append((cls_id, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(r.boxes.conf[i].item())))

    if not boxes:
        return []

    # NMS for hero detection
    kept = []
    for cls_id in [0, 1]:
        cb = [b for b in boxes if b[0] == cls_id]
        cb.sort(key=lambda b: b[5], reverse=True)
        while cb:
            best = cb.pop(0)
            _, bx1, by1, bx2, by2, _ = best
            cx_mm = int(((bx1 + bx2) / 2) * scale_x) if backend == "onnx" else int(bx1 + bx2 / 2)
            cy_mm = int(((by1 + by2) / 2) * scale_y) if backend == "onnx" else int(by1 + by2 / 2)
            cx_mm = max(0, min(IMG_W - 1, cx_mm))
            cy_mm = max(0, min(IMG_H - 1, cy_mm))
            kept.append((cls_id, cx_mm, cy_mm))

            remaining = []
            for b in cb:
                _, x1, y1, x2, y2, _ = b
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (bx2 - bx1) * (by2 - by1) + (x2 - x1) * (y2 - y1) - inter
                    if union > 0 and inter / union > 0.3:
                        continue
                remaining.append(b)
            cb = remaining

    return kept


# ═══════════════════════════════════════════
#  JUNGLE DETECTION (Template Matching)
# ═══════════════════════════════════════════

def detect_jungle(mm):
    """
    Template matching untuk jungle camp di fixed position.

    Returns:
        list of (class_id, cx, cy)
    """
    h, w = mm.shape[:2]
    gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    results = []

    for camp in JUNGLE_CAMPS:
        cx, cy, cls_id = camp["px"], camp["py"], camp["cls"]

        if not (BOX_SIZE // 2 < cx < w - BOX_SIZE // 2 and BOX_SIZE // 2 < cy < h - BOX_SIZE // 2):
            continue

        if cls_id not in TEMPLATES or TEMPLATES[cls_id] is None:
            continue

        half = BOX_SIZE // 2
        patch = gray[cy - half:cy + half, cx - half:cx + half]
        if patch.shape != (BOX_SIZE, BOX_SIZE):
            continue

        template = TEMPLATES[cls_id]
        t_h, t_w = template.shape[:2]
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


# ═══════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════

def save_labels(img_name: str, boxes: list):
    """Simpan YOLO format labels."""
    label_path = DATASET_DIR / "labels" / "train" / f"{Path(img_name).stem}.txt"
    lines = []
    for cls, cx, cy in boxes:
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = BOX_SIZE / IMG_W
        nh = BOX_SIZE / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Auto-label heroes + jungle camps (YOLO + template matching)")
    ap.add_argument("--video", "-v", type=str, default=None, help="Video file name")
    ap.add_argument("--all-videos", action="store_true", help="Process all .mp4 videos")
    ap.add_argument("--conf", type=float, default=YOLO_CONF, help=f"YOLO confidence (default: {YOLO_CONF})")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help=f"Every N frames (default: {FRAME_SKIP})")
    ap.add_argument("--dry-run", action="store_true", help="Don't save files")
    ap.add_argument("--preview", "-p", action="store_true", help="Show preview")
    args = ap.parse_args()

    # Setup
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
        # Interactive video picker (sama seperti debug_vision)
        cs = sorted(VIDEO_DIR.glob("*.mp4"))
        if not cs:
            print(f"❌ No .mp4 videos found in {VIDEO_DIR}")
            return
        print("\n🎬 Pilih video untuk auto-label:")
        for idx, vid in enumerate(cs, 1):
            size_mb = vid.stat().st_size / (1024 * 1024)
            print(f"  [{idx}] {vid.name} ({size_mb:.1f} MB)")
        print(f"  [a] Semua video ({len(cs)} file)")
        while True:
            try:
                choice = input(f"\nMasukkan nomor video (1-{len(cs)}) [a]: ").strip().lower()
                if not choice or choice == 'a':
                    video_files = cs
                    break
                selected_idx = int(choice) - 1
                if 0 <= selected_idx < len(cs):
                    video_files = [cs[selected_idx]]
                    break
                print(f"❌ Masukkan angka 1-{len(cs)} atau 'a'")
            except (ValueError, EOFError):
                video_files = cs
                break

    # Load YOLO
    print("📦 Loading YOLO model...")
    yolo = load_yolo()
    if yolo is None:
        print("❌ No YOLO model found.")
        return
    print(f"   ✅ YOLO ready ({yolo[0].upper()})")

    print("=" * 60)
    print(f"🤖 Full Auto-Label — Hero (YOLO) + Jungle (Template)")
    print("=" * 60)
    print(f"  Output:      {DATASET_DIR.absolute()}/ (train split)")
    print(f"  Prefix:      {OUTPUT_PREFIX} (tidak menyentuh file existing)")
    print(f"  YOLO conf:   ≥{args.conf}")
    print(f"  Jungle:      {len(JUNGLE_CAMPS)} positions (template match)")
    print(f"  Frame skip:  every {args.frame_skip} frame(s)")
    print(f"  Videos:      {len(video_files)} file(s)")
    print("-" * 60)

    total_saved = 0
    total_heroes = 0
    total_jungle = 0
    per_class = Counter()

    for vid_idx, vid in enumerate(video_files):
        print(f"\n📹 [{vid_idx + 1}/{len(video_files)}] Processing: {vid.name}")

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"   ❌ Cannot open {vid.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   Frames: {total_frames}, FPS: {fps:.1f}")

        skip_frames = int(fps * 5) if fps > 0 else 150
        frame_idx = 0
        saved = 0
        empty = 0
        frame_heroes = 0
        frame_jungle = 0

        # Skip intro
        while frame_idx < skip_frames and frame_idx < total_frames - 1:
            cap.read()
            frame_idx += 1

        if args.preview:
            cv2.namedWindow("Full Auto-Label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Full Auto-Label", IMG_W * 2 + 20, IMG_H * 2 + 60)

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % args.frame_skip != 0:
                frame_idx += 1
                continue

            mm = crop_region(frame, "map")
            if mm is None or mm.shape[:2] != (IMG_H, IMG_W):
                frame_idx += 1
                continue

            # ── Hero detection ──
            heroes = detect_heroes_yolo(yolo, mm)

            # Filter by confidence
            # (YOLO output already filtered in detect_heroes_yolo)
            hero_boxes = [(cls, cx, cy) for cls, cx, cy in heroes]

            # ── Jungle detection ──
            jungle_boxes = detect_jungle(mm)

            # ── Combine ──
            all_boxes = hero_boxes + jungle_boxes

            if not all_boxes:
                empty += 1
                if args.preview:
                    vis = mm.copy()
                    vis = cv2.resize(vis, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(vis, f"[FULL] Frame {frame_idx} — No detections", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.imshow("Full Auto-Label", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            img_name = f"{OUTPUT_PREFIX}{vid.stem}_frame_{frame_idx}.png"

            if not args.dry_run:
                cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                save_labels(img_name, all_boxes)

            saved += 1
            frame_heroes += len(hero_boxes)
            frame_jungle += len(jungle_boxes)
            total_heroes += len(hero_boxes)
            total_jungle += len(jungle_boxes)
            for cls, _, _ in all_boxes:
                per_class[cls] += 1

            # Preview
            if args.preview:
                # Simple overlay
                colors = {0: (255, 180, 30), 1: (30, 30, 255), 2: (255, 0, 255), 3: (0, 255, 0),
                          4: (255, 255, 0), 5: (0, 140, 255), 6: (200, 255, 0), 7: (0, 215, 255),
                          8: (180, 0, 180), 9: (0, 100, 255), 10: (255, 100, 200)}
                names = {0:'blue',1:'red',2:'lord',3:'turtle',4:'thunder',5:'molten',6:'litho',
                         7:'crab',8:'golem',9:'beetle',10:'lizard'}
                vis = cv2.resize(mm.copy(), (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
                for cls, cx, cy in all_boxes:
                    c = colors.get(cls, (200, 200, 200))
                    sx, sy = int(cx * 2), int(cy * 2)
                    cv2.circle(vis, (sx, sy), 17, c, 2)
                    cv2.putText(vis, names.get(cls, '?'), (sx - 15, sy - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
                cv2.putText(vis, f"[FULL] Frame {frame_idx} — {len(hero_boxes)} heroes + {len(jungle_boxes)} jungle", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Full Auto-Label", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

        print(f"   ✅ {saved} saved, {empty} empty, {frame_heroes} heroes, {frame_jungle} jungle")
        total_saved += saved

    # Summary
    names = {0:'blue_hero',1:'red_hero',2:'lord',4:'thunder_fenrir',5:'molten_fiend',
             6:'lithowanderer',7:'crab',8:'lava_golem',9:'fire_beetle',10:'horned_lizard'}
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY — {total_saved} frames, {total_heroes} heroes + {total_jungle} jungle")
    print("=" * 60)
    for cls in sorted(per_class.keys()):
        name = names.get(cls, '?')
        print(f"  {cls:2d} {name:20s} {per_class[cls]:6d}")
    print(f"\n  Prefix: {OUTPUT_PREFIX} (tidak menyentuh file existing)")
    if not args.dry_run:
        total_f = len(list((DATASET_DIR / "images" / "train").glob(f"{OUTPUT_PREFIX}*.png")))
        print(f"  Total files: {total_f}")
        print(f"  Dir: {DATASET_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

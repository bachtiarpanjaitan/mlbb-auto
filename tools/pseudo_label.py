"""
Pseudo-Labeling Tool — MLBB minimap hero auto-labeling menggunakan YOLO model.

Menggunakan model YOLO yang sudah ada untuk inference pada frame baru,
lalu mengambil prediksi high-confidence sebagai auto-label untuk
memperluas dataset training (self-training / iterative training).

Pipeline:
  Model YOLO → inference on frame → filter high-conf → save as auto-label → retrain

Ideal untuk iterative self-training loop:
  1. python tools/pseudo_label.py --all-videos --conf 0.85
  2. python tools/train_yolo.py
  3. Ulangi langkah 1-2 (model baru lebih akurat → lebih banyak auto-label berkualitas)

Usage:
  python tools/pseudo_label.py --video alpha_1.mp4
  python tools/pseudo_label.py --all-videos --conf 0.9
  python tools/pseudo_label.py --all-videos --preview
"""

import cv2
import numpy as np
import os
import sys
import json
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

# Model paths
MODEL_ONNX = Path("models/hero_tracker.onnx")
MODEL_PT = Path("trainings/hero_detector") / "yolo11n_minimap" / "weights" / "best.pt"

# Default confidence threshold
DEFAULT_CONF = 0.15  # Hero detection masih rendah (max ~0.25)

# Frame skip (default: ~1 fps)
FRAME_SKIP = 30

# File prefix untuk auto-labeled
AUTO_PREFIX = "auto_"

# Class filter: hanya hero (0=blue_hero, 1=red_hero)
HERO_CLASSES = {0}  # Model baru: hero = class 0

# Nama class untuk display
CLASS_NAMES = {0: "blue_hero", 1: "red_hero"}


def load_yolo():
    """Load YOLO model — prefer ONNX, fallback ke ultralytics."""
    # Coba ONNX dulu
    if MODEL_ONNX.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(MODEL_ONNX))
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)
            _, _, model_h, model_w = input_shape
            print(f"   Loaded ONNX model: {MODEL_ONNX.name} ({model_w}x{model_h})")
            return ("onnx", session, input_name, model_w, model_h)
        except Exception as e:
            print(f"   ⚠️  ONNX load failed: {e}")

    # Fallback ke ultralytics .pt
    if MODEL_PT.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(MODEL_PT))
            print(f"   Loaded PyTorch model: {MODEL_PT.name}")
            return ("pt", model, None, None, None)
        except Exception as e:
            print(f"   ⚠️  PyTorch load failed: {e}")

    return None


def run_inference_onnx(session, input_name, mm, model_w, model_h):
    """ONNX inference on minimap crop with proper NMS."""
    # YOLO ONNX format: (1, 4+num_classes, num_predictions)
    # Row 0-3 = bbox (cx, cy, w, h) in image pixel coordinates (0..model_w/h)
    # Row 4+ = class scores

    img = cv2.cvtColor(mm, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_w, model_h))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    out = outputs[0][0]

    scale_x = IMG_W / model_w
    scale_y = IMG_H / model_h

    # Collect raw detections (in model coordinate space)
    boxes = []  # (cls, x1, y1, x2, y2, score) in model pixel coords

    for i in range(out.shape[1]):
        scores = out[4:, i]
        max_cls = int(np.argmax(scores))
        max_score = float(scores[max_cls])

        if max_cls not in HERO_CLASSES or max_score < 0.1:
            continue

        cx, cy = float(out[0, i]), float(out[1, i])
        bw, bh = float(out[2, i]), float(out[3, i])

        # Filter degenerate boxes
        if bw < 3 or bh < 3:
            continue
        if not (0 <= cx < model_w and 0 <= cy < model_h):
            continue

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes.append((max_cls, x1, y1, x2, y2, max_score))

    if not boxes:
        return []

    # Apply NMS per class
    kept = []
    for cls_id in [0]:
        cls_boxes = [b for b in boxes if b[0] == cls_id]
        cls_boxes.sort(key=lambda b: b[5], reverse=True)

        while cls_boxes:
            best = cls_boxes.pop(0)
            # Convert to minimap pixel coords
            cx_mm = int(((best[1] + best[3]) / 2) * scale_x)
            cy_mm = int(((best[2] + best[4]) / 2) * scale_y)
            # Filter: skip deteksi di pinggir (false positive di luar minimap)
            if cx_mm < 10 or cx_mm >= IMG_W - 10 or cy_mm < 10 or cy_mm >= IMG_H - 10:
                continue
            kept.append((cls_id, cx_mm, cy_mm, best[5]))

            # Remove overlapping
            _, bx1, by1, bx2, by2, _ = best
            remaining = []
            for b in cls_boxes:
                _, x1, y1, x2, y2, _ = b
                # IoU check
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    union = (bx2 - bx1) * (by2 - by1) + (x2 - x1) * (y2 - y1) - intersection
                    if union > 0 and intersection / union > 0.15:
                        continue
                remaining.append(b)
            cls_boxes = remaining

    return kept


def run_inference_pt(model, mm):
    """Ultralytics inference on minimap crop."""
    results_pt = model(mm, verbose=False, conf=0.01, iou=0.5)
    results = []

    for r in results_pt:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in HERO_CLASSES:
                continue
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            results.append((cls_id, cx, cy, conf))

    return results


def draw_preview(mm, results):
    """Buat visual preview deteksi."""
    vis = cv2.resize(mm.copy(), (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
    colors = {0: (255, 180, 30), 1: (30, 30, 255)}
    for cls, cx, cy, conf in results:
        color = colors[cls]
        sx, sy = int(cx * 2), int(cy * 2)
        cv2.circle(vis, (sx, sy), 17, color, 3)
        cv2.circle(vis, (sx, sy), 3, (255, 255, 255), -1)
        cv2.putText(vis, f"{CLASS_NAMES[cls]} {conf:.2f}", (sx - 35, sy - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    return vis


def save_labels(img_name: str, boxes: list, split: str = "train"):
    """Simpan YOLO format labels. boxes = list of (cls, cx, cy) atau (cls, cx, cy, conf)."""
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
    lines = []
    for item in boxes:
        cls, cx, cy = item[0], item[1], item[2]
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = 37.0 / IMG_W
        nh = 37.0 / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def get_existing_pseudo_labels(video_stem):
    """Cari auto-label yang sudah ada untuk video tertentu."""
    existing = set()
    pattern = f"{AUTO_PREFIX}{video_stem}_frame_"
    label_dir = DATASET_DIR / "labels" / split
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
        description="Pseudo-label minimap hero dots using YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/pseudo_label.py --video alpha_1.mp4\n"
            "  python tools/pseudo_label.py --all-videos --conf 0.9\n"
            "  python tools/pseudo_label.py --video alpha_1.mp4 --preview\n"
            "\nIterative training loop:\n"
            "  1. python tools/pseudo_label.py --all-videos\n"
            "  2. python tools/train_yolo.py\n"
            "  3. Ulangi (model baru lebih akurat)\n"
        ),
    )
    ap.add_argument("--video", "-v", type=str, default=None, help="Video file name in videos/")
    ap.add_argument("--all-videos", action="store_true", help="Process all .mp4 videos")
    ap.add_argument("--dataset", action="store_true", help="Proses gambar yang sudah ada di dataset (bukan video)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"Confidence threshold (default: {DEFAULT_CONF})")
    ap.add_argument("--max-frames", type=int, default=0, help="Max frames to save per video")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help=f"Process every N frames (default: {FRAME_SKIP})")
    ap.add_argument("--preview", "-p", action="store_true", help="Show preview window")
    ap.add_argument("--dry-run", action="store_true", help="Don't save files")
    ap.add_argument("--no-skip-existing", action="store_true", help="Process all frames, even if auto-label exists")
    ap.add_argument("--val", action="store_true", help="Simpan ke val split (default: train)")
    ap.add_argument("--classes", type=str, default="heroes", choices=["heroes", "all"],
                    help="'heroes' = blue_hero+red_hero only, 'all' = all 11 classes (default: heroes)")
    args = ap.parse_args()

    # Setup
    split = "val" if args.val else "train"
    os.makedirs(DATASET_DIR / "images" / split, exist_ok=True)
    os.makedirs(DATASET_DIR / "labels" / split, exist_ok=True)

    # ── Load model dulu (sebelum interaksi, biar model_info ready) ──
    print("\n📦 Loading YOLO model...")
    model_info = load_yolo()
    if model_info is None:
        print(f"❌ No model found. Train one first: python tools/train_yolo.py")
        print(f"   Looked for: {MODEL_ONNX} or {MODEL_PT}")
        return
    backend = model_info[0]
    print(f"   ✅ Model ready ({backend.upper()})")

    # Mode dataset: proses gambar existing
    if args.dataset:
        img_dir = DATASET_DIR / "images" / split
        image_files = sorted(img_dir.glob("*.png"))
        if not image_files:
            print(f"❌ No images found in {img_dir}")
            return
        print(f"\n📸 Memproses {len(image_files)} gambar dari dataset ({split})...")
        process_images(args, model_info, backend, image_files, split)
        return

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
        # ── Interactive mode ──
        # Step 1: Pilih split dulu
        inp = input("\nPilih split [t]rain / [v]al / [s]emua [t]: ").strip().lower()
        if inp == "v":
            split = "val"
        elif inp == "s":
            split = "all"

        # Step 2: Pilih sumber data
        cs = sorted(VIDEO_DIR.glob("*.mp4"))
        if split != "all":
            ds_count = len(list((DATASET_DIR / "images" / split).glob("*.png")))
        else:
            ds_count = len(list(DATASET_DIR.glob("images/*/*.png")))
        print(f"\n🎬 Pilih sumber data ({split}):")
        print(f"  [d] Dataset ({ds_count} existing images)")
        print(f"  [v] Video ({len(cs)} file)")
        src = input("\nPilih [v/d]: ").strip().lower()

        if src == "d":
            if split == "all":
                for s in ["train", "val"]:
                    img_dir = DATASET_DIR / "images" / s
                    files = sorted(img_dir.glob("*.png"))
                    if files:
                        print(f"\n📸 Memproses {len(files)} gambar dari dataset ({s})...")
                        process_images(args, model_info, backend, files, s)
                return
            else:
                img_dir = DATASET_DIR / "images" / split
                image_files = sorted(img_dir.glob("*.png"))
                if not image_files:
                    print(f"❌ No images in {img_dir}")
                    return
                print(f"\n📸 Memproses {len(image_files)} gambar dari dataset ({split})...")
                process_images(args, model_info, backend, image_files, split)
                return

        # Step 3: Pilih video
        if split == "all":
            split = "train"
        print(f"\n🎬 Pilih video ({split}):")
        for idx_v, vid in enumerate(cs, 1):
            size_mb = vid.stat().st_size / (1024 * 1024)
            print(f"  [{idx_v}] {vid.name} ({size_mb:.1f} MB)")
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


    # Print header
    print("=" * 60)
    print("🤖 Pseudo-Label Minimap — YOLO-based Self-Training")
    print("=" * 60)
    print(f"  Model:       {MODEL_ONNX if MODEL_ONNX.exists() else MODEL_PT}")
    print(f"  Confidence:  ≥{args.conf}")
    print(f"  Classes:     {'blue_hero + red_hero' if args.classes == 'heroes' else 'all 11 classes'}")
    print(f"  Output:      {DATASET_DIR}/ ({split} split)")
    print(f"  Prefix:      {AUTO_PREFIX}")
    print(f"  Preview:     {'ON' if args.preview else 'OFF'}")
    print(f"  Dry-run:     {'YES' if args.dry_run else 'NO'}")
    print(f"  Frame skip:  every {args.frame_skip} frame(s)")
    print(f"  Videos:      {len(video_files)} file(s)")
    print("-" * 60)


    if args.classes == "heroes":
        global HERO_CLASSES
        HERO_CLASSES = {0}  # Model baru: hero = class 0

    # Process each video
    total_saved = 0
    total_dots = 0


def process_images(args, model_info, backend, image_files, split):
    """Proses existing dataset images (bukan video)."""
    import cv2
    import numpy as np
    from pathlib import Path
    from collections import Counter

    DATASET_DIR = Path("trainings/hero_detector")
    IMG_W, IMG_H = 350, 340
    AUTO_PREFIX = "auto_"

    total_saved = 0
    total_dots = 0
    total_added = 0
    per_class = Counter()

    for idx, img_path in enumerate(image_files):
        # Baca existing dari file ORIGINAL (bukan auto_ prefix)
        orig_lbl = DATASET_DIR / "labels" / split / f"{img_path.stem}.txt"

        existing = []
        cleaned = 0
        if orig_lbl.exists():
            for line in orig_lbl.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    nx, ny = float(parts[1]), float(parts[2])
                    cx_px, cy_px = int(nx * IMG_W), int(ny * IMG_H)
                    if cx_px < 10 or cx_px >= IMG_W - 10 or cy_px < 10 or cy_px >= IMG_H - 10:
                        cleaned += 1
                    else:
                        existing.append((int(parts[0]), cx_px, cy_px))
        if cleaned:
            print(f"   \U0001f9f9 Cleaned {cleaned} edge labels from {img_path.name}")

        mm = cv2.imread(str(img_path))
        if mm is None or mm.shape[:2] != (IMG_H, IMG_W):
            continue

        # YOLO inference
        if backend == "onnx":
            session, input_name, model_w, model_h = model_info[1], model_info[2], model_info[3], model_info[4]
            results = run_inference_onnx(session, input_name, mm, model_w, model_h)
        else:
            results = run_inference_pt(model_info[1], mm)

        results = [(cls, cx, cy, conf) for cls, cx, cy, conf in results if conf >= args.conf]

        # Merge: simpan existing + semua deteksi baru (tanpa filter duplikat)
        merged = list(existing)
        added = len(results)
        for cls, cx, cy, conf in results:
            merged.append((cls, cx, cy))

        if not merged:
            continue

        if not args.dry_run:
            # Simpan label langsung dengan nama original (bukan auto_ prefix)
            orig_name = f"{img_path.stem}.png"
            save_labels(orig_name, merged, split)

        total_saved += 1
        total_dots += len(merged)
        total_added += added
        for cls, _, _ in merged:
            per_class[cls] += 1

        if (idx + 1) % 50 == 0:
            print(f"   [{idx+1}/{len(image_files)}] {total_saved} saved, {total_dots} dots")

    names = {0: "blue_hero", 1: "red_hero"}
    print(f"\n✅ Dataset done: {total_saved} labeled, {total_dots} total dots (+{total_added} new)")
    for cls in sorted(per_class.keys()):
        print(f"   {names.get(cls, '?')}: {per_class[cls]}")


    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  Total frames:  {total_saved}")
    print(f"  Total dots:    {total_dots}")
    if total_saved:
        print(f"  Avg/frame:     {total_dots / total_saved:.1f}")
    if not args.dry_run:
        pseudo_count = len(list((DATASET_DIR / "images" / split).glob(f"{AUTO_PREFIX}*.png")))
        total_imgs = len(list((DATASET_DIR / "images" / split).glob("*.png")))
        print(f"  Dataset: {total_imgs} images ({total_imgs - pseudo_count} manual, {pseudo_count} pseudo)")
        print(f"\n  📌 Next steps:")
        print(f"     python tools/train_yolo.py            # retrain dengan auto-labels")
        print(f"     python tools/pseudo_label.py --all-videos --conf 0.9   # auto-label lebih banyak")
        print(f"     # ulangi: train → auto-label → train lagi")
    print("=" * 60)


if __name__ == "__main__":
    main()

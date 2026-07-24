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
DEFAULT_CONF = 0.85

# Frame skip (default: ~1 fps)
FRAME_SKIP = 30

# File prefix untuk auto-labeled
AUTO_PREFIX = "auto_"

# Class filter: hanya hero (0=blue_hero, 1=red_hero)
HERO_CLASSES = {0, 1}

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
    for cls_id in [0, 1]:
        cls_boxes = [b for b in boxes if b[0] == cls_id]
        cls_boxes.sort(key=lambda b: b[5], reverse=True)

        while cls_boxes:
            best = cls_boxes.pop(0)
            # Convert to minimap pixel coords
            cx_mm = int(((best[1] + best[3]) / 2) * scale_x)
            cy_mm = int(((best[2] + best[4]) / 2) * scale_y)
            cx_mm = max(0, min(IMG_W - 1, cx_mm))
            cy_mm = max(0, min(IMG_H - 1, cy_mm))
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
                    if union > 0 and intersection / union > 0.3:
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
    """Simpan YOLO format labels."""
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
    lines = []
    for cls, cx, cy, conf in boxes:
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = 34.0 / IMG_W
        nh = 34.0 / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def get_existing_pseudo_labels(video_stem):
    """Cari auto-label yang sudah ada untuk video tertentu."""
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
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"Confidence threshold (default: {DEFAULT_CONF})")
    ap.add_argument("--max-frames", type=int, default=0, help="Max frames to save per video")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help=f"Process every N frames (default: {FRAME_SKIP})")
    ap.add_argument("--preview", "-p", action="store_true", help="Show preview window")
    ap.add_argument("--dry-run", action="store_true", help="Don't save files")
    ap.add_argument("--no-skip-existing", action="store_true", help="Process all frames, even if auto-label exists")
    ap.add_argument("--classes", type=str, default="heroes", choices=["heroes", "all"],
                    help="'heroes' = blue_hero+red_hero only, 'all' = all 11 classes (default: heroes)")
    args = ap.parse_args()

    # Setup
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

    # Print header
    print("=" * 60)
    print("🤖 Pseudo-Label Minimap — YOLO-based Self-Training")
    print("=" * 60)
    print(f"  Model:       {MODEL_ONNX if MODEL_ONNX.exists() else MODEL_PT}")
    print(f"  Confidence:  ≥{args.conf}")
    print(f"  Classes:     {'blue_hero + red_hero' if args.classes == 'heroes' else 'all 11 classes'}")
    print(f"  Output:      {DATASET_DIR}/ (train split)")
    print(f"  Prefix:      {AUTO_PREFIX}")
    print(f"  Preview:     {'ON' if args.preview else 'OFF'}")
    print(f"  Dry-run:     {'YES' if args.dry_run else 'NO'}")
    print(f"  Frame skip:  every {args.frame_skip} frame(s)")
    print(f"  Videos:      {len(video_files)} file(s)")
    print("-" * 60)

    # Load model
    print("\n📦 Loading YOLO model...")
    model_info = load_yolo()
    if model_info is None:
        print(f"❌ No model found. Train one first: python tools/train_yolo.py")
        print(f"   Looked for: {MODEL_ONNX} or {MODEL_PT}")
        return

    backend = model_info[0]
    print(f"   ✅ Model ready ({backend.upper()})")

    if args.classes == "heroes":
        global HERO_CLASSES
        HERO_CLASSES = {0, 1}

    # Process each video
    total_saved = 0
    total_dots = 0

    for vid_idx, vid in enumerate(video_files):
        print(f"\n📹 [{vid_idx + 1}/{len(video_files)}] Processing: {vid.name}")

        # Cek yang sudah ada
        existing_labels = get_existing_pseudo_labels(vid.stem) if not args.no_skip_existing else set()
        if existing_labels:
            print(f"   📋 Existing auto-labels: {len(existing_labels)} frames")

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"   ❌ Cannot open {vid.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = args.frame_skip
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
            cv2.namedWindow("Pseudo Label", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pseudo Label", IMG_W * 2 + 20, IMG_H * 2 + 60)

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in existing_labels:
                skipped += 1
                frame_idx += 1
                continue
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            mm = crop_region(frame, "map")
            if mm is None or mm.shape[:2] != (IMG_H, IMG_W):
                frame_idx += 1
                continue

            # YOLO inference
            if backend == "onnx":
                session, input_name, model_w, model_h = model_info[1], model_info[2], model_info[3], model_info[4]
                results = run_inference_onnx(session, input_name, mm, model_w, model_h)
            else:
                results = run_inference_pt(model_info[1], mm)

            # Filter by confidence
            results = [(cls, cx, cy, conf) for cls, cx, cy, conf in results if conf >= args.conf]

            if not results:
                empty += 1
                if args.preview:
                    vis = draw_preview(mm, results)
                    cv2.putText(vis, f"[PS] Frame {frame_idx} — No detections ≥{args.conf}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.imshow("Pseudo Label", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            img_name = f"{AUTO_PREFIX}{vid.stem}_frame_{frame_idx}.png"

            if not args.dry_run:
                cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                save_labels(img_name, results)

            saved += 1
            frame_dots += len(results)
            total_dots += len(results)

            if args.preview:
                vis = draw_preview(mm, results)
                counts = Counter(c for c, _, _, _ in results)
                info = " | ".join(f"{CLASS_NAMES[c]}={n} (min conf < {min(conf for _,_,_,conf in results):.2f})" for c, n in sorted(counts.items()))
                cv2.putText(vis, f"[PS] Frame {frame_idx} — {info}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Pseudo Label", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if saved >= max_frames:
                break

        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

        print(f"   ✅ {saved} saved, {empty} empty, {skipped} skipped, {frame_dots} dots")
        total_saved += saved

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  Total frames:  {total_saved}")
    print(f"  Total dots:    {total_dots}")
    if total_saved:
        print(f"  Avg/frame:     {total_dots / total_saved:.1f}")
    if not args.dry_run:
        pseudo_count = len(list((DATASET_DIR / "images" / "train").glob(f"{AUTO_PREFIX}*.png")))
        total_imgs = len(list((DATASET_DIR / "images" / "train").glob("*.png")))
        print(f"  Dataset: {total_imgs} images ({total_imgs - pseudo_count} manual, {pseudo_count} pseudo)")
        print(f"\n  📌 Next steps:")
        print(f"     python tools/train_yolo.py            # retrain dengan auto-labels")
        print(f"     python tools/pseudo_label.py --all-videos --conf 0.9   # auto-label lebih banyak")
        print(f"     # ulangi: train → auto-label → train lagi")
    print("=" * 60)


if __name__ == "__main__":
    main()

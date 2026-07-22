#!/usr/bin/env python3
"""
Inference — MLBB Minimap Hero Detection (YOLOv11n)
Menampilkan deteksi hero dot + skor model di minimap.
Usage:
  python tools/inference.py
  python tools/inference.py --video alpha_1.mp4
  python tools/inference.py --model path/to/best.pt --conf 0.3
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.core import layout
from vision.core.cropper import crop_region
from vision.detectors.minimap.yolo_detector import YOLOMinimapDetector

# Colors (BGR)
BLUE = (255, 180, 30)
RED = (30, 30, 255)
GREEN = (50, 200, 50)
YELLOW = (50, 200, 200)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)


def load_model_score(project_dir: str = "runs/detect/trainings/hero_detector/yolo11n_minimap") -> float:
    """
    Baca mAP50 dari hasil training terakhir.
    Returns: mAP50 (0-1) atau 0 jika tidak ditemukan.
    """
    csv_path = Path(project_dir) / "results.csv"
    if not csv_path.exists():
        return 0.0
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 0.0
            # Ambil mAP50 dari epoch terbaik (kolom terakhir)
            last = rows[-1]
            # Cari kolom metrics/mAP50(B)
            for key in last:
                if "mAP50" in key and "B" in key:
                    return float(last[key])
            return 0.0
    except Exception:
        return 0.0


def get_model_quality(map50: float) -> tuple[str, tuple]:
    """
    Kualitas model berdasarkan mAP50.
    Returns: (label, color_bgr)
    """
    if map50 >= 0.80:
        return "EXCELLENT", GREEN
    elif map50 >= 0.70:
        return "GOOD", GREEN
    elif map50 >= 0.60:
        return "DECENT", YELLOW
    elif map50 >= 0.50:
        return "LOW", YELLOW
    else:
        return "POOR (need more data)", RED


def draw_minimap_detections(frame, minimap_img, detections, mm_bbox):
    """Draw YOLO detection boxes + labels."""
    mm_x, mm_y, mm_w, mm_h = mm_bbox
    vis = minimap_img.copy()

    for team, cx, cy, r, conf in detections:
        color = BLUE if team == "blue" else RED
        x1 = int(cx - r)
        y1 = int(cy - r)
        x2 = int(cx + r)
        y2 = int(cy + r)

        # Box + center (clean lines & center dot)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), 2, color, -1)

        # Label (larger & clear font)
        label_text = f"{'BLUE' if team=='blue' else 'RED'} {conf:.0%}"
        font_scale = 0.45
        thick = 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

        # Label position & background box
        ly1 = max(th + 4, y1)
        cv2.rectangle(vis, (x1, ly1 - th - 4), (x1 + tw + 6, ly1 + 2), BLACK, -1)
        cv2.rectangle(vis, (x1, ly1 - th - 4), (x1 + tw + 6, ly1 + 2), color, 1)
        cv2.putText(vis, label_text, (x1 + 3, ly1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thick, cv2.LINE_AA)

    frame[mm_y:mm_y+mm_h, mm_x:mm_x+mm_w] = vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", "-v", type=str, default=None)
    ap.add_argument("--model", "-m", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    # ── Model score ──
    model_path = args.model or "trainings/hero_detector/yolo11n_minimap/weights/best.pt"
    map50 = load_model_score()
    quality_label, quality_color = get_model_quality(map50)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  YOLOv11n Minimap Detection")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Model:  {model_path}")
    print(f"  mAP50:  {map50:.3f}  ({quality_label})")
    print(f"  Conf:   {args.conf}")
    if map50 < 0.5:
        print(f"  ⚠️  mAP50 < 0.5 — perlu lebih banyak labeling")
    elif map50 < 0.7:
        print(f"  📝 mAP50 0.5-0.7 — usable untuk tracking dasar")
    elif map50 < 0.85:
        print(f"  👍 mAP50 0.7-0.85 — bagus untuk detection")
    else:
        print(f"  🎯 mAP50 > 0.85 — sangat akurat")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Init YOLO ──
    detector = YOLOMinimapDetector(model_path=model_path, conf_threshold=args.conf)

    # ── Video ──
    vid_dir = Path("videos")
    if args.video:
        video_path = vid_dir / args.video
    else:
        videos = sorted(vid_dir.glob("*.mp4"))
        if not videos:
            print("❌ No videos found in videos/")
            return
        video_path = videos[0]

    print(f"📹 Video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps / args.speed))
    mm_bbox = layout.bbox("map") or (80, 0, 350, 340)

    # Skip intro
    for _ in range(int(fps * 5)):
        cap.read()

    print("▶️  Running... (Q to quit)")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mm = crop_region(frame, "map")
        if mm is None or mm.size == 0:
            continue

        # YOLO inference
        t0 = time.perf_counter()
        detections = detector.detect(mm)
        inf_ms = (time.perf_counter() - t0) * 1000
        frame_count += 1

        # Draw minimap
        draw_minimap_detections(frame, mm, detections, mm_bbox)

        # ── Overlay info ──
        lines = [
            f"YOLO: {inf_ms:.1f}ms | {len(detections)} heroes",
            f"mAP50: {map50:.3f}  [{quality_label}]",
            f"Conf threshold: {args.conf}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, quality_color if i == 1 else WHITE, 2)

        # Show
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (w // 2, h // 2))
        cv2.imshow("YOLO Minimap Detection", display)

        k = cv2.waitKey(delay) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. {frame_count} frames processed")


if __name__ == "__main__":
    main()

"""
Labeling Tool — MLBB Minimap Hero Detection
Klik kiri = blue hero, klik kanan = red hero.
Tekan N = next frame tanpa save, S = save & next, Q = quit.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.core import layout
from vision.core.cropper import crop_region

DATASET_DIR = Path("trainings/hero_detector")
VIDEO_DIR = Path("videos")
IMG_W, IMG_H = 350, 340
HERO_DOT_SIZE = 34  # Ukuran lingkaran dot hero lebih pas/presisi (34px)



def main():
    import argparse
    ap = argparse.ArgumentParser(description="Label hero dots on minimap")
    ap.add_argument("--video", "-v", type=str, default=None,
                    help="Video file name in videos/ (e.g. alpha_1.mp4)")
    args = ap.parse_args()

    os.makedirs(DATASET_DIR / "images" / "train", exist_ok=True)
    os.makedirs(DATASET_DIR / "labels" / "train", exist_ok=True)

    if args.video:
        video_path = VIDEO_DIR / args.video
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        video_files = [video_path]
    else:
        video_files = sorted(VIDEO_DIR.glob("*.mp4"))
        if not video_files:
            print(f"❌ No .mp4 videos found in {VIDEO_DIR.absolute()}")
            print(f"   Place video files in videos/ directory")
            return

    print("=== 🏷️  MLBB Minimap Hero Labeling Tool ===")
    print("  Left click  = blue_hero")
    print("  Right click = red_hero")
    print("  N = next frame (auto-save) | S = save + next | Q = quit")
    print(f"  Videos: {[v.name for v in video_files]}")
    print(f"  Output: {DATASET_DIR.absolute()}/")
    print("━" * 50)

    for vid in video_files:
        print(f"\n📹 Processing: {vid.name}")
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"  ❌ Cannot open {vid.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

        frame_idx = 0
        saved_count = 0

        # Skip intro (first ~5 seconds)
        skip_frames = int(fps * 5) if fps > 0 else 150
        while frame_idx < skip_frames:
            ret = cap.read()
            if not ret:
                break
            frame_idx += 1

        boxes: list = []
        img_name = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"  ✅ End of video ({saved_count} frames labeled)")
                break

            mm = crop_region(frame, "map")

            # Show this frame, then jump 5 seconds on 'n'/'s'
            pass

            if mm is None or mm.shape != (IMG_H, IMG_W, 3):
                print(f"  ⚠️  Frame {frame_idx}: minimap crop failed ({mm.shape if mm is not None else 'None'})")
                frame_idx += 1
                continue

            img_name = f"{vid.stem}_frame_{frame_idx}.png"
            display = mm.copy()
            display = cv2.resize(display, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
            boxes = []
            frame_idx += 1

            # ── Draw overlay ──
            def draw_overlay(vis, bx, idx, name, _label_mode):
                for cls, cx, cy, w, h in bx:
                    center_x = int(cx * 2)
                    center_y = int(cy * 2)
                    radius = int((w / 2) * 2)  # radius dalam skala visual (2x)
                    # BGR colors: Blue=(255,180,30), Red=(30,30,255)
                    if cls == 0:
                        box_color = (255, 180, 30)  # biru (B=255)
                        text = "BLUE"
                    else:
                        box_color = (30, 30, 255)   # merah (R=255)
                        text = "RED"
                    # Draw circle overlay instead of rectangle
                    cv2.circle(vis, (center_x, center_y), radius, box_color, 3)
                    # Draw center point dot
                    cv2.circle(vis, (center_x, center_y), 3, (255, 255, 255), -1)
                    # Label bg
                    cv2.rectangle(vis, (center_x - 25, center_y - radius - 20), (center_x + 25, center_y - radius), (0, 0, 0), -1)
                    cv2.putText(vis, text, (center_x - 20, center_y - radius - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 3)
                cv2.putText(vis, f"Frame {idx} - {name}  [{len(bx)} dots]",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                mode_text = "BLUE" if _label_mode[0] == 0 else "RED"
                mode_color = (255, 180, 30) if _label_mode[0] == 0 else (30, 30, 255)
                cv2.putText(vis, f"[B]lue [R]ed  Mode: {mode_text}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
                cv2.putText(vis, "[N]ext [S]ave+Next [Q]uit",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                return vis

            _label_mode = [0]  # [0]=BLUE, [1]=RED

            def mouse_cb(event, x, y, flags, param):
                nonlocal boxes
                if event == cv2.EVENT_LBUTTONDOWN:
                    cls = _label_mode[0]
                    label = "BLUE" if cls == 0 else "RED"
                    boxes.append((cls, x / 2, y / 2, HERO_DOT_SIZE, HERO_DOT_SIZE))
                    print(f"    {label:4s} at ({x//2:3d}, {y//2:3d})")

            cv2.namedWindow("Label Minimap Heroes", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Label Minimap Heroes", IMG_W * 2, IMG_H * 2)
            cv2.setMouseCallback("Label Minimap Heroes", mouse_cb)

            while True:
                vis = draw_overlay(display.copy(), boxes, frame_idx, vid.name, _label_mode)
                cv2.imshow("Label Minimap Heroes", vis)
                k = cv2.waitKey(1) & 0xFF

                if k == ord('b'):
                    _label_mode[0] = 0
                    print("  Mode: BLUE")
                elif k == ord('r'):
                    _label_mode[0] = 1
                    print("  Mode: RED")
                elif k == ord('n'):
                    if boxes:
                        cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                        save_labels(img_name, boxes)
                        saved_count += 1
                        print(f"  ✅ Saved {img_name} ({len(boxes)} labels)")
                    else:
                        print(f"  ⏭️  Skipped frame {frame_idx} (no labels)")
                    frame_idx += 30  # 1 detik (30fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    break
                elif k == ord('s'):
                    if boxes:
                        cv2.imwrite(str(DATASET_DIR / "images" / "train" / img_name), mm)
                        save_labels(img_name, boxes)
                        saved_count += 1
                        print(f"  ✅ Saved {img_name} ({len(boxes)} labels)")
                    else:
                        print(f"  ⏭️  Skipped frame {frame_idx} (tanpa label)")
                    break

                elif k == ord('q'):
                    cv2.destroyAllWindows()
                    print(f"\nDone! {saved_count} frames labeled.")
                    print(f"Dataset: {DATASET_DIR.absolute()}/")
                    return

        cap.release()

    cv2.destroyAllWindows()
    print(f"\n🎉 Done! Dataset in {DATASET_DIR.absolute()}/")
    print("   Run: python tools/train_yolo.py")


def save_labels(img_name: str, boxes: list):
    """Save YOLO format labels (normalized cx, cy, w, h). Only saves if boxes is not empty."""
    label_path = DATASET_DIR / "labels" / "train" / f"{Path(img_name).stem}.txt"
    if not boxes:
        if label_path.exists():
            label_path.unlink()
        return False

    lines = []
    for cls, cx, cy, w, h in boxes:
        nx = cx / IMG_W
        ny = cy / IMG_H
        nw = w / IMG_W
        nh = h / IMG_H
        lines.append(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + "\n")
    return True


if __name__ == "__main__":
    main()

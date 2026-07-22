"""
Labeling Tool — MLBB Minimap Hero Detection
Klik kiri = label dengan mode aktif, klik kanan = red hero (shortcut).
Tekan B = blue_hero, R = red_hero, J = jungle.
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

# Class definitions
CLASS_NAMES = {0: "BLUE", 1: "RED", 2: "JUNGLE"}
CLASS_COLORS = {
    0: (255, 180, 30),   # biru (BGR)
    1: (30, 30, 255),    # merah (BGR)
    2: (50, 220, 50),    # hijau (BGR)
}
CLASS_KEYS = {
    ord('b'): 0,  # blue_hero
    ord('r'): 1,  # red_hero
    ord('j'): 2,  # jungle
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Label hero dots on minimap")
    ap.add_argument("--video", "-v", type=str, default=None,
                    help="Video file name in videos/ (e.g. alpha_1.mp4)")
    ap.add_argument("--split", "-s", type=str, choices=["train", "val", "sample"], default=None,
                    help="Target dataset split: 'train' or 'val'/'sample'")
    args = ap.parse_args()

    if args.split:
        target_split = "val" if args.split in ["val", "sample"] else "train"
    else:
        print("\n📌 Pilih folder tujuan penyimpanan dataset:")
        print("  [1] Train  (trainings/hero_detector/images/train)")
        print("  [2] Val / Sample (trainings/hero_detector/images/val)")
        choice = input("Pilihan [1/2] (default: 1): ").strip()
        target_split = "val" if choice in ["2", "val", "sample"] else "train"

    split_label = "VAL (SAMPLE)" if target_split == "val" else "TRAIN"

    os.makedirs(DATASET_DIR / "images" / target_split, exist_ok=True)
    os.makedirs(DATASET_DIR / "labels" / target_split, exist_ok=True)

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

    print("=== 🏷️  MLBB Minimap Labeling Tool ===")
    print(f"  Target Split: 🎯 {split_label}")
    print("  Klik Kiri   = label mode aktif")
    print("  Klik Kanan  = red_hero (shortcut)")
    print("  B = blue_hero | R = red_hero | J = jungle")
    print("  N = next frame (+1s) | S = save + next | U = undo | Q = quit")
    print(f"  Videos: {[v.name for v in video_files]}")
    print(f"  Output: {DATASET_DIR.absolute()}/[images|labels]/{target_split}/")
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

            if mm is None or mm.shape != (IMG_H, IMG_W, 3):
                print(f"  ⚠️  Frame {frame_idx}: minimap crop failed ({mm.shape if mm is not None else 'None'})")
                frame_idx += 1
                continue

            img_name = f"{vid.stem}_frame_{frame_idx}.png"
            display = mm.copy()
            display = cv2.resize(display, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
            boxes = []
            frame_idx += 1

            BOTTOM_BAR_H = 60

            # ── Draw overlay ──
            def draw_overlay(vis_map, bx, idx, total_f, fps_val, name, _label_mode):
                canvas = np.zeros((IMG_H * 2 + BOTTOM_BAR_H, IMG_W * 2, 3), dtype=np.uint8)
                canvas[IMG_H * 2:, :] = (30, 30, 35)

                for cls, cx, cy, w, h in bx:
                    center_x = int(cx * 2)
                    center_y = int(cy * 2)
                    radius = int((w / 2) * 2)  # radius dalam skala visual (2x)
                    box_color = CLASS_COLORS.get(cls, (200, 200, 200))
                    text = CLASS_NAMES.get(cls, f"CLS_{cls}")
                    cv2.circle(vis_map, (center_x, center_y), radius, box_color, 3)
                    cv2.circle(vis_map, (center_x, center_y), 3, (255, 255, 255), -1)
                    cv2.rectangle(vis_map, (center_x - 35, center_y - radius - 20), (center_x + 35, center_y - radius), (0, 0, 0), -1)
                    cv2.putText(vis_map, text, (center_x - 30, center_y - radius - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 3)

                canvas[:IMG_H * 2, :] = vis_map

                # Top overlay info
                cv2.putText(canvas, f"[{split_label}] Frame {idx} - {name}  [{len(bx)} dots]",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if target_split == "val" else (255, 255, 255), 2)
                cv2.putText(canvas, "[N]ext (+1s) [S]ave+Next [U]ndo [Q]uit",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Mode display
                mode_cls = _label_mode[0]
                mode_text = CLASS_NAMES.get(mode_cls, f"CLS_{mode_cls}")
                mode_color = CLASS_COLORS.get(mode_cls, (200, 200, 200))
                cv2.putText(canvas, f"Mode: {mode_text} | [B]lue [R]ed [J]ungle | RClick=RED", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

                # Separator line
                cv2.line(canvas, (0, IMG_H * 2), (IMG_W * 2, IMG_H * 2), (80, 80, 80), 2)

                # Timestamp calculation
                curr_sec = (idx - 1) / fps_val if fps_val > 0 else 0
                tot_sec = total_f / fps_val if fps_val > 0 else 0
                cur_m, cur_s = int(curr_sec // 60), int(curr_sec % 60)
                tot_m, tot_s = int(tot_sec // 60), int(tot_sec % 60)

                timestamp_str = f"Timestamp: {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} ({curr_sec:.1f}s)"
                frame_str = f"Frame: {idx} / {total_f}"

                # Timestamp display below map
                cv2.putText(canvas, timestamp_str, (15, IMG_H * 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(canvas, frame_str, (15, IMG_H * 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                return canvas

            _label_mode = [0]  # [0]=BLUE, [1]=RED, [2]=JUNGLE

            def mouse_cb(event, x, y, flags, param):
                nonlocal boxes
                # Click must be inside minimap bounds
                if y >= IMG_H * 2 or x >= IMG_W * 2:
                    return

                if event == cv2.EVENT_LBUTTONDOWN:
                    cls = _label_mode[0]
                    label = CLASS_NAMES.get(cls, f"CLS_{cls}")
                    boxes.append((cls, x / 2, y / 2, HERO_DOT_SIZE, HERO_DOT_SIZE))
                    print(f"    {label:7s} at ({x//2:3d}, {y//2:3d})")
                elif event == cv2.EVENT_RBUTTONDOWN:
                    cls = 1  # shortcut: right click = red_hero
                    boxes.append((cls, x / 2, y / 2, HERO_DOT_SIZE, HERO_DOT_SIZE))
                    print(f"    RED     at ({x//2:3d}, {y//2:3d})")

            cv2.namedWindow("Label Minimap", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Label Minimap", IMG_W * 2, IMG_H * 2 + BOTTOM_BAR_H)
            cv2.setMouseCallback("Label Minimap", mouse_cb)

            while True:
                vis = draw_overlay(display.copy(), boxes, frame_idx, total_frames, fps, vid.name, _label_mode)
                cv2.imshow("Label Minimap", vis)
                k = cv2.waitKey(1) & 0xFF

                # Class mode switching
                if k in CLASS_KEYS:
                    _label_mode[0] = CLASS_KEYS[k]
                    mode_name = CLASS_NAMES[_label_mode[0]]
                    print(f"  Mode: {mode_name}")
                elif k == ord('u'):
                    if boxes:
                        removed = boxes.pop()
                        label = CLASS_NAMES.get(removed[0], f"CLS_{removed[0]}")
                        print(f"  ↩️  Undo: removed {label} dot")
                elif k == ord('n'):
                    if boxes:
                        cv2.imwrite(str(DATASET_DIR / "images" / target_split / img_name), mm)
                        save_labels(img_name, boxes, target_split)
                        saved_count += 1
                        print(f"  ✅ Saved to [{target_split}] {img_name} ({len(boxes)} labels)")
                    else:
                        print(f"  ⏭️  Skipped frame {frame_idx} (no labels)")
                    frame_idx += 30  # 1 detik (30fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    break
                elif k == ord('s'):
                    if boxes:
                        cv2.imwrite(str(DATASET_DIR / "images" / target_split / img_name), mm)
                        save_labels(img_name, boxes, target_split)
                        saved_count += 1
                        print(f"  ✅ Saved to [{target_split}] {img_name} ({len(boxes)} labels)")
                    else:
                        print(f"  ⏭️  Skipped frame {frame_idx} (tanpa label)")
                    break

                elif k == ord('q'):
                    cv2.destroyAllWindows()
                    print(f"\nDone! {saved_count} frames labeled for [{target_split}].")
                    print(f"Dataset: {DATASET_DIR.absolute()}/")
                    return

        cap.release()

    cv2.destroyAllWindows()
    print(f"\n🎉 Done! Dataset in {DATASET_DIR.absolute()}/")
    print("   Run: python tools/train_yolo.py")


def save_labels(img_name: str, boxes: list, split: str = "train"):
    """Save YOLO format labels (normalized cx, cy, w, h). Only saves if boxes is not empty."""
    label_path = DATASET_DIR / "labels" / split / f"{Path(img_name).stem}.txt"
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

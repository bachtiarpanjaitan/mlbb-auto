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

# Class definitions for all 11 classes in data.yaml
CLASS_NAMES = {
    0: "blue_hero",
    1: "red_hero",
    2: "lord",
    3: "turtle",
    4: "thunder_fenrir",
    5: "molten_fiend",
    6: "lithowanderer",
    7: "crab",
    8: "lava_golem",
    9: "fire_beetle",
    10: "horned_lizard",
}

CLASS_LABELS_SHORT = {
    0: "0/B: blue_hero",
    1: "1/R: red_hero",
    2: "2/L: lord",
    3: "3/T: turtle",
    4: "4:   blue_buff (fenrir)",
    5: "5:   red_buff (fiend)",
    6: "6:   lithowanderer",
    7: "7:   crab (gold)",
    8: "8:   lava_golem",
    9: "9:   fire_beetle",
    10: "K/10:horned_lizard",
}

CLASS_COLORS = {
    0: (255, 180, 30),    # Blue (BGR)
    1: (30, 30, 255),     # Red
    2: (255, 0, 255),     # Lord (Magenta)
    3: (0, 255, 0),       # Turtle (Green)
    4: (255, 255, 0),     # Thunder Fenrir / Blue Buff (Cyan)
    5: (0, 140, 255),     # Molten Fiend / Red Buff (Orange)
    6: (200, 255, 0),     # Lithowanderer (Yellow-Green)
    7: (0, 215, 255),     # Crab (Gold)
    8: (180, 0, 180),     # Lava Golem (Purple)
    9: (0, 100, 255),     # Fire Beetle (Dark Orange)
    10: (255, 100, 200),  # Horned Lizard (Pink)
}

CLASS_KEYS = {
    ord('b'): 0,
    ord('r'): 1,
    ord('l'): 2,
    ord('t'): 3,
    ord('0'): 0,
    ord('1'): 1,
    ord('2'): 2,
    ord('3'): 3,
    ord('4'): 4,
    ord('5'): 5,
    ord('6'): 6,
    ord('7'): 7,
    ord('8'): 8,
    ord('9'): 9,
    ord('k'): 10,
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Label hero & jungle dots on minimap")
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

    print("=== 🏷️  MLBB Minimap Labeling Tool (11 Classes) ===")
    print(f"  Target Split: 🎯 {split_label}")
    print("  Klik Kiri   = label kelas aktif")
    print("  Tekan 0-9/K = pilih kelas (0=blue, 1=red, 2=lord, 3=turtle, 4=blue_buff, 5=red_buff, dst)")
    print("  Tekan TAB   = ganti kelas berikutnya")
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

        # Cari index terakhir untuk auto-increment
        img_dir = DATASET_DIR / "images" / target_split
        lbl_dir = DATASET_DIR / "labels" / target_split
        existing = [f.stem for f in img_dir.glob("*.png") if f.stem.isdigit()]
        last_idx = max((int(s) for s in existing), default=0)
        save_counter = last_idx + 1
        print(f"   📝 Index terakhir: {last_idx} → mulai dari {save_counter}")

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

            img_name = f"{save_counter}.png"
            display = mm.copy()
            display = cv2.resize(display, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
            boxes = []
            frame_idx += 1

            BOTTOM_BAR_H = 60
            RIGHT_PANEL_W = 320

            # ── Draw overlay ──
            def draw_overlay(vis_map, bx, idx, total_f, fps_val, name, _label_mode):
                map_w = IMG_W * 2
                map_h = IMG_H * 2
                canvas = np.zeros((map_h + BOTTOM_BAR_H, map_w + RIGHT_PANEL_W, 3), dtype=np.uint8)

                # Map area
                for cls, cx, cy, w, h in bx:
                    center_x = int(cx * 2)
                    center_y = int(cy * 2)
                    radius = int((w / 2) * 2)
                    box_color = CLASS_COLORS.get(cls, (200, 200, 200))
                    text = CLASS_NAMES.get(cls, f"CLS_{cls}")
                    cv2.circle(vis_map, (center_x, center_y), radius, box_color, 3)
                    cv2.circle(vis_map, (center_x, center_y), 3, (255, 255, 255), -1)
                    cv2.rectangle(vis_map, (center_x - 35, center_y - radius - 20), (center_x + 35, center_y - radius), (0, 0, 0), -1)
                    cv2.putText(vis_map, text, (center_x - 30, center_y - radius - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                canvas[:map_h, :map_w] = vis_map

                # Top overlay info on map
                cv2.putText(canvas, f"[{split_label}] Frame {idx} - {name} [{len(bx)} dots]",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if target_split == "val" else (255, 255, 255), 2)
                cv2.putText(canvas, "[N]ext (+1s) [S]ave+Next [U]ndo [Q]uit",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Bottom bar
                canvas[map_h:, :] = (30, 30, 35)
                cv2.line(canvas, (0, map_h), (map_w + RIGHT_PANEL_W, map_h), (80, 80, 80), 2)

                # Timestamp calculation
                curr_sec = (idx - 1) / fps_val if fps_val > 0 else 0
                tot_sec = total_f / fps_val if fps_val > 0 else 0
                cur_m, cur_s = int(curr_sec // 60), int(curr_sec % 60)
                tot_m, tot_s = int(tot_sec // 60), int(tot_sec % 60)

                timestamp_str = f"Timestamp: {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} ({curr_sec:.1f}s)"
                frame_str = f"Frame: {idx} / {total_f}"

                cv2.putText(canvas, timestamp_str, (15, map_h + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(canvas, frame_str, (15, map_h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                # Right Panel (Class Selection Legend)
                canvas[:map_h, map_w:] = (20, 20, 25)
                cv2.line(canvas, (map_w, 0), (map_w, map_h), (80, 80, 80), 2)

                cv2.putText(canvas, "SELECT CLASS [0-9, K, TAB]", (map_w + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.line(canvas, (map_w + 10, 32), (map_w + RIGHT_PANEL_W - 10, 32), (100, 100, 100), 1)

                active_cls = _label_mode[0]
                y_offset = 55
                for cls_id in range(11):
                    color = CLASS_COLORS[cls_id]
                    lbl = CLASS_LABELS_SHORT[cls_id]
                    count = sum(1 for b in bx if b[0] == cls_id)

                    # Highlight active class
                    if cls_id == active_cls:
                        cv2.rectangle(canvas, (map_w + 5, y_offset - 16), (map_w + RIGHT_PANEL_W - 5, y_offset + 6), (60, 60, 75), -1)
                        cv2.rectangle(canvas, (map_w + 5, y_offset - 16), (map_w + RIGHT_PANEL_W - 5, y_offset + 6), color, 2)
                        prefix = "👉 "
                    else:
                        prefix = "   "

                    # Color swatch box
                    cv2.rectangle(canvas, (map_w + 30, y_offset - 12), (map_w + 45, y_offset + 2), color, -1)

                    # Class text
                    cnt_str = f"({count})" if count > 0 else ""
                    cv2.putText(canvas, f"{prefix}{lbl} {cnt_str}", (map_w + 50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255) if cls_id == active_cls else (170, 170, 170), 1)

                    y_offset += 26

                cv2.putText(canvas, "Tip: Press TAB to cycle class", (map_w + 10, map_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                return canvas

            _label_mode = [0]  # default = 0 (blue_hero)

            def mouse_cb(event, x, y, flags, param):
                nonlocal boxes
                # Click must be inside minimap bounds
                if y >= IMG_H * 2 or x >= IMG_W * 2:
                    return

                if event == cv2.EVENT_LBUTTONDOWN:
                    cls = _label_mode[0]
                    label = CLASS_NAMES.get(cls, f"CLS_{cls}")
                    boxes.append((cls, x / 2, y / 2, HERO_DOT_SIZE, HERO_DOT_SIZE))
                    print(f"    {label:15s} at ({x//2:3d}, {y//2:3d})")
                elif event == cv2.EVENT_RBUTTONDOWN:
                    cls = 1  # shortcut: right click = red_hero
                    boxes.append((cls, x / 2, y / 2, HERO_DOT_SIZE, HERO_DOT_SIZE))
                    print(f"    red_hero        at ({x//2:3d}, {y//2:3d})")

            cv2.namedWindow("Label Minimap (11 Classes)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Label Minimap (11 Classes)", IMG_W * 2 + RIGHT_PANEL_W, IMG_H * 2 + BOTTOM_BAR_H)
            cv2.setMouseCallback("Label Minimap (11 Classes)", mouse_cb)

            while True:
                vis = draw_overlay(display.copy(), boxes, frame_idx, total_frames, fps, vid.name, _label_mode)
                cv2.imshow("Label Minimap (11 Classes)", vis)
                k = cv2.waitKey(1) & 0xFF

                # Class mode switching
                if k in CLASS_KEYS:
                    _label_mode[0] = CLASS_KEYS[k]
                    mode_name = CLASS_NAMES[_label_mode[0]]
                    print(f"  Mode: [{_label_mode[0]}] {mode_name}")
                elif k == 9:  # TAB key
                    _label_mode[0] = (_label_mode[0] + 1) % 11
                    mode_name = CLASS_NAMES[_label_mode[0]]
                    print(f"  Mode: [{_label_mode[0]}] {mode_name}")
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
                        save_counter += 1
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
                        save_counter += 1
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


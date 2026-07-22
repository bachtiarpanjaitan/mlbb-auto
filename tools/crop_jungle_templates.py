#!/usr/bin/env python3
"""
Crop Jungle Minimap Icon Templates
Mengekstrak patch ikon jungle/creep dari dataset minimap yang sudah terlabeli (class 2)
dan menyimpannya ke assets/creeps_minimap/ sebagai template matching.

Penggunaan:
  python tools/crop_jungle_templates.py
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "trainings" / "hero_detector"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "creeps_minimap"

CREEP_CLASSES = {
    "1": "turtle",          # Turtle
    "2": "lord",            # Lord
    "3": "thunder_fenrir",  # Thunder Fenrir (Purple/Blue Buff)
    "4": "molten_fiend",    # Molten Fiend (Orange/Red Buff)
    "5": "crab",            # Crab / Klomang
    "6": "lithowanderer",   # Lithowanderer / Walkie Grass
    "7": "fire_beetle",     # Fire Beetle
    "8": "horned_lizard",   # Horned Lizard
    "9": "lava_golem",      # Lava Golem
}


def load_jungle_patches():
    """Cari semua patch class 2 (jungle) dari dataset train & val."""
    patches = []

    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            seen_centers = []
            with open(lbl_path, "r") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id != 2:  # Hanya class jungle
                        continue

                    cx_norm, cy_norm = float(parts[1]), float(parts[2])
                    w_norm, h_norm = float(parts[3]), float(parts[4])

                    cx, cy = int(cx_norm * w), int(cy_norm * h)
                    bw, bh = int(w_norm * w), int(h_norm * h)

                    # Deduplication: Abaikan jika ada dot jungle lain dalam radius < 14px di gambar yang sama
                    if any(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 < 14 for pcx, pcy in seen_centers):
                        continue
                    seen_centers.append((cx, cy))

                    # Crop patch
                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(w, cx + bw // 2)
                    y2 = min(h, cy + bh // 2)

                    patch = img[y1:y2, x1:x2]
                    if patch.size > 0:
                        patches.append({
                            "img_path": img_path,
                            "img_name": img_path.name,
                            "split": split,
                            "label_idx": idx + 1,
                            "cx": cx, "cy": cy,
                            "bw": bw, "bh": bh,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "patch": patch
                        })

    return patches


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patches = load_jungle_patches()

    if not patches:
        print(f"❌ Tidak ada label class 2 (jungle) ditemukan di {DATASET_DIR}")
        print("   Silakan pelajari / labeli objek jungle terlebih dahulu di inspect/label_minimap.")
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 🌿 MLBB Jungle Icon Crop Tool (9 Jenis Creeps)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" Ditemukan: {len(patches)} patch jungle terlabeli")
    print(f" Output folder: {OUTPUT_DIR.absolute()}")
    print(" Kontrol Simpan Template:")
    print("   [ 1 ] : Turtle              [ 2 ] : Lord")
    print("   [ 3 ] : Thunder Fenrir (Blue Buff)")
    print("   [ 4 ] : Molten Fiend (Red Buff)")
    print("   [ 5 ] : Crab (Klomang)      [ 6 ] : Lithowanderer")
    print("   [ 7 ] : Fire Beetle         [ 8 ] : Horned Lizard")
    print("   [ 9 ] : Lava Golem")
    print("   [ Space / N ] : Skip patch  [ Q / ESC ] : Keluar")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    window_name = "Jungle Minimap Icon Cropper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 560)

    idx = 0
    saved_count = {}

    while idx < len(patches):
        item = patches[idx]
        patch = item["patch"]
        full_img = cv2.imread(str(item["img_path"]))

        if full_img is None:
            idx += 1
            continue

        h_full, w_full = full_img.shape[:2]

        # Gambarkan highlight kotak merah & lingkaran di posisi minimap
        vis_minimap = full_img.copy()
        cx, cy = item["cx"], item["cy"]
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]

        cv2.rectangle(vis_minimap, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis_minimap, (cx, cy), 12, (0, 255, 255), 2)
        cv2.circle(vis_minimap, (cx, cy), 3, (0, 0, 255), -1)

        # Scale full minimap ke ukuran ~320x310
        target_map_w, target_map_h = 320, 310
        scaled_map = cv2.resize(vis_minimap, (target_map_w, target_map_h), interpolation=cv2.INTER_LINEAR)

        # Zoom 6x untuk cropped patch
        zoom_scale = 6
        ph, pw = patch.shape[:2]
        resized_patch = cv2.resize(patch, (pw * zoom_scale, ph * zoom_scale), interpolation=cv2.INTER_NEAREST)

        # UI Canvas (800 x 560)
        canvas_h, canvas_w = 560, 800
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:60, :] = (35, 35, 40)

        # Header info
        cv2.putText(canvas, f"Patch ({idx + 1}/{len(patches)}): {item['img_name']} [{item['split'].upper()}]",
                    (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 220, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Posisi Minimap: ({cx}, {cy})  |  Ukuran: {pw}x{ph} px", (15, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Label Kolom
        cv2.putText(canvas, "📌 Posisi Minimap (Sorotan Merah)", (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "🔍 Zoom Detail Patch", (410, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        # Tempel gambar minimap penuh (Sisi Kiri)
        map_x, map_y = 20, 95
        canvas[map_y:map_y+target_map_h, map_x:map_x+target_map_w] = scaled_map
        cv2.rectangle(canvas, (map_x - 1, map_y - 1), (map_x + target_map_w + 1, map_y + target_map_h + 1), (100, 100, 100), 1)

        # Tempel patch di-zoom (Sisi Kanan)
        patch_x = 410 + (360 - resized_patch.shape[1]) // 2
        patch_y = map_y + (target_map_h - resized_patch.shape[0]) // 2
        canvas[patch_y:patch_y+resized_patch.shape[0], patch_x:patch_x+resized_patch.shape[1]] = resized_patch
        cv2.rectangle(canvas, (patch_x - 2, patch_y - 2),
                      (patch_x + resized_patch.shape[1] + 2, patch_y + resized_patch.shape[0] + 2), (50, 220, 50), 2)

        # Footer Bantuan Tombol (3 Baris Rapi)
        cv2.rectangle(canvas, (0, 420), (canvas_w, canvas_h), (25, 25, 30), -1)
        cv2.putText(canvas, "[1] turtle  [2] lord  [3] thunder_fenrir (Purple)  [4] molten_fiend (Orange)",
                    (15, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[5] crab  [6] lithowanderer  [7] fire_beetle  [8] horned_lizard  [9] lava_golem",
                    (15, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[Space/N] Skip Patch  |  [P/Left] Prev Patch  |  [Q/ESC] Quit",
                    (15, 512), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(0)

        if key in (27, ord('q'), ord('Q')):
            break

        char_key = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""

        if char_key in CREEP_CLASSES:
            creep_name = CREEP_CLASSES[char_key]
            save_path = OUTPUT_DIR / f"{creep_name}.png"
            cv2.imwrite(str(save_path), patch)
            saved_count[creep_name] = saved_count.get(creep_name, 0) + 1
            print(f"  ✅ Saved ({creep_name}): {save_path.name} (pos: {cx},{cy})")
            idx += 1
        elif key in (32, ord('n'), ord('N'), 83, 65363):  # Space / N / Right Arrow -> Skip
            idx += 1
        elif key in (81, 65361, ord('p'), ord('P')):  # Left Arrow / P -> Prev
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print("\n📊 Ringkasan Template Tersimpan:")
    for k, v in saved_count.items():
        print(f"  - {k}.png: {v} kali")


if __name__ == "__main__":
    main()

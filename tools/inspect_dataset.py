#!/usr/bin/env python3
"""
MLBB Minimap Dataset Inspector
Memeriksa gambar training & validation beserta label YOLO (.txt).

Fitur:
  - Navigasi tombol Panah Kiri (←) / Panah Kanan (→) atau A / D / N / P
  - Menampilkan lingkaran/kotak bounding box label (Class 0: blue_hero, Class 1: red_hero)
  - Zoom-in preview minimap dengan informasi detail piksel & label

Penggunaan:
  python tools/inspect_dataset.py
  python tools/inspect_dataset.py --val   # Periksa dataset validation
  python tools/inspect_dataset.py --all   # Periksa train + val
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Warna BGR
BLUE = (255, 180, 30)
RED = (30, 30, 255)
GREEN = (50, 220, 50)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (120, 120, 120)

CLASS_NAMES = {
    0: "blue_hero", 1: "red_hero",
    2: "lord", 3: "turtle",
    4: "thunder_fenrir", 5: "molten_fiend",
    6: "lithowanderer", 7: "crab",
    8: "lava_golem", 9: "fire_beetle", 10: "horned_lizard"
}

CLASS_COLORS = {
    0: (255, 180, 30),    # Blue
    1: (30, 30, 255),     # Red
    2: (255, 0, 255),     # Lord (Magenta)
    3: (0, 255, 0),       # Turtle (Green)
    4: (255, 255, 0),     # Thunder Fenrir (Cyan)
    5: (0, 140, 255),     # Molten Fiend (Orange)
    6: (200, 255, 0),     # Lithowanderer (Yellow-Green)
    7: (0, 215, 255),     # Crab (Gold)
    8: (180, 0, 180),     # Lava Golem (Purple)
    9: (0, 100, 255),     # Fire Beetle (Dark Orange)
    10: (255, 100, 200),  # Horned Lizard (Pink)
}

CLASS_KEYS = {
    ord('b'): 0, ord('r'): 1, ord('l'): 2, ord('t'): 3,
    ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3,
    ord('4'): 4, ord('5'): 5, ord('6'): 6, ord('7'): 7,
    ord('8'): 8, ord('9'): 9, ord('k'): 10, ord('K'): 10,
}

def load_dataset_items(base_dir: Path, split: str = "train") -> list[tuple[Path, Path | None]]:
    """Cari semua file gambar dan pasangannya file label .txt"""
    img_dir = base_dir / "images" / split
    lbl_dir = base_dir / "labels" / split

    if not img_dir.exists():
        return []

    items = []
    extensions = ("*.png", "*.jpg", "*.jpeg")
    img_paths = []
    for ext in extensions:
        img_paths.extend(sorted(img_dir.glob(ext)))

    for img_path in sorted(img_paths):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        items.append((img_path, lbl_path if lbl_path.exists() else None))

    return items


def save_labels(lbl_path: Path, labels_data: list[tuple[int, float, float, float, float]]):
    """Simpan list label ke file txt format YOLO."""
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        for cls_id, cx, cy, bw, bh in labels_data:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def deduplicate_labels(labels: list[tuple[int, float, float, float, float]], w: int = 350, h: int = 340, dist_thresh: float = 12.0) -> tuple[list[tuple[int, float, float, float, float]], bool]:
    """Hapus label duplikat yang berjarak < dist_thresh piksel di gambar yang sama."""
    deduped = []
    seen = []
    has_changed = False

    for cls_id, cx_n, cy_n, w_n, h_n in labels:
        cx_px, cy_px = cx_n * w, cy_n * h
        is_dup = False
        for p_cls, p_cx, p_cy in seen:
            if p_cls == cls_id and ((cx_px - p_cx) ** 2 + (cy_px - p_cy) ** 2) ** 0.5 < dist_thresh:
                is_dup = True
                has_changed = True
                break
        if not is_dup:
            seen.append((cls_id, cx_px, cy_px))
            deduped.append((cls_id, cx_n, cy_n, w_n, h_n))

    return deduped, has_changed


def load_raw_labels(lbl_path: Path | None, auto_clean: bool = True) -> list[tuple[int, float, float, float, float]]:
    """Baca data mentah label YOLO dari file txt dengan auto-deduplikasi."""
    if lbl_path is None or not lbl_path.exists():
        return []
    labels = []
    try:
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    labels.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        if auto_clean and labels:
            clean_labels, has_changed = deduplicate_labels(labels)
            if has_changed:
                save_labels(lbl_path, clean_labels)
                print(f"  🧹 Auto-cleaned duplicate labels in {lbl_path.name}")
                return clean_labels
    except Exception as e:
        print(f" ⚠️ Error membaca {lbl_path}: {e}")
    return labels


def draw_labels_on_image(img: np.ndarray, labels_data: list[tuple[int, float, float, float, float]]) -> tuple[np.ndarray, list[dict]]:
    """Gambarkan bounding box dan titik pusat label YOLO di gambar."""
    vis = img.copy()
    h, w = img.shape[:2]
    labels_info = []

    for idx, (cls_id, cx_norm, cy_norm, w_norm, h_norm) in enumerate(labels_data):
        # Konversi koordinat ternormalisasi ke piksel
        cx = int(cx_norm * w)
        cy = int(cy_norm * h)
        bw = int(w_norm * w)
        bh = int(h_norm * h)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + bh // 2)

        color = CLASS_COLORS.get(cls_id, GREEN)
        cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        # Gambar Kotak + Lingkaran Tengah
        r = max(bw, bh) // 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (cx, cy), max(2, r), color, 2)
        cv2.circle(vis, (cx, cy), 2, (255, 255, 255), -1)

        # Label teks
        text = f"#{idx+1} {cls_name}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        ly = max(th + 4, y1)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), BLACK, -1)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, 1)
        cv2.putText(vis, text, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        labels_info.append({
            "id": idx + 1,
            "class_id": cls_id,
            "class_name": cls_name,
            "cx": cx, "cy": cy, "w": bw, "h": bh,
            "cx_norm": cx_norm, "cy_norm": cy_norm,
            "w_norm": w_norm, "h_norm": h_norm,
        })

    return vis, labels_info


def main():
    parser = argparse.ArgumentParser(description="MLBB Dataset Inspector")
    parser.add_argument("--dir", default="trainings/hero_detector", help="Path folder dataset")
    parser.add_argument("--val", action="store_true", help="Buka dataset validation")
    parser.add_argument("--all", action="store_true", help="Buka dataset train & val gabungan")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    items = []

    if args.all:
        items.extend(load_dataset_items(base_dir, "train"))
        items.extend(load_dataset_items(base_dir, "val"))
    elif args.val:
        items = load_dataset_items(base_dir, "val")
    else:
        items = load_dataset_items(base_dir, "train")

    if not items:
        print(f"❌ Tidak ada gambar ditemukan di {base_dir.absolute()}")
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 🔍 MLBB Dataset Inspector & Editor")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" Total sampel: {len(items)} gambar")
    print(" Kontrol Navigasi:")
    print("   [ -> / D / N / Space ] : Gambar Selanjutnya")
    print("   [ <- / A / P ]         : Gambar Sebelumnya")
    print("   [ X / DEL ]            : Hapus Gambar + File Label")
    print("   [ Q / ESC ]            : Keluar")
    print(" Kontrol Labeling (Interaktif):")
    print("   [ Klik Kiri pada Kosong ] : Tambah dot mode aktif")
    print("   [ Klik pada Dot Ada ]    : Hapus dot tersebut")
    print("   [ Klik Kanan ]           : Tambah red_hero (shortcut) / Hapus dot")
    print("   [ 0-9, K, B, R, L, T ]   : Pilih Class (0=Blue, 1=Red, 2-10=Jungle)")
    print("   [ TAB ]                  : Ganti class berikutnya")
    print("   [ U ] = Undo Dot  | [ C ] = Clear All Dots | [ E ] = Crop Jungle")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    current_idx = 0
    active_class = 0  # Default class: blue_hero
    display_scale = 1.6
    header_h = 130
    RIGHT_PANEL_W = 260

    # Shortcut legend for right panel
    CLASS_LABELS_SHORT = {
        0:  "0/B: blue_hero",
        1:  "1/R: red_hero",
        2:  "2/L: lord",
        3:  "3/T: turtle",
        4:  "4:   blue_buff (fenrir)",
        5:  "5:   red_buff (fiend)",
        6:  "6:   lithowanderer",
        7:  "7:   crab (gold)",
        8:  "8:   lava_golem",
        9:  "9:   fire_beetle",
        10: "K/10:horned_lizard",
    }

    window_name = "MLBB Minimap Dataset Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 700 + RIGHT_PANEL_W, 750)

    # Variables for mouse callback
    mouse_event_triggered = [False]
    last_action_msg = [""]

    def mouse_callback(event, x, y, flags, param):
        nonlocal active_class
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return

        img_path, lbl_path = items[current_idx]
        if lbl_path is None:
            split_dir = "val" if "val" in str(img_path) else "train"
            lbl_path = base_dir / "labels" / split_dir / f"{img_path.stem}.txt"
            items[current_idx] = (img_path, lbl_path)

        img_temp = cv2.imread(str(img_path))
        if img_temp is None:
            return
        h, w = img_temp.shape[:2]

        # Calculate coordinates relative to image canvas (ignore right panel)
        rel_y = y - header_h
        if rel_y < 0 or rel_y >= int(h * display_scale) or x < 0 or x >= int(w * display_scale):
            return

        img_x = int(x / display_scale)
        img_y = int(rel_y / display_scale)

        raw_labels = load_raw_labels(lbl_path)

        # Check if clicked near an existing dot (within 14px radius)
        hit_idx = -1
        min_dist = 999.0
        for i, (c_id, cx_n, cy_n, w_n, h_n) in enumerate(raw_labels):
            cx_px, cy_px = int(cx_n * w), int(cy_n * h)
            dist = ((img_x - cx_px) ** 2 + (img_y - cy_px) ** 2) ** 0.5
            if dist < 16 and dist < min_dist:
                min_dist = dist
                hit_idx = i

        if hit_idx >= 0:
            # Delete clicked dot
            removed = raw_labels.pop(hit_idx)
            c_name = CLASS_NAMES.get(removed[0], f"cls_{removed[0]}")
            save_labels(lbl_path, raw_labels)
            last_action_msg[0] = f"🗑️ Label #{hit_idx+1} ({c_name}) dihapus!"
            print(f"  {last_action_msg[0]}")
        else:
            # Add new dot
            target_cls = 1 if event == cv2.EVENT_RBUTTONDOWN else active_class
            c_name = CLASS_NAMES.get(target_cls, f"cls_{target_cls}")
            dot_size = 34
            cx_norm = img_x / max(1, w)
            cy_norm = img_y / max(1, h)
            w_norm = dot_size / max(1, w)
            h_norm = dot_size / max(1, h)

            raw_labels.append((target_cls, cx_norm, cy_norm, w_norm, h_norm))
            save_labels(lbl_path, raw_labels)
            last_action_msg[0] = f"➕ Tambah {c_name} at ({img_x},{img_y})"
            print(f"  {last_action_msg[0]}")

        mouse_event_triggered[0] = True

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_path, lbl_path = items[current_idx]

        # Selalu cek ulang lbl_path dari disk — agar label yang ditambah
        # setelah startup (misal dari label_minimap.py) ikut terbaca
        split_dir = "val" if "val" in str(img_path) else "train"
        real_lbl = base_dir / "labels" / split_dir / f"{img_path.stem}.txt"
        if real_lbl.exists() and lbl_path is None:
            lbl_path = real_lbl
            items[current_idx] = (img_path, lbl_path)

        img = cv2.imread(str(img_path))

        if img is None:
            canvas = np.zeros((400, 500 + RIGHT_PANEL_W, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Gagal membaca: {img_path.name}", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        else:
            raw_labels = load_raw_labels(lbl_path)
            vis_img, labels_info = draw_labels_on_image(img, raw_labels)
            h, w = vis_img.shape[:2]

            # Skalakan gambar minimap agar nyaman dilihat
            dw, dh = int(w * display_scale), int(h * display_scale)
            resized_vis = cv2.resize(vis_img, (dw, dh), interpolation=cv2.INTER_NEAREST)

            # Buat canvas dengan right panel
            canvas = np.zeros((dh + header_h, dw + RIGHT_PANEL_W, 3), dtype=np.uint8)
            canvas[:header_h, :dw] = (30, 30, 35)

            # Judul & File Info
            split_tag = "VAL" if "val" in str(img_path) else "TRAIN"
            cv2.putText(canvas, f"[{split_tag}] ({current_idx + 1}/{len(items)}): {img_path.name}",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1, cv2.LINE_AA)

            # Label status
            blue_cnt = sum(1 for l in labels_info if l["class_id"] == 0)
            red_cnt = sum(1 for l in labels_info if l["class_id"] == 1)
            jungle_cnt = sum(1 for l in labels_info if l["class_id"] >= 2)

            if lbl_path and lbl_path.exists() and len(labels_info) > 0:
                lbl_status = f"Labels: {len(labels_info)} | Blue:{blue_cnt} Red:{red_cnt} Jungle:{jungle_cnt}"
                lbl_color = (200, 255, 200)
            else:
                lbl_status = "Belum ada label / Empty"
                lbl_color = RED if not (lbl_path and lbl_path.exists()) else (180, 180, 180)

            cv2.putText(canvas, lbl_status, (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lbl_color, 1, cv2.LINE_AA)

            # Detail koordinat label / Action msg
            if last_action_msg[0]:
                cv2.putText(canvas, last_action_msg[0], (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                details_str = " | ".join([f"#{l['id']}{l['class_name'][:2]}({l['cx']},{l['cy']})" for l in labels_info])
                if not details_str:
                    details_str = "Klik gambar untuk tambah dot"
                cv2.putText(canvas, details_str[:75], (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            # Baris Bantuan Navigasi
            cv2.rectangle(canvas, (0, 80), (dw, header_h), (45, 45, 55), -1)
            cv2.putText(canvas, "[Klik] Add/Del | [TAB] Ganti class | [U] Undo | [C] Clear | [X] Del | [<->] Nav",
                        (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "[0-9/K/B/R/L/T] Pilih Class | [E] Crop Jungle | [Q] Quit",
                        (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 220, 255), 1, cv2.LINE_AA)

            # Tempel gambar ke canvas
            canvas[header_h:, :dw] = resized_vis

            # ── RIGHT PANEL ──
            panel_x = dw
            canvas[:, panel_x:] = (20, 20, 25)
            cv2.line(canvas, (panel_x, 0), (panel_x, dh + header_h), (80, 80, 80), 2)

            cv2.putText(canvas, "SELECT CLASS", (panel_x + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "[0-9 / K / TAB]", (panel_x + 10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.line(canvas, (panel_x + 8, 50), (panel_x + RIGHT_PANEL_W - 8, 50), (80, 80, 80), 1)

            y_off = 72
            for cls_id in range(11):
                color = CLASS_COLORS[cls_id]
                lbl = CLASS_LABELS_SHORT[cls_id]
                count = sum(1 for l in labels_info if l["class_id"] == cls_id)

                if cls_id == active_class:
                    cv2.rectangle(canvas,
                                  (panel_x + 4, y_off - 16),
                                  (panel_x + RIGHT_PANEL_W - 4, y_off + 6),
                                  (60, 60, 75), -1)
                    cv2.rectangle(canvas,
                                  (panel_x + 4, y_off - 16),
                                  (panel_x + RIGHT_PANEL_W - 4, y_off + 6),
                                  color, 2)

                # Color swatch
                cv2.rectangle(canvas,
                              (panel_x + 10, y_off - 12),
                              (panel_x + 24, y_off + 2),
                              color, -1)

                cnt_str = f" ({count})" if count > 0 else ""
                txt_color = (255, 255, 255) if cls_id == active_class else (160, 160, 160)
                cv2.putText(canvas, f"{lbl}{cnt_str}",
                            (panel_x + 30, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, 1, cv2.LINE_AA)
                y_off += 24

            cv2.line(canvas, (panel_x + 8, y_off + 4), (panel_x + RIGHT_PANEL_W - 8, y_off + 4), (80, 80, 80), 1)
            cv2.putText(canvas, "TAB = cycle class",
                        (panel_x + 10, dh + header_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)

        # Selalu gunakan timeout kecil agar mouse click bisa memicu redraw
        # waitKeyEx(0) memblok indefinitely → klik mouse tidak trigger refresh
        mouse_event_triggered[0] = False
        key = cv2.waitKeyEx(16)  # ~60fps polling
        if key == -1:
            continue

        last_action_msg[0] = ""  # Clear message on key press

        # Handle Tombol Keyboard
        if key in (27, ord('q'), ord('Q')):  # ESC / Q
            break
        elif key in CLASS_KEYS:
            active_class = CLASS_KEYS[key]
            mode_name = CLASS_NAMES[active_class]
            last_action_msg[0] = f"Mode: {mode_name.upper()}"
        elif key == 9:  # TAB key
            active_class = (active_class + 1) % 11
            mode_name = CLASS_NAMES[active_class]
            last_action_msg[0] = f"Mode: {mode_name.upper()}"
        elif key in (ord('u'), ord('U')):  # Undo last dot
            raw_labels = load_raw_labels(lbl_path)
            if raw_labels:
                popped = raw_labels.pop()
                save_labels(lbl_path, raw_labels)
                c_name = CLASS_NAMES.get(popped[0], f"cls_{popped[0]}")
                last_action_msg[0] = f"↩️ Undo label {c_name}"
        elif key in (ord('c'), ord('C')):  # Clear all dots
            if lbl_path and lbl_path.exists():
                save_labels(lbl_path, [])
                last_action_msg[0] = "🧹 Semua label dihapus"
        elif key in (ord('e'), ord('E')):  # Crop jungle patches
            raw_labels = load_raw_labels(lbl_path)
            jungle_labels = [l for l in raw_labels if l[0] >= 2]
            if not jungle_labels:
                last_action_msg[0] = "⚠️ Tidak ada label jungle di gambar ini"
            else:
                out_dir = base_dir.parent.parent / "assets" / "creeps_minimap"
                out_dir.mkdir(parents=True, exist_ok=True)
                h, w = img.shape[:2]
                saved_count = 0
                for i, (cls_id, cx_n, cy_n, w_n, h_n) in enumerate(jungle_labels):
                    cx, cy = int(cx_n * w), int(cy_n * h)
                    bw, bh = int(w_n * w), int(h_n * h)
                    x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
                    x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)
                    patch = img[y1:y2, x1:x2]
                    if patch.size > 0:
                        save_name = f"crop_{img_path.stem}_{i+1}.png"
                        cv2.imwrite(str(out_dir / save_name), patch)
                        saved_count += 1
                last_action_msg[0] = f"✂️ Crop {saved_count} patch ke assets/creeps_minimap/"
                print(f"  {last_action_msg[0]}")
        elif key in (ord('x'), ord('X'), 127, 8, 3014656, 65535, 2162688):  # X / Delete / Backspace -> HAPUS Gambar
            img_path, lbl_path = items[current_idx]
            print(f" 🗑️ Hapus gambar & label: {img_path.name}")

            try:
                img_path.unlink(missing_ok=True)
                if lbl_path and lbl_path.exists():
                    lbl_path.unlink(missing_ok=True)
            except Exception as e:
                print(f" ❌ Gagal menghapus {img_path.name}: {e}")

            items.pop(current_idx)

            if not items:
                print(" ✅ Semua sampel gambar telah diperiksa/dihapus.")
                break

            if current_idx >= len(items):
                current_idx = max(0, len(items) - 1)

        elif key in (83, 65363, 2555904, ord('d'), ord('D'), ord('n'), ord('N'), 32):  # Right Arrow / D / N / Space -> Next
            current_idx = (current_idx + 1) % len(items)
        elif key in (81, 65361, 2424832, ord('a'), ord('A'), ord('p'), ord('P')):  # Left Arrow / A / P -> Prev
            current_idx = (current_idx - 1 + len(items)) % len(items)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

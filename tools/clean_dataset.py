#!/usr/bin/env python3
"""
Clean Dataset — MLBB Hero Detector
Menghapus gambar yang tidak memiliki data label (.txt) atau file labelnya kosong.
Menghapus juga file label yatim (tanpa gambar).

Usage:
  python tools/clean_dataset.py
  python tools/clean_dataset.py --dry-run   # Hanya simulasi (tidak menghapus)
"""

import argparse
from pathlib import Path

def clean_split(base_dir: Path, split: str, dry_run: bool = False):
    img_dir = base_dir / "images" / split
    lbl_dir = base_dir / "labels" / split

    if not img_dir.exists():
        return 0, 0

    removed_imgs = 0
    removed_lbls = 0

    # 1. Hapus gambar yang labelnya tidak ada / kosong
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        is_empty = False

        if not lbl_path.exists():
            is_empty = True
        else:
            # Cek isi file label
            try:
                content = lbl_path.read_text().strip()
                if not content:
                    is_empty = True
            except Exception:
                is_empty = True

        if is_empty:
            print(f" 🗑️ [{split.upper()}] Hapus gambar tanpa label: {img_path.name}")
            if not dry_run:
                img_path.unlink(missing_ok=True)
                if lbl_path.exists():
                    lbl_path.unlink(missing_ok=True)
            removed_imgs += 1

    # 2. Hapus label yatim (tanpa gambar)
    if lbl_dir.exists():
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            matching_img = [img_dir / f"{lbl_path.stem}{ext}" for ext in (".png", ".jpg", ".jpeg")]
            if not any(img.exists() for img in matching_img):
                print(f" 🗑️ [{split.upper()}] Hapus label tanpa gambar: {lbl_path.name}")
                if not dry_run:
                    lbl_path.unlink(missing_ok=True)
                removed_lbls += 1

    return removed_imgs, removed_lbls


def main():
    parser = argparse.ArgumentParser(description="Clean empty/unlabeled images in dataset")
    parser.add_argument("--dir", default="trainings/hero_detector", help="Path folder dataset")
    parser.add_argument("--dry-run", action="store_true", help="Hanya tampilkan file yang akan dihapus")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 🧹 MLBB Dataset Cleaner")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if args.dry_run:
        print(" ⚠️  MODE DRY-RUN (Simulasi saja, tidak ada file yang dihapus)\n")

    train_imgs, train_lbls = clean_split(base_dir, "train", dry_run=args.dry_run)
    val_imgs, val_lbls = clean_split(base_dir, "val", dry_run=args.dry_run)

    total_imgs = train_imgs + val_imgs
    total_lbls = train_lbls + val_lbls

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" ✅ Selesai! Total dihapus: {total_imgs} gambar, {total_lbls} label yatim")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

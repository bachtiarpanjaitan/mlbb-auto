"""
relabel_jungle.py — Auto-relabel label generik jungle (class 2) ke kelas spesifik per kamp.
Menggunakan koordinat kluster presisi dari ground-truth dataset & radius ketat (0.045).
"""

import argparse
import math
import shutil
from pathlib import Path

# ── Fixed jungle camp registry presisi berdasarkan kluster dataset ──────────
# Format: (class_id, name, norm_x, norm_y, radius)
JUNGLE_CAMPS = [
    # Lord & Turtle Pits (hanya titik persis di pit sungai)
    (2,  "lord",           0.500, 0.240, 0.045),
    (3,  "turtle",         0.500, 0.760, 0.045),

    # Thunder Fenrir (Blue Buff) — 2 lokasi
    (4,  "thunder_fenrir", 0.400, 0.390, 0.045),  # blue side
    (4,  "thunder_fenrir", 0.650, 0.550, 0.045),  # red side

    # Molten Fiend (Red Buff) — 2 lokasi
    (5,  "molten_fiend",   0.680, 0.700, 0.045),  # blue side
    (5,  "molten_fiend",   0.320, 0.290, 0.045),  # red side

    # Lithowanderer — 1 lokasi (tengah)
    (6,  "lithowanderer",  0.500, 0.500, 0.040),

    # Crab — 2 lokasi (top & bot)
    (7,  "crab",           0.200, 0.260, 0.045),  # top crab
    (7,  "crab",           0.810, 0.730, 0.045),  # bot crab

    # Lava Golem — 2 lokasi
    (8,  "lava_golem",     0.258, 0.515, 0.045),  # blue side golem
    (8,  "lava_golem",     0.756, 0.476, 0.045),  # red side golem

    # Fire Beetle — 2 lokasi
    (9,  "fire_beetle",    0.622, 0.814, 0.045),  # blue side beetle
    (9,  "fire_beetle",    0.387, 0.177, 0.045),  # red side beetle

    # Horned Lizard — 2 lokasi
    (10, "horned_lizard",  0.176, 0.425, 0.045),  # blue side lizard
    (10, "horned_lizard",  0.832, 0.565, 0.045),  # red side lizard
]

DATASET_DIR = Path("trainings/hero_detector")


def find_camp(cx: float, cy: float) -> tuple[int, str] | None:
    """Cari kamp terdekat untuk (cx, cy) dengan radius ketat 0.045."""
    best_dist = float("inf")
    best = None
    for cls_id, name, camp_x, camp_y, radius in JUNGLE_CAMPS:
        dist = math.sqrt((cx - camp_x) ** 2 + (cy - camp_y) ** 2)
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best = (cls_id, name)
    return best


def relabel_split(split: str, dry_run: bool) -> dict:
    label_dir = DATASET_DIR / "labels" / split
    backup_dir = DATASET_DIR / "labels" / f"{split}_backup"

    if not label_dir.is_dir():
        print(f"  [SKIP] {label_dir} tidak ditemukan")
        return {}

    # Restore from backup first to ensure clean state
    if backup_dir.exists():
        for f in label_dir.glob("*.txt"):
            f.unlink()
        for f in backup_dir.glob("*.txt"):
            shutil.copy(f, label_dir / f.name)
        print(f"  [RESTORED] Restored label murni dari {backup_dir}")

    stats = {
        "total_files": 0,
        "total_jungle_generic": 0,
        "matched": {},
        "unmatched": 0,
    }

    for txt_path in sorted(label_dir.glob("*.txt")):
        stats["total_files"] += 1
        lines = txt_path.read_text().splitlines()
        new_lines = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                new_lines.append(line)
                continue

            cls_id = int(parts[0])
            cx, cy = float(parts[1]), float(parts[2])
            rest = " ".join(parts[3:])

            if cls_id == 2:   # generik jungle lama
                stats["total_jungle_generic"] += 1
                camp = find_camp(cx, cy)
                if camp:
                    new_cls_id, camp_name = camp
                    stats["matched"][camp_name] = stats["matched"].get(camp_name, 0) + 1
                    new_lines.append(f"{new_cls_id} {cx} {cy} {rest}")
                    if new_cls_id != cls_id:
                        changed = True
                else:
                    stats["unmatched"] += 1
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if changed and not dry_run:
            txt_path.write_text("\n".join(new_lines) + "\n")

    return stats


def main():
    ap = argparse.ArgumentParser(description="Relabel jungle generic → kelas spesifik per kamp (presisi)")
    ap.add_argument("--dry-run", action="store_true", help="Preview perubahan tanpa menyimpan")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all", help="Split yang direlabel")
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    print("━" * 52)
    print("  Auto-relabel Jungle Kamp Presisi (Cluster Ground-Truth)")
    print(f"  Mode : {'DRY RUN' if args.dry_run else 'LIVE (Restore + Relabel)'}")
    print("━" * 52)

    for split in splits:
        print(f"\n[{split.upper()}]")
        stats = relabel_split(split, args.dry_run)
        if not stats:
            continue
        print(f"  Files   : {stats['total_files']}")
        print(f"  Jungle generik  : {stats['total_jungle_generic']}")
        print(f"  Cocok ke kamp   : {sum(stats['matched'].values())}")
        print(f"  Tidak cocok     : {stats['unmatched']}")
        if stats["matched"]:
            print("  Distribusi Baru yang Presisi:")
            for name, count in sorted(stats["matched"].items(), key=lambda x: -x[1]):
                print(f"    {name:<18}: {count}")

    print("\n" + "━" * 52)
    if args.dry_run:
        print("DRY RUN selesai.")
    else:
        print("Selesai! Selanjutnya: ./scripts/train.sh")


if __name__ == "__main__":
    main()

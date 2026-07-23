"""
Viewer — Lihat hasil record parquet di terminal.
Usage:
    python tools/view_parquet.py                              # list semua file
    python tools/view_parquet.py match_alpha_1                 # detail file
    python tools/view_parquet.py match_alpha_1 --head 5        # 5 baris pertama
    python tools/view_parquet.py match_alpha_1 --cols hero,hp  # kolom tertentu
"""

import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = Path("data/game_states")

try:
    import pandas as pd
except ImportError:
    print("Install dulu: pip install pandas pyarrow")
    sys.exit(1)


def list_files():
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        print(f"❌ Tidak ada file .parquet di {DATA_DIR}/")
        return
    print(f"📁 File di {DATA_DIR}/:")
    for f in files:
        size = f.stat().st_size / 1024
        df = pd.read_parquet(f)
        print(f"  {f.stem:30s} {size:>8.1f} KB  ({len(df)} frames, {len(df.columns)} columns)")


def show_file(name: str, head: int = 0, cols: list[str] | None = None):
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        # Coba partial match
        matches = list(DATA_DIR.glob(f"{name}*.parquet"))
        if not matches:
            print(f"❌ File '{name}.parquet' tidak ditemukan di {DATA_DIR}/")
            print(f"   Coba: python tools/view_parquet.py (tanpa argumen)")
            return
        path = matches[0]

    df = pd.read_parquet(path)
    size = path.stat().st_size / 1024
    print(f"\n📄 {path.name}  ({size:.1f} KB, {len(df)} frames, {len(df.columns)} columns)")
    print(f"   Range: frame {df['frame_idx'].min()} - {df['frame_idx'].max()}")
    print(f"   Duration: {df['video_time'].min():.1f}s - {df['video_time'].max():.1f}s")

    # Default: kolom penting aja, kecuali --cols all
    if cols:
        if "all" in [c.strip().lower() for c in cols]:
            pass  # tampilkan semua kolom
        else:
            col_patterns = [c.strip().lower() for c in cols]
            selected = [c for c in df.columns if any(p in c.lower() for p in col_patterns)]
            if selected:
                df = df[selected]
    else:
        # Default: kolom penting aja
        default_cols = [
            "frame_idx", "video_time",
            "hero_name", "level", "hp_pct", "mana_pct", "gold", "kda",
            "item_1", "item_2", "item_3",
            "skill_1_ready", "skill_2_ready", "skill_3_ready",
            "battle_spell_name", "battle_spell_ready", "battle_spell_remaining_cd",
            "blue_team_complete",
        ]
        # Tambah nama hero roster
        for team in ("blue", "red"):
            for slot in range(1, 6):
                default_cols.append(f"{team}_hero_{slot}_name")
                default_cols.append(f"{team}_hero_{slot}_hp_pct")
        # Tambah minimap positions (beberapa aja)
        for team in ("blue", "red"):
            for slot in range(1, 3):
                default_cols += [
                    f"{team}_mm_{slot}_name", f"{team}_mm_{slot}_norm_x",
                    f"{team}_mm_{slot}_norm_y", f"{team}_mm_{slot}_region",
                ]
        # Tambah jungle camp status
        status_cols = [c for c in df.columns if c.endswith("_status")]
        default_cols += sorted(status_cols)
        # Filter kolom yang beneran ada
        existing = [c for c in default_cols if c in df.columns]
        df = df[existing]

    # Drop kolom yang semua null
    df = df.dropna(axis=1, how="all")

    if head > 0:
        df = df.head(head)

    # Pretty print — transposed kalau banyak kolom
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", 50,
        "display.width", 140,
        "display.max_colwidth", 25,
    ):
        if len(df.columns) > 12:
            # Transpose: baris = kolom, kolom = frame
            print(f"\n{df.T.to_string()}")
        else:
            print(f"\n{df.to_string()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    name = None
    head = 10
    cols = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--head", "-n") and i + 1 < len(args):
            head = int(args[i + 1])
            i += 2
        elif a.startswith("--head="):
            head = int(a.split("=")[1])
            i += 1
        elif a in ("--cols", "-c") and i + 1 < len(args):
            cols = args[i + 1].split(",")
            i += 2
        elif a.startswith("--cols="):
            cols = a.split("=")[1].split(",")
            i += 1
        elif not a.startswith("-"):
            name = a
            i += 1
        else:
            i += 1

    if name:
        show_file(name, head=head, cols=cols)
    else:
        list_files()

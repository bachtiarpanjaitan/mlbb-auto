#!/usr/bin/env python3
"""
MLBB Auto — Main Launcher
Unified CLI untuk semua tool yang tersedia.

Usage:
  python main.py
  python main.py --tool train
  python main.py --tool label --video alpha_1.mp4
"""

import os
import sys
from pathlib import Path


# ── Colors (ANSI, auto-disabled kalau tidak didukung terminal) ──
class C:
    enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    RESET = "\033[0m" if enabled else ""
    BOLD = "\033[1m" if enabled else ""
    DIM = "\033[2m" if enabled else ""
    CYAN = "\033[36m" if enabled else ""
    YELLOW = "\033[33m" if enabled else ""
    GREEN = "\033[32m" if enabled else ""
    RED = "\033[31m" if enabled else ""
    MAGENTA = "\033[35m" if enabled else ""
    BLUE = "\033[34m" if enabled else ""


# ── Tool Registry ──
TOOLS = {
    "1": {
        "name": "Label Minimap (Manual)",
        "desc": "Label hero & jungle dots secara manual (keyboard + mouse)",
        "cmd": "python3 tools/label_minimap.py",
    },
    "2": {
        "name": "Auto-Label Heroes (YOLO)",
        "desc": "Deteksi hero biru/merah otomatis pakai YOLO model",
        "cmd": "python3 tools/pseudo_label.py",
    },
    "3": {
        "name": "Auto-Label Jungle (Template)",
        "desc": "Deteksi jungle camp otomatis (fixed position + template matching)",
        "cmd": "python3 tools/auto_label_jungle.py",
    },
    "4": {
        "name": "Auto-Label Full (Hero + Jungle)",
        "desc": "Gabungan: hero (YOLO) + jungle (template) dalam 1x jalan",
        "cmd": "python3 tools/auto_label_full.py",
    },
    "5": {
        "name": "Train YOLO Model",
        "desc": "Training YOLOv11n dengan dataset yang ada",
        "cmd": "python3 tools/train_yolo.py",
    },
    "6": {
        "name": "Run Inference",
        "desc": "Jalankan YOLO detection pada video",
        "cmd": "python3 tools/inference.py",
    },
    "7": {
        "name": "Debug Vision (Interactive)",
        "desc": "Video player dengan overlay deteksi (multi-thread)",
        "cmd": "python3 tools/debug_vision.py",
    },
    "8": {
        "name": "Inspect Dataset",
        "desc": "Lihat statistik & sample dari dataset",
        "cmd": "python3 tools/inspect_dataset.py",
    },
    "9": {
        "name": "Clean Dataset",
        "desc": "Bersihkan label kosong & gambar orphan",
        "cmd": "python3 tools/clean_dataset.py",
    },
    "10": {
        "name": "Region Editor",
        "desc": "Draw polygon regions pada minimap",
        "cmd": "python3 tools/region_editor.py",
    },
    "11": {
        "name": "Compress Video",
        "desc": "Kompres video untuk mengurangi ukuran file",
        "cmd": "python3 tools/compress_video.py",
    },
    "12": {
        "name": "Crawl Game Data",
        "desc": "Download hero, spell, item, creep databases",
        "cmd": "python crawlings/crawl_heroes.py",
    },
    "13": {
        "name": "Extract Jungle Templates",
        "desc": "Crop template jungle dari data yang sudah di-label",
        "cmd": "python3 tools/crop_jungle_templates.py",
    },
    "14": {
        "name": "Simulate Minimap",
        "desc": "Replay hero positions dari data .parquet (animasi minimap)",
        "cmd": "python3 tools/simulate_minimap.py",
    },
    "15": {
        "name": "Dataset Status",
        "desc": "Lihat ringkasan lengkap dataset (class distribution)",
        "cmd": None,  # Built-in
    },
    "16": {
        "name": "Interactive Label Tool",
        "desc": "Label minimap via GUI (B=blue, R=red, J=jungle)",
        "cmd": "bash scripts/label_minimap.sh",
    },
    "17": {
        "name": "Train Skill Classifier",
        "desc": "Training CNN untuk deteksi cooldown skill hero",
        "cmd": "bash scripts/train_skill.sh",
    },
    "18": {
        "name": "Train CD Number",
        "desc": "Training OCR untuk angka cooldown skill",
        "cmd": "bash scripts/train_cd.sh",
    },
    "19": {
        "name": "Crop Skills Dataset",
        "desc": "Crop dataset skill icon dari video untuk training",
        "cmd": "python3 tools/crop_skills_dataset.py",
    },
    "20": {
        "name": "Export to Kaggle",
        "desc": "Package dataset (.zip) + notebook siap upload ke Kaggle",
        "cmd": "python3 tools/export_kaggle.py --zip",
    },
}

# ── Quick presets ──
PRESETS = {
    "a": {
        "name": "Auto-Label ALL (Hero + Jungle)",
        "desc": "Jalankan auto-label full untuk semua video",
        "cmd": "python3 tools/auto_label_full.py --all-videos",
    },
    "b": {
        "name": "Full Pipeline: Label -> Train -> Export",
        "desc": "Auto-label semua video, lalu train model",
        "cmd": None,  # Multi-step
    },
    "x": {
        "name": "Exit",
        "desc": "Keluar dari main.py",
        "cmd": None,
    },
}


# ── Layout constants (single source of truth so every border/row lines up) ──
KEY_W = 3     # width reserved for the menu key, e.g. " 12"
NAME_W = 40   # width reserved for the tool name (longest name must fit)


def _cell(key: str, name: str) -> str:
    """One column's inner content, colored version (variable length due to ANSI codes)."""
    key_s = f"{C.YELLOW}{key:>{KEY_W}}{C.RESET}" if key else " " * KEY_W
    name_s = f"{name:<{NAME_W}}"
    return f" {key_s} {C.DIM}│{C.RESET} {name_s} "


# Visible (non-colored) length of one cell — used for border math, since ANSI
# escape codes are invisible but count toward len() otherwise.
_CELL_LEN = len(f" {'x' * KEY_W} │ {'x' * NAME_W} ")


def _row(key1, name1, key2, name2) -> str:
    return f"  {C.DIM}│{C.RESET}{_cell(key1, name1)}{C.DIM}│{C.RESET}{_cell(key2, name2)}{C.DIM}│{C.RESET}"


def _border(left: str, mid: str, right: str) -> str:
    seg = "─" * _CELL_LEN
    return f"  {C.DIM}{left}{seg}{mid}{seg}{right}{C.RESET}"


def _title_row(title: str) -> str:
    full = _CELL_LEN * 2 + 1  # both cells + the shared divider column
    return f"  {C.DIM}│{C.RESET}{C.BOLD}{title:^{full}}{C.RESET}{C.DIM}│{C.RESET}"


def print_banner():
    width = _CELL_LEN * 2 + 3  # matches the full menu box width
    print()
    print(f"  {C.CYAN}┌{'─' * (width - 4)}┐{C.RESET}")
    print(f"  {C.CYAN}│{C.RESET}{C.BOLD}{'MLBB AUTO  —  MAIN MENU':^{width - 4}}{C.RESET}{C.CYAN}│{C.RESET}")
    print(f"  {C.CYAN}│{C.RESET}{C.DIM}{'Mobile Legends: Bang Bang Vision Pipeline':^{width - 4}}{C.RESET}{C.CYAN}│{C.RESET}")
    print(f"  {C.CYAN}└{'─' * (width - 4)}┘{C.RESET}")
    print()


def print_menu():
    # ── TOOLS ──
    print(_border("┌", "┬", "┐"))
    print(_title_row("TOOLS"))
    print(_border("├", "┼", "┤"))

    items = sorted(TOOLS.items(), key=lambda kv: int(kv[0]))
    mid_idx = (len(items) + 1) // 2
    col1 = items[:mid_idx]
    col2 = items[mid_idx:]

    for i in range(max(len(col1), len(col2))):
        k1, t1 = col1[i] if i < len(col1) else ("", {"name": ""})
        k2, t2 = col2[i] if i < len(col2) else ("", {"name": ""})
        n1 = t1["name"][:NAME_W]
        n2 = t2["name"][:NAME_W]
        print(_row(k1, n1, k2, n2))

    # ── QUICK ACTIONS ──
    print(_border("├", "┴", "┤"))
    print(_title_row("QUICK ACTIONS"))
    print(_border("├", "┬", "┤"))

    presets = sorted(PRESETS.items())
    for i in range(0, len(presets), 2):
        k1, p1 = presets[i]
        n1 = p1["name"][:NAME_W]
        if i + 1 < len(presets):
            k2, p2 = presets[i + 1]
            n2 = p2["name"][:NAME_W]
        else:
            k2, n2 = "", ""
        print(_row(k1, n1, k2, n2))

    print(_border("└", "┴", "┘"))
    print()


def dataset_status():
    """Built-in: tampilkan status dataset."""
    from collections import Counter

    d = Path("trainings/hero_detector")
    lbl_dir = d / "labels" / "train"
    img_dir = d / "images" / "train"

    if not lbl_dir.exists():
        print(f"\n  {C.RED}❌ Dataset belum ada. Jalankan labeling dulu.{C.RESET}")
        return

    labels = list(lbl_dir.glob("*.txt"))
    images = list(img_dir.glob("*.png"))

    c = Counter()
    total_annotations = 0
    for f in labels:
        for line in f.read_text().strip().split("\n"):
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                c[cls] += 1
                total_annotations += 1

    names = {
        0: "hero",
        2: "legend",
        3: "turtle",
        4: "thunder_fenrir",
        5: "molten_fiend",
        6: "lithowanderer",
        7: "crab",
        8: "lava_golem",
        9: "fire_beetle",
        10: "horned_lizard",
    }

    # Cek model
    model_onnx = Path("models/hero_tracker.onnx")
    model_pt = d / "yolo11n_minimap" / "weights" / "best.pt"

    print(f"\n  {C.BOLD}📊 Dataset:{C.RESET} {len(images)} images, {len(labels)} label files")
    print(f"  {C.DIM}{'─' * 45}{C.RESET}")
    print(f"  Total annotations: {C.BOLD}{total_annotations}{C.RESET}")
    for cls in sorted(c.keys()):
        name = names.get(cls, f"class_{cls}")
        cnt = c[cls]
        bar = "█" * int(cnt / max(total_annotations, 1) * 30)
        print(f"    {cls:2d} {name:20s} {cnt:5d} {C.GREEN}{bar}{C.RESET}")
    print()
    onnx_mark = f"{C.GREEN}✅{C.RESET}" if model_onnx.exists() else f"{C.RED}❌{C.RESET}"
    pt_mark = f"{C.GREEN}✅{C.RESET}" if model_pt.exists() else f"{C.RED}❌{C.RESET}"
    print(f"  Model ONNX: {onnx_mark} {model_onnx}")
    print(f"  Model PT:   {pt_mark} {model_pt}")
    print()


def run_full_pipeline():
    """Multi-step: auto-label semua video -> train."""
    print(f"\n  {C.MAGENTA}⚙️  Full Pipeline: Auto-Label -> Train{C.RESET}")
    print(f"  {C.DIM}{'─' * 45}{C.RESET}")

    # Step 1: Auto-label
    print("\n  [1/2] Auto-label full untuk semua video...")
    ret = os.system("python3 tools/auto_label_full.py --all-videos")
    if ret != 0:
        print(f"  {C.RED}❌ Auto-label gagal. Cek error di atas.{C.RESET}")
        input("\n  Press Enter untuk kembali ke menu...")
        return

    # Step 2: Clean
    print("\n  [*] Bersihkan data kosong...")
    d = Path("trainings/hero_detector")
    lbl_dir = d / "labels" / "train"
    img_dir = d / "images" / "train"
    for f in list(lbl_dir.glob("*.txt")):
        if not f.read_text().strip():
            f.unlink()
    for f in list(img_dir.glob("*.png")):
        if not (lbl_dir / (f.stem + ".txt")).exists():
            f.unlink()

    # Step 3: Train
    print("\n  [2/2] Training model...")
    ret = os.system("python3 tools/train_yolo.py")
    if ret != 0:
        print(f"  {C.RED}❌ Training gagal.{C.RESET}")
        input("\n  Press Enter untuk kembali ke menu...")
        return

    print(f"\n  {C.GREEN}✅ Pipeline selesai!{C.RESET}")
    input("\n  Press Enter untuk kembali ke menu...")


def run_tool(choice: str, extra_args: str = ""):
    """Run a tool by menu key."""
    # Check presets first
    if choice in PRESETS:
        preset = PRESETS[choice]
        if choice == "a":
            cmd = preset["cmd"]
        elif choice == "b":
            run_full_pipeline()
            return
        elif choice in ("x", "q"):
            print(f"\n  {C.CYAN}👋 Bye!{C.RESET}")
            sys.exit(0)
        else:
            cmd = preset.get("cmd", "")
    elif choice in TOOLS:
        tool = TOOLS[choice]
        cmd = tool["cmd"]
        if cmd is None:  # Built-in tool
            if choice == "15":
                dataset_status()
            return
    else:
        print(f"\n  {C.RED}❌ Pilihan '{choice}' tidak valid.{C.RESET}")
        return

    # Prefer .venv python untuk hindari bentrok numpy/dependency
    venv_py = Path(".venv/bin/python3")
    py = str(venv_py) if venv_py.exists() else sys.executable
    full_cmd = cmd.replace("python3 ", f"{py} ").replace("python ", f"{py} ")
    if extra_args:
        full_cmd = f"{full_cmd} {extra_args}"
    print(f"\n  {C.CYAN}🚀 Running:{C.RESET} {full_cmd}\n")
    ret = os.system(full_cmd)
    if ret != 0:
        print(f"\n  {C.YELLOW}⚠️  Tool selesai dengan kode {ret}{C.RESET}")
    # Pause di interactive mode saja
    if _interactive:
        try:
            input("\n  Press Enter untuk kembali ke menu...")
        except EOFError:
            pass


def interactive_mode():
    """Interactive menu loop."""
    while True:
        os.system("clear" if os.name == "posix" else "cls")
        print_banner()
        print_menu()
        print()
        raw = input(f"  {C.BOLD}Pilih menu [1-20 / a/b/q/x]:{C.RESET} ")
        # ESC key biasa kirim \x1b, q/x untuk exit
        if raw in ("\x1b", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D", "", "q", "x"):
            print(f"\n  {C.YELLOW}👋 Bye!{C.RESET}")
            sys.exit(0)
        choice = raw.strip().lower()
        run_tool(choice)


_interactive = True


def main():
    # Parse only --tool flag, pass everything else to the tool
    tool_arg = None
    tool_extra = []
    args_list = sys.argv[1:]
    i = 0
    while i < len(args_list):
        if args_list[i] in ("--tool", "-t") and i + 1 < len(args_list):
            tool_arg = args_list[i + 1]
            i += 2
        else:
            tool_extra.append(args_list[i])
            i += 1

    global _interactive
    _interactive = tool_arg is None

    if tool_arg:
        # Find tool by exact key first (numbers, "a"/"b"/"x"), THEN by
        # partial name match — otherwise "x" would match a tool whose
        # name happens to contain an "x" (e.g. "Extract Jungle Templates").
        choice = None
        if tool_arg in TOOLS or tool_arg in PRESETS:
            choice = tool_arg
        else:
            for key, tool in TOOLS.items():
                if tool_arg.lower() in tool["name"].lower():
                    choice = key
                    break

        if choice:
            extra = " ".join(tool_extra)
            run_tool(choice, extra)
        else:
            print(f"❌ Tool '{tool_arg}' tidak ditemukan.")
            sys.exit(1)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
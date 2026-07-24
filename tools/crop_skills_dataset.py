#!/usr/bin/env python3
"""
crop_skills_dataset.py — Koleksi dataset skill icon dari video replay.

Crop tiap skill slot (skill_1, skill_2, skill_3, battle_spell) pakai bbox
dari layout.yaml, lalu label otomatis pakai SkillsDetector yang sudah ada.

Hasil:
  trainings/hero_skills/dataset/
    <video>/
      <hero>/
        skill_1/ready/    frame_0001.png
        skill_1/cooldown/ frame_0002.png
        skill_3/available/frame_0003.png
        battle_spell/ready/frame_0004.png
        ...
      metadata.csv   ← track asal + hero setiap sample

Cara pakai:
  python tools/crop_skills_dataset.py video.mp4 [--interval 15] [--max-frames 5000]
  python tools/crop_skills_dataset.py video.mp4 --hero alpha  # paksa hero name
  python tools/crop_skills_dataset.py video.mp4 --interactive  # review manual tiap crop
"""

from __future__ import annotations

import sys
import os
import argparse
import csv
import time
import logging
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.core import layout
from vision.core.cropper import crop_region
from vision.core.frame_reader import FrameReader
from vision.detectors import SkillsDetector
from vision.matcher.template import TemplateMatcher
from vision.ocr.reader import OCRReader

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("crop_skills_hero")
log.setLevel(logging.INFO)

# Skill slots yang terdefinisi di layout.yaml
SKILL_SLOTS = ("skill_1", "skill_2", "skill_3", "battle_spell")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trainings" / "hero_skills" / "dataset"


HERO_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "assets" / "heroes"


class DatasetCollector:
    """Crop skill regions from video, label with SkillsDetector, save dataset."""

    def __init__(self, output_dir: str | Path = OUTPUT_DIR, detect_hero: bool = True):
        self.output_dir = Path(output_dir)
        self.skills_det = SkillsDetector(OCRReader())
        self.ocr = OCRReader()
        self.metadata: list[dict] = []
        self._sample_count = 0

        # ── Hero matcher (portrait template matching) ──
        self._hero_matcher = None
        if detect_hero:
            self._hero_matcher = self._load_hero_matcher()

    def _load_hero_matcher(self) -> TemplateMatcher | None:
        """Load hero portrait templates for hero name detection."""
        pt_region = layout.get_region("hero_panel", "portrait")
        if pt_region and "bbox" in pt_region:
            pt_w, pt_h = pt_region["bbox"][2], pt_region["bbox"][3]
        else:
            pt_w, pt_h = 110, 100

        templates: dict[str, np.ndarray] = {}
        if not HERO_TEMPLATE_DIR.exists():
            log.warning("Hero templates dir not found: %s", HERO_TEMPLATE_DIR)
            return None

        for fname in sorted(os.listdir(str(HERO_TEMPLATE_DIR))):
            if fname.endswith(".png"):
                stem = fname[:-4]
                img = cv2.imread(str(HERO_TEMPLATE_DIR / fname))
                if img is not None:
                    img = cv2.resize(img, (pt_w, pt_h), interpolation=cv2.INTER_AREA)
                    templates[stem] = img

        if templates:
            log.info("Loaded %d hero templates (%dx%d)", len(templates), pt_w, pt_h)
            return TemplateMatcher(threshold=0.40, templates=templates)
        else:
            log.warning("No hero templates loaded")
            return None

    def detect_hero_name(self, frame: np.ndarray) -> str | None:
        """Detect hero name from portrait via template matching."""
        if self._hero_matcher is None:
            return None
        portrait = crop_region(frame, "hero_panel", "portrait")
        if portrait is None or portrait.size == 0:
            return None
        match = self._hero_matcher.match(portrait)
        if match and match.success:
            return match.label
        return None

    def collect_from_video(
        self,
        video_path: str,
        interval: int = 15,
        max_frames: int | None = None,
        interactive: bool = False,
        hero_override: str | None = None,
    ) -> int:
        """
        Process a video and collect skill crop samples.

        Args:
            video_path: Path to video file.
            interval: Process every Nth frame.
            max_frames: Stop after processing this many frames (None = entire video).
            interactive: Show each crop for manual approval.
            hero_override: Force hero name (skip auto-detection).

        Returns:
            Number of samples collected.
        """
        video_name = Path(video_path).stem
        # Auto-detect hero dari filename (alpha_1.mp4 → alpha)
        if hero_override is None and not self._hero_matcher:
            # Coba ambil dari nama file: "alpha_1" → "alpha"
            guess = video_name.split("_")[0] if "_" in video_name else video_name
            if (HERO_TEMPLATE_DIR / f"{guess}.png").exists():
                hero_override = guess
                log.info("Hero auto-detected from filename: %s", guess)

        print(f"\n🎬 Processing: {video_path}")
        print(f"   Interval: every {interval} frame(s)")
        if max_frames:
            print(f"   Max frames: {max_frames}")
        hero_out = hero_override or "<hero>"
        print(f"   Output: {self.output_dir / hero_out}/<slot>/<label>/")
        if hero_override:
            print(f"   Hero:   {hero_override} (override)")
        elif self._hero_matcher:
            print(f"   Hero:   auto-detect via portrait matching")
        else:
            print(f"   Hero:   none (use --hero untuk set manual)")
        print()

        fr = FrameReader(video_path)
        total_frames = int(fr.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fr.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"   Video: {total_frames} frames @ {fps:.2f} fps ({total_frames/fps/60:.1f} min)")

        self._sample_count = 0
        processed = 0
        frame_count = 0
        t_start = time.time()

        while True:
            frame = fr.read()
            if frame is None:
                break

            # Sampling interval
            if frame_count % interval != 0:
                frame_count += 1
                continue

            video_time = frame_count / fps

            # ── Hero name detection (sekali per frame, untuk semua slot) ──
            hero_name = hero_override
            if hero_name is None and self._hero_matcher is not None:
                portrait = crop_region(frame, "hero_panel", "portrait")
                if portrait is not None and portrait.size > 0:
                    match = self._hero_matcher.match(portrait)
                    if match and match.success:
                        hero_name = match.label
            if hero_name is None:
                hero_name = "unknown"

            # Collect crops for all skill slots
            for skill_name in SKILL_SLOTS:
                skill_img = crop_region(frame, "hero_panel", "skills", skill_name)
                if skill_img is None or skill_img.size == 0:
                    continue

                # Label pakai SkillsDetector yang sudah ada
                result = self.skills_det.run(skill_img)
                label = "unknown"
                confidence = 0.0

                if result and result.value:
                    value = result.value if isinstance(result.value, dict) else {}
                    if value.get("cooldown"):
                        label = "cooldown"
                    elif value.get("available"):
                        label = "available"
                    elif value.get("ready"):
                        label = "ready"
                    confidence = getattr(result, "confidence", 0.0) or 0.0

                # Skip samples dengan confidence terlalu rendah
                if confidence < 0.6 and not interactive:
                    continue

                # Interactive mode: show crop and ask user
                if interactive:
                    label = self._interactive_review(skill_img, skill_name, label, confidence)
                    if label is None:  # User skipped
                        continue

                # Save image (organize by hero name)
                sample_path = self._save_sample(video_name, hero_name, skill_name, label, skill_img)

                # Record metadata
                self.metadata.append({
                    "video": video_name,
                    "frame": frame_count,
                    "video_time": round(video_time, 2),
                    "hero": hero_name,
                    "skill_slot": skill_name,
                    "label": label,
                    "confidence": round(confidence, 3),
                    "filepath": str(sample_path),
                })
                self._sample_count += 1

            processed += 1
            frame_count += 1

            # Progress
            if processed % 50 == 0:
                elapsed = time.time() - t_start
                pct = frame_count / total_frames * 100 if total_frames else 0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"   [{pct:5.1f}%] frame={frame_count}/{total_frames} "
                      f"samples={self._sample_count} rate={rate:.0f} fr/s")

            # Max frames check
            if max_frames and processed >= max_frames:
                print(f"   [max] Reached max_frames={max_frames}")
                break

        fr.release()
        elapsed = time.time() - t_start
        print(f"\n✅ Done! Collected {self._sample_count} samples from {processed} frames "
              f"in {elapsed:.1f}s")

        # Hero untuk metadata (pakai override atau last detected)
        meta_hero = hero_override or hero_name or "unknown"
        self._save_metadata(video_name, meta_hero)
        return self._sample_count

    def _interactive_review(
        self, image: np.ndarray, skill_name: str, auto_label: str, confidence: float
    ) -> str | None:
        """Show crop and let user confirm/reassign label. Returns None to skip."""
        from_label = auto_label
        while True:
            display = cv2.resize(image, (140, 140), interpolation=cv2.INTER_NEAREST)
            # Show label on image
            info = f"{skill_name} | auto: {auto_label} (conf={confidence:.2f})"
            cv2.putText(display, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 255, 0), 1)

            # Draw colored border based on label
            border_color = {
                "ready": (0, 255, 0),      # Green
                "cooldown": (0, 0, 255),   # Red
                "available": (255, 165, 0), # Orange
                "unknown": (128, 128, 128), # Gray
            }.get(auto_label, (255, 255, 255))
            display = cv2.copyMakeBorder(display, 4, 4, 4, 4,
                                         cv2.BORDER_CONSTANT, value=border_color)

            cv2.imshow(f"Review: {skill_name}", display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(f"Review: {skill_name}")

            if key == ord(" "):  # Space = accept auto-label
                return auto_label
            elif key == ord("1"):
                return "ready"
            elif key == ord("2"):
                return "cooldown"
            elif key == ord("3"):
                return "available"
            elif key == ord("0"):
                return "empty"
            elif key in (ord("s"), ord("S")):  # Skip
                return None
            elif key == ord("q"):
                print("   [quit] Interactive review stopped by user")
                sys.exit(0)
            else:
                print(f"   Keys: [Space]=accept | [1]=ready [2]=cooldown [3]=available "
                      f"[0]=empty [s]=skip [q]=quit")

    def _save_sample(
        self, video_name: str, hero_name: str, skill_name: str, label: str, image: np.ndarray
    ) -> Path:
        """Save cropped image to per-hero directory.

        Struktur: dataset/<hero>/<slot>/<label>/<video>_frame_xxx.png
        """
        out_dir = self.output_dir / hero_name / skill_name / label
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob(f"{video_name}_*.png")))
        filename = f"{video_name}_frame_{existing:05d}.png"
        filepath = out_dir / filename

        cv2.imwrite(str(filepath), image)
        return filepath

    def _save_metadata(self, video_name: str, hero_name: str):
        """Append metadata to per-hero CSV."""
        csv_dir = self.output_dir / hero_name
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "metadata.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            if self.metadata:
                writer = csv.DictWriter(f, fieldnames=self.metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.metadata)

        print(f"   📋 Metadata: {csv_path} ({len(self.metadata)} rows)")

    def print_dataset_stats(self):
        """Print dataset statistics from collected samples."""
        if not self.metadata:
            print("   No data collected.")
            return

        from collections import Counter
        total = len(self.metadata)
        slots = Counter(m["skill_slot"] for m in self.metadata)
        labels = Counter(m["label"] for m in self.metadata)
        heroes = Counter(m.get("hero", "unknown") for m in self.metadata)
        slot_labels = Counter(f"{m['skill_slot']}/{m['label']}" for m in self.metadata)

        print(f"\n📊 Dataset Stats: {total} samples")
        print(f"\n   Per hero:")
        for hero, count in sorted(heroes.items()):
            print(f"     {hero}: {count}")
        print(f"\n   Per slot:")
        for slot, count in sorted(slots.items()):
            print(f"     {slot}: {count}")
        print(f"\n   Per label:")
        for label, count in sorted(labels.items()):
            print(f"     {label}: {count}")
        print(f"\n   Per slot × label:")
        for sl, count in sorted(slot_labels.items()):
            print(f"     {sl}: {count}")


# ═══════════════════════════════════════════════════════════════════════
#  Batch Mode: Process multiple videos
# ═══════════════════════════════════════════════════════════════════════

def batch_process(video_list: list[str], interval: int = 15, max_frames: int | None = None):
    """Process multiple videos in batch mode."""
    collector = DatasetCollector()
    total = 0

    for vpath in video_list:
        if not os.path.isfile(vpath):
            print(f"⚠️  File not found: {vpath}")
            continue
        count = collector.collect_from_video(vpath, interval=interval, max_frames=max_frames)
        total += count

    collector.print_dataset_stats()
    print(f"\n🎯 Grand total: {total} samples across {len(video_list)} video(s)")


# ═══════════════════════════════════════════════════════════════════════
#  Hero Picker
# ═══════════════════════════════════════════════════════════════════════

_HERO_DB: list[dict] | None = None

def _load_hero_list() -> list[dict]:
    """Load hero database (cached)."""
    global _HERO_DB
    if _HERO_DB is not None:
        return _HERO_DB
    db_path = Path(__file__).resolve().parent.parent / "assets" / "databases" / "heroes.json"
    if db_path.exists():
        with open(db_path) as f:
            _HERO_DB = json.load(f)
    else:
        # Fallback: scan PNG files in assets/heroes/
        hero_dir = Path(__file__).resolve().parent.parent / "assets" / "heroes"
        _HERO_DB = [{"key": f[:-4]} for f in sorted(os.listdir(str(hero_dir))) if f.endswith(".png")]
    return _HERO_DB


def _pick_hero_from_list(video_name: str) -> str | None:
    """Show interactive hero picker, return hero key or None to skip."""
    heroes = _load_hero_list()
    if not heroes:
        return None

    # Coba auto-detect dari filename: "alpha_1" -> "alpha", "lapu_2" -> "lapu"
    guess = video_name.split("_")[0].lower() if "_" in video_name else video_name.lower()
    hero_keys = [h["key"] for h in heroes]
    exact_match = guess if guess in hero_keys else None

    # Filter heroes yang match dengan guess
    if exact_match:
        default_idx = hero_keys.index(exact_match)
    else:
        # Cari partial match
        matching = [k for k in hero_keys if k.startswith(guess)]
        default_idx = hero_keys.index(matching[0]) if matching else 0

    print(f"\n🎮 Pilih hero untuk video '{video_name}':")
    print(f"   (tekan Enter untuk default, 0 untuk skip video, q untuk quit)")
    print()

    # Tampilkan dalam kolom
    cols = 4
    for i, h in enumerate(heroes):
        marker = " ⬅" if i == default_idx else ""
        print(f"   [{i+1:3d}] {h['key']:20s}{marker}", end="")
        if (i + 1) % cols == 0:
            print()
    if len(heroes) % cols != 0:
        print()

    while True:
        try:
            inp = input(f"\nNomor hero [{default_idx + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if inp.lower() in ("q", "quit"):
            print("   ❌ Quit")
            sys.exit(0)
        if inp == "0":
            print("   ⏭️  Skipped")
            return None
        if inp == "":
            return hero_keys[default_idx]
        try:
            idx = int(inp) - 1
            if 0 <= idx < len(heroes):
                return hero_keys[idx]
        except ValueError:
            pass
        # Coba match partial name
        inp_lower = inp.lower()
        matches = [k for k in hero_keys if k.startswith(inp_lower)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"   ❓ Multiple matches: {', '.join(matches)}")
        else:
            print(f"   ❓ Hero '{inp}' not found")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collect skill icon dataset from MLBB video replays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("videos", nargs="*", help="Video file(s) to process")
    parser.add_argument("--interval", type=int, default=15,
                        help="Process every Nth frame (default: 15 = ~0.5s at 30fps)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after processing this many sampled frames")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Review each crop manually")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats from existing dataset dir")
    parser.add_argument("--hero", type=str, default=None,
                        help="Force hero name (skip auto-detect)")
    parser.add_argument("--no-pick", action="store_true",
                        help="Skip interactive hero picker (auto-detect from filename)")

    args = parser.parse_args()

    # Stats mode
    if args.stats:
        collector = DatasetCollector()
        collector.print_dataset_stats()
        return

    # Interactive video picker kalo video tidak diberikan
    if not args.videos:
        vd = Path("videos")
        cs = sorted(vd.glob("*.mp4"))
        if not cs:
            print("❌ No .mp4 videos found in videos/")
            sys.exit(1)
        print("\n🎬 Pilih video untuk crop skills dataset:")
        for idx, vid in enumerate(cs, 1):
            size_mb = vid.stat().st_size / (1024 * 1024)
            print(f"  [{idx}] {vid.name} ({size_mb:.1f} MB)")
        while True:
            try:
                choice = input(f"\nMasukkan nomor video (wajib) [1-{len(cs)}]: ").strip()
                if not choice:
                    print("❌ Wajib pilih video.")
                    continue
                selected_idx = int(choice) - 1
                if 0 <= selected_idx < len(cs):
                    args.videos = [str(cs[selected_idx])]
                    break
                print(f"❌ Masukkan angka 1-{len(cs)}")
            except (ValueError, EOFError):
                print("❌ Masukkan angka yang valid.")

    # Interactive mode hanya untuk 1 video
    if args.interactive and len(args.videos) > 1:
        print("⚠️  Interactive mode only supports 1 video at a time.")
        sys.exit(1)

    print("╔══════════════════════════════════════════════╗")
    print("║   MLBB Skill Dataset Collector              ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"   Skill slots: {', '.join(SKILL_SLOTS)}")
    print(f"   Output dir:  {OUTPUT_DIR}")
    print()

    collector = DatasetCollector(detect_hero=False)

    for vpath in args.videos:
        if not os.path.isfile(vpath):
            print(f"⚠️  File not found: {vpath}")
            continue

        vname = Path(vpath).stem
        hero_name = args.hero

        # If no --hero flag, try to pick interactively
        if hero_name is None and not args.no_pick:
            # Auto-detect from filename first
            guess = vname.split("_")[0].lower() if "_" in vname else vname.lower()
            hero_dir = Path(__file__).resolve().parent.parent / "assets" / "heroes"
            if (hero_dir / f"{guess}.png").exists():
                hero_name = guess
                print(f"   ✅ Hero auto-detected from filename: {guess}")
            else:
                hero_name = _pick_hero_from_list(vname)

        if hero_name is None and args.no_pick:
            # Fallback: pakai guess dari filename
            guess = vname.split("_")[0] if "_" in vname else vname
            print(f"   ⚠️  No hero specified, using filename guess: {guess}")
            hero_name = guess

        if hero_name is None:
            print(f"   ⏭️  Skipping {vname} (no hero selected)")
            continue

        collector.collect_from_video(
            vpath,
            interval=args.interval,
            max_frames=args.max_frames,
            interactive=args.interactive,
            hero_override=hero_name,
        )

    collector.print_dataset_stats()


if __name__ == "__main__":
    main()

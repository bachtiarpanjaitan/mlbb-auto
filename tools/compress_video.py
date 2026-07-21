#!/usr/bin/env python3
"""
Video Compressor — compresses videos in the videos/ folder to ~50% smaller.

Usage:
    python3 tools/compress_video.py <filename>          # Compress one file
    python3 tools/compress_video.py <filename1> <filename2> ...  # Multiple
    python3 tools/compress_video.py --all               # Compress all videos
    python3 tools/compress_video.py --check             # Show video sizes

The compressed file is saved as <name>_compressed.mp4 next to the original.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"


def get_video_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def format_size(bytes_: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def get_video_info(path: Path) -> dict:
    """Get basic video info using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    import json
    data = json.loads(result.stdout)
    info = {"size": get_video_size(path)}

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["codec"] = stream.get("codec_name", "?")
            info["resolution"] = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
            info["fps"] = eval(stream.get("r_frame_rate", "0/1")) if "/" in stream.get("r_frame_rate", "") else stream.get("r_frame_rate")
            info["bitrate"] = int(stream.get("bit_rate", 0))
            break

    if not info.get("bitrate") and data.get("format", {}).get("bit_rate"):
        info["bitrate"] = int(data["format"]["bit_rate"])

    duration = data.get("format", {}).get("duration")
    if duration:
        mins, secs = divmod(float(duration), 60)
        info["duration"] = f"{int(mins):02d}:{int(secs):02d}"

    return info


def is_valid_mp4(path: Path) -> bool:
    """Check if a video file is valid (not truncated)."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", str(path)],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def persen_to_crf(persen: int) -> int:
    """Konversi persentase kompresi ke CRF H.265.

    Semakin tinggi persen → semakin kecil file → CRF lebih besar.
    """
    mapping = {
        10: 18, 20: 22, 30: 24, 40: 26,
        50: 28, 60: 30, 70: 33,
        80: 36, 90: 40,
    }
    # Find nearest mapping
    if persen in mapping:
        return mapping[persen]
    # Interpolasi linear
    keys = sorted(mapping.keys())
    for i, k in enumerate(keys[:-1]):
        if keys[i] <= persen <= keys[i + 1]:
            ratio = (persen - keys[i]) / (keys[i + 1] - keys[i])
            return int(mapping[keys[i]] + (mapping[keys[i + 1]] - mapping[keys[i]]) * ratio)
    return mapping[keys[-1]] if persen > keys[-1] else mapping[keys[0]]


def compress_video(
    input_path: Path,
    crf: int = 28,
    persen: int = 50,
    preset: str = "medium",
) -> Path:
    """Compress a video using H.265 (HEVC) with CRF targeting ~50% reduction.

    Args:
        input_path: Path to the video file.
        target_reduction: Target size reduction (0.50 = 50% smaller).
        preset: x265 preset (medium, fast, slow, veryslow, etc.).

    Returns:
        Path to the compressed video file.
    """
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    # Build output filename
    stem = input_path.stem
    # Remove _compressed suffix if already compressed
    if stem.endswith("_compressed"):
        stem = stem.replace("_compressed", "")

    output_path = input_path.with_name(f"{stem}_compressed.mp4")

    if output_path.exists():
        if is_valid_mp4(output_path):
            print(f"⚠️  Sudah terkompres: {output_path.name} ({format_size(get_video_size(output_path))})")
            return output_path
        else:
            print(f"⚠️  File sebelumnya rusak/tidak lengkap, kompres ulang...")
            output_path.unlink()

    original_size = get_video_size(input_path)

    print(f"\n📦 Original:  {input_path.name}")
    print(f"   Size:      {format_size(original_size)}")

    info = get_video_info(input_path)
    if info.get("resolution"):
        print(f"   Resolution: {info['resolution']}")
    if info.get("duration"):
        print(f"   Duration:   {info['duration']}")

    print(f"   Target:     ~{persen}% lebih kecil (CRF {crf})")
    print(f"   Encoding:   H.265 (HEVC), preset {preset}")
    print()

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vcodec", "libx265",
        "-crf", str(crf),
        "-preset", preset,
        "-tag:v", "hvc1",  # Better compatibility on macOS
        "-pix_fmt", "yuv420p",
        # Remove audio (game replay — no need for sound)
        "-an",
        # Progress output
        "-y",
        str(output_path),
    ]

    print("⏳ Compressing...")
    subprocess.run(cmd, check=True)

    compressed_size = get_video_size(output_path)
    reduction = 1 - (compressed_size / original_size)

    print(f"\n✅ Compressed: {output_path.name}")
    print(f"   Size:       {format_size(original_size)} → {format_size(compressed_size)}")
    print(f"   Reduction:  {reduction:.1%}")

    return output_path


def list_videos() -> list[Path]:
    """List all video files in the videos directory."""
    if not VIDEOS_DIR.exists():
        print(f"❌ Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)

    extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    videos = sorted([p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in extensions])
    return videos


def show_sizes():
    """Show sizes of all videos."""
    videos = list_videos()
    if not videos:
        print("📂 No videos found in videos/")
        return

    print(f"\n{'Name':30s} {'Size':>10s} {'Compressed':>12s} {'Reduction':>10s}")
    print("-" * 62)

    for v in videos:
        size = get_video_size(v)
        compressed = v.with_name(f"{v.stem}_compressed.mp4")
        if compressed.exists():
            if is_valid_mp4(compressed):
                c_size = get_video_size(compressed)
                reduction = 1 - (c_size / size)
                print(f"{v.name:30s} {format_size(size):>10s} {format_size(c_size):>12s} {reduction:>9.0%}")
            else:
                print(f"{v.name:30s} {format_size(size):>10s} {'⚠️  RUSAK':>12s} {'—':>10s}")
        else:
            print(f"{v.name:30s} {format_size(size):>10s} {'—':>12s} {'—':>10s}")


def confirm_videos(video_paths: list[Path], persen: int) -> bool:
    """Show video info and ask for confirmation before compressing."""
    total_original = sum(get_video_size(v) for v in video_paths)

    print(f"\n{'Name':30s} {'Size':>10s} {'Duration':>10s} {'Resolution':>14s}")
    print("-" * 66)

    for v in video_paths:
        info = get_video_info(v)
        size = format_size(get_video_size(v))
        dur = info.get("duration", "?:??")
        res = info.get("resolution", "?")
        print(f"{v.name:30s} {size:>10s} {dur:>10s} {res:>14s}")

    print(f"\n📦 Total: {format_size(total_original)} → ~{format_size(int(total_original * (100 - persen) / 100))} (target {persen}% lebih kecil)")
    print(f"   Codec: H.265 (HEVC) | Audio: dihapus")
    print()

    while True:
        answer = input("❓ Kompres video di atas? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        print("   Masukkan 'y' atau 'n'.")


def main():
    parser = argparse.ArgumentParser(
        description="Kompres video replay MLBB — atur sendiri persentase kompresinya",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Video filename(s) from the videos/ folder",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compress all videos in the videos/ folder",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Show video sizes without compressing",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--persen",
        type=int,
        default=50,
        choices=range(10, 100),
        metavar="10-90",
        help="Target kompresi dalam persen (default: 50, makin besar makin kecil)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Langsung pakai CRF H.265 (0-51, lower=better quality). "
             "Mengabaikan --persen jika diisi.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="x265 encoding preset (default: medium, makin lambat makin kecil)",
    )

    args = parser.parse_args()

    if args.check:
        show_sizes()
        return

    # Collect video paths
    video_paths = []

    if args.all:
        video_paths = list_videos()
        if not video_paths:
            print("📂 No videos found in videos/")
            return
    elif args.filenames:
        for name in args.filenames:
            video_path = VIDEOS_DIR / name
            if not video_path.exists():
                if not name.endswith(".mp4"):
                    video_path = VIDEOS_DIR / f"{name}.mp4"
            if not video_path.exists():
                print(f"❌ Video not found: {name}")
                continue
            video_paths.append(video_path)
    else:
        parser.print_help()
        print("\n📂 Videos available:")
        for v in list_videos():
            size = get_video_size(v)
            print(f"   {v.name:30s} {format_size(size):>10s}")
        return

    if not video_paths:
        return

    # Resolve CRF (from --crf or calculate from --persen)
    crf = args.crf if args.crf is not None else persen_to_crf(args.persen)

    # Confirmation
    if not args.yes:
        if not confirm_videos(video_paths, args.persen):
            print("⏹  Dibatalkan.")
            return

    # Compress
    for v in video_paths:
        compress_video(v, crf=crf, persen=args.persen, preset=args.preset)

    # Show final summary
    print("\n" + "=" * 40)
    show_sizes()


if __name__ == "__main__":
    main()

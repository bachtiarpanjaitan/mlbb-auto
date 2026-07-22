#!/usr/bin/env python3
"""
Quick test: Crop blue team portraits from frame & match against hero assets.
Run ini untuk verifikasi bbox & threshold matching.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.matcher import ORBMatcher


HERO_ASSETS = os.path.join(
    os.path.dirname(__file__), "..", "assets", "heroes"
)

# BBox dari layout.yaml (blue_team.portrait_hero_1 ... 5)
PORTRAIT_BBOXES = [
    (257, 968, 60, 60),   # hero_1
    (444, 968, 60, 60),   # hero_2
    (636, 966, 60, 60),   # hero_3
    (823, 968, 60, 60),   # hero_4
    (1011, 967, 60, 60),  # hero_5
]

CONFIDENCE_THRESHOLD = 0.35


def load_templates(matcher: ORBMatcher, assets_path: str):
    """Load all hero templates."""
    count = 0
    for fname in sorted(os.listdir(assets_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(assets_path, fname))
            if img is not None:
                matcher.add_template(name, img)
                count += 1
    print(f"Loaded {count} hero templates")
    return count


def test_frame(frame_path: str):
    """Test detection on a single frame."""
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Cannot read frame: {frame_path}")
        return

    matcher = ORBMatcher().load_from_config()
    load_templates(matcher, HERO_ASSETS)

    print(f"\n=== Testing frame: {frame_path} ===")
    frame_h, frame_w = frame.shape[:2]
    print(f"Frame size: {frame_w}x{frame_h}")

    for i, (x, y, w, h) in enumerate(PORTRAIT_BBOXES, 1):
        # Crop portrait
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            print(f"  Hero {i}: Empty crop at ({x},{y})")
            continue

        # Save crop for inspection
        cv2.imwrite(f"debug_portrait_{i}.png", crop)

        # Match
        result = matcher.match(crop)
        if result and result.success:
            conf = result.confidence
            status = "✅" if conf >= CONFIDENCE_THRESHOLD else "⚠️"
            print(f"  Hero {i}: {result.label} (conf: {conf:.3f}) {status}")
        else:
            print(f"  Hero {i}: NO MATCH")


def test_video(video_path: str, frame_idx: int = 0):
    """Test on video frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read frame {frame_idx}")
        return

    # Save frame for testing
    test_frame_path = "test_frame.jpg"
    cv2.imwrite(test_frame_path, frame)
    print(f"Saved test frame to {test_frame_path}")

    test_frame(test_frame_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test blue team hero detection")
    parser.add_argument("input", help="Video file or frame image")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (for video)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.threshold

    if args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        test_video(args.input, args.frame)
    else:
        test_frame(args.input)
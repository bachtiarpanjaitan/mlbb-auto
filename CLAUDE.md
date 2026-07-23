# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLBB Auto is an automated game-state extraction pipeline for **Mobile Legends: Bang Bang** replays. It uses computer vision (OpenCV), a YOLOv11n model, OCR (Tesseract), and template matching to detect heroes, items, skills, HP, gold, minimap positions, jungle camps, and other game elements from video files.

**Pipeline:** `Video → FrameReader → Crop Region → Detector → StateBuilder → GameState`

## Dataset & Model

The YOLOv11n model detects 11 classes on the minimap:
- `0=blue_hero`, `1=red_hero`
- `2=lord`, `3=turtle`, `4=thunder_fenrir`, `5=molten_fiend`, `6=lithowanderer`
- `7=crab`, `8=lava_golem`, `9=fire_beetle`, `10=horned_lizard`

Labels are stored in YOLO format at `trainings/hero_detector/`. The model auto-deploys to `models/hero_tracker.onnx` after training.

## Key Commands

```bash
# Activate environment
source .venv/bin/activate

# Run full debug vision pipeline (interactive video player with overlays)
python tools/debug_vision.py [video_name.mp4]

# Label minimap dataset (B=blue hero, R=red hero, J=jungle, N=next, S=save+next)
./scripts/label_minimap.sh [-v video.mp4]

# Train YOLOv11n model
./scripts/train.sh

# Run YOLO inference on a video
./scripts/inference.sh [-v video.mp4] [--conf 0.3]

# Generate hero/spell/item/creep databases (run once after cloning)
python crawlings/crawl_heroes.py
python crawlings/crawl_spells.py
python crawlings/crawl_items.py
python crawlings/crawl_creeps.py

# Region editor (draw polygon regions on minimap for region_mapper)
python tools/region_editor.py

# Compress video files
python tools/compress_video.py

# Inspect / clean dataset
python tools/inspect_dataset.py
python tools/clean_dataset.py
```

## Architecture

### `/vision/core/` — Core pipeline engine

- `layout.py` — Loads `layout.yaml` (region definitions with bbox coordinates on a 2400×1080 replay frame). Provides `load()`, `bbox()`, `get_region()`, `enumerate_regions()`.
- `frame_reader.py` — `FrameReader` class wraps `cv2.VideoCapture` with LRU caching and seeking support.
- `cropper.py` — Crops frame regions by dot-path keys (e.g. `crop_region(frame, "hero_panel", "portrait")`).
- `pipeline.py` — `Pipeline` class that registers detector callbacks per region path and runs them frame-by-frame, yielding `FrameResult` objects.
- `state_builder.py` — `StateBuilder` converts `FrameResult` → `GameState` dataclass (hero, minimap, objectives, towers, events).

### `/vision/detectors/` — Individual game state detectors

Each extends `BaseDetector` (base class providing `detect()`, `preprocess()`, `run()` with timing). One subdirectory per detection type:

- `hp/` — Green HP bar percentage via HSV color extraction
- `mana/` — Blue mana/energy bar percentage
- `level/` — Hero level via OCR
- `gold/` — Gold amount via OCR
- `hero/` — Hero portrait template matching
- `items/` — Item slot template matching (6 slots + CDR calculation)
- `skills/` — Skill cooldown state detection (brightness + template matching)
- `timer/` — Match timer OCR (MM:SS format)
- `towers/` — Tower count detection
- `objectives/` — Lord/Turtle respawn timer OCR
- `minimap/` — **Three approaches** for minimap analysis:
  - `yolo_detector.py` — YOLO via ONNX Runtime (primary, `.onnx`) or ultralytics (fallback, `.pt`)
  - `minimap_hero_tracker.py` — Hero-level tracking combining YOLO detection with circular portrait template matching, velocity-gated smoothing, and NMS. Includes fixed jungle camp registry.
  - `minimap.py` — Simple HSV-based minimap detector (legacy fallback)
- `team/` — Blue/Red team roster detection via scoreboard portrait matching (`blue_team.py`, `red_team.py`)

### `/vision/mapper/` — Coordinate systems

- `coordinate_mapper.py` — Converts minimap normalized coordinates (0-1) to game space (10000×10000 units) with lane/landmark/zone lookups
- `region_mapper.py` — Maps pixel coordinates to named regions from `regions.json` (polygons drawn by `region_editor.py`)
- `landmarks.py` — Fixed landmark positions (bases, towers, buffs, objectives) in game coordinates
- `map_generator.py` — Generates structured match map JSON from tracking data

### `/vision/matcher/` — Template matching engines

- `template.py` — `TemplateMatcher` (TM_CCOEFF_NORMED) for pixel-based icon matching
- `orb.py` — `ORBMatcher` for feature-based matching
- `akaze.py` — `AKAZEMatcher` for feature-based matching

### `/vision/ocr/` — Text recognition

- `reader.py` — Tesseract OCR (pytesseract) with hint-specific preprocessing (number/clock/kda/text)

### `/vision/trackers/` — Cross-frame tracking

- `hero_tracker.py` — Tracks hero states across frames with history
- `object_tracker.py` — Generic object/event tracker
- `team_hp_tracker.py` — Background thread reading HP bars for all 10 heroes

### `/vision/overlay/` — Visualization

- `team_roster_overlay.py` — Renders hero roster overlay on frames

### `/tools/` — Standalone utilities

- `debug_vision.py` — **Main interactive debug tool.** Full-featured video player with: status overlay (HP/level/gold/KDA/skills/items/team), minimap tracking visualization, grid, interactive layout editor (drag/resize regions), cooldown tracking, and multiple toggleable overlays. Runs detectors in separate threads.
- `label_minimap.py` — Minimap labeling tool (B=blue, R=red, J=jungle, N=next frame, S=save+next, Q=quit). Supports all 11 classes.
- `train_yolo.py` — YOLOv11n training script (100 epochs, imgsz=352, no hue shift, no mosaic, ONNX export)
- `inference.py` — YOLO detection viewer with confidence/adjustment
- `region_editor.py` — Polygon region editor for minimap → `regions.json`
- `crop_jungle_templates.py` — Extracts jungle creep minimap templates from labeled data
- `relabel_jungle.py`, `clean_dataset.py`, `inspect_dataset.py` — Dataset maintenance

### `/scripts/` — Shell wrappers

- `label_minimap.sh`, `train.sh`, `inference.sh` — One-command wrappers for tools/

### `/crawlings/` — Data crawlers

- `crawl_heroes.py`, `crawl_spells.py`, `crawl_items.py`, `crawl_creeps.py` — Scrape game data from external sources into `assets/databases/*.json`

### `/assets/` — Static data

- `heroes/` — Hero portrait PNG files (keyed by name)
- `items/`, `spells/`, `creeps/`, `creeps_minimap/` — Template images
- `databases/` — JSON files for heroes, items, spells, creeps, regions

### Layout System (`vision/layout.yaml` / `vision/layout_2400×1080.yaml`)

Defines all screen regions by bbox coordinates (x, y, w, h) for 2400×1080 resolution. Hierarchical with region types:
- `template` — For template matching (hero portraits, item/spell icons)
- `ocr` — For text recognition (level, timer, gold, KDA)
- `bar` — For color bar percentage (HP, mana)
- `composite` — Groups nested sub-regions (hero_panel, skills, items, blue/red team panels)

## Key Design Decisions

- **ONNX Runtime preferred** over PyTorch for inference — lighter, no PyTorch dependency, supports CoreML on Apple Silicon
- **Threaded architecture** in `debug_vision.py` — vision detection runs on background threads to maintain playback FPS
- **Hue shift disabled (hsv_h=0)** during training to preserve blue vs red hero color distinction
- **Mosaic augmentation disabled** because hero icons are too small to survive cropping
- **Velocity-gated position smoothing** for minimap tracking — rejects outlier detections, coasts with prediction + damping
- **CV-based fallback** (HSV contour detection + HoughCircles) when YOLO model is unavailable
- **3-frame hysteresis** for skill cooldown state changes to prevent flickering

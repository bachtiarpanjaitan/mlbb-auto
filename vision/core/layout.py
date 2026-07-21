"""
MLBB Vision — Layout Loader
Memuat dan mengakses layout.yaml untuk mendefinisikan region deteksi.
"""

from __future__ import annotations

import os
import yaml
from typing import Any


_LAYOUT_CACHE: dict[str, dict] = {}
_LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "layout.yaml")

# Scale factor for input frame downscaling.
# When set to < 1.0, all bbox coordinates returned by get_region()
# are proportionally scaled.
_SCALE: float = 1.0


def set_scale(s: float):
    """Set global layout scale factor (for input frame downscaling)."""
    global _SCALE
    _SCALE = s


def load(path: str | None = None) -> dict:
    """Load layout YAML (cached)."""
    p = path or _LAYOUT_PATH
    if p not in _LAYOUT_CACHE:
        with open(p) as f:
            _LAYOUT_CACHE[p] = yaml.safe_load(f)
    return _LAYOUT_CACHE[p]


def regions() -> dict[str, dict]:
    """Return all top-level region definitions."""
    return {k: v for k, v in load().items() if k not in ("video", "detectors", "matchers", "preprocessing")}


def get_region(*keys: str) -> dict | None:
    """Traverse layout by dot-path and return region dict."""
    data = load()
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    if isinstance(data, dict) and "bbox" in data and _SCALE != 1.0:
        bx, by, bw, bh = data["bbox"]
        data = dict(data)  # shallow copy — cache tetap aman
        data["bbox"] = [round(bx * _SCALE), round(by * _SCALE), round(bw * _SCALE), round(bh * _SCALE)]
    return data if isinstance(data, dict) else None


def bbox(*keys: str) -> tuple[int, int, int, int] | None:
    """Return [x, y, w, h] tuple for a region, or None."""
    region = get_region(*keys)
    if region and "bbox" in region:
        return tuple(region["bbox"])
    return None


def video_meta() -> dict[str, Any]:
    """Return video metadata (width, height, fps, etc)."""
    return load().get("video", {})


def detectors() -> dict[str, dict]:
    """Return detector configs."""
    return load().get("detectors", {})


def matchers() -> dict[str, dict]:
    """Return matcher configs."""
    return load().get("matchers", {})


def preprocessing() -> dict:
    """Return preprocessing pipeline config."""
    return load().get("preprocessing", {})


def enumerate_regions(prefix: str = "") -> list[tuple[str, dict]]:
    """
    Recursively enumerate all regions with bbox definitions.
    Returns list of (dot_path, region_dict).
    """
    result: list[tuple[str, dict]] = []

    def _walk(node: dict, path: str):
        for key, val in node.items():
            cur = f"{path}.{key}" if path else key
            if isinstance(val, dict):
                if "bbox" in val and _SCALE != 1.0:
                    bx, by, bw, bh = val["bbox"]
                    scaled = dict(val)
                    scaled["bbox"] = [round(bx * _SCALE), round(by * _SCALE),
                                      round(bw * _SCALE), round(bh * _SCALE)]
                    result.append((cur, scaled))
                elif "bbox" in val:
                    result.append((cur, val))
                _walk(val, cur)

    _walk(load(), "")
    return result


def region_type(*keys: str) -> str | None:
    """Return region type (static, ocr, template, bar, composite, ...)."""
    r = get_region(*keys)
    return r.get("type") if r else None


def layout_to_json() -> str:
    """Debug helper: serialize layout to JSON for inspection."""
    import json
    return json.dumps(load(), indent=2)

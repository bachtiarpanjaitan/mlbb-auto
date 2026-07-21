"""
MLBB Vision — Frame Cropper
Memotong region dari frame berdasarkan definisi layout.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from . import layout as layout_mod


def crop(frame: np.ndarray, bbox: Sequence[int]) -> np.ndarray | None:
    """
    Crop a region from the frame.

    Args:
        frame: Full video frame (H, W, C).
        bbox: [x, y, width, height] in pixels.

    Returns:
        Cropped region or None if out of bounds.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = map(int, bbox)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def crop_region(frame: np.ndarray, *keys: str) -> np.ndarray | None:
    """
    Crop region from layout by dot-path keys.

    Example:
        crop_region(frame, "hero_panel", "portrait")
    """
    region = layout_mod.get_region(*keys)
    if region is None or "bbox" not in region:
        return None
    return crop(frame, region["bbox"])


def crop_multi(frame: np.ndarray, region_keys: list[list[str]]) -> dict[str, np.ndarray | None]:
    """
    Crop multiple regions at once.

    Args:
        frame: Full video frame.
        region_keys: List of dot-path key lists.

    Returns:
        dict mapping flat key name -> cropped image.
    """
    results = {}
    for keys in region_keys:
        name = "_".join(keys)
        results[name] = crop_region(frame, *keys)
    return results


def crop_all_of_type(frame: np.ndarray, region_type: str) -> dict[str, np.ndarray]:
    """
    Crop all regions of a given type (e.g. "template", "ocr", "bar").

    Returns:
        dict of dot_path -> cropped image.
    """
    results = {}
    for path, region in layout_mod.enumerate_regions():
        if region.get("type") == region_type:
            img = crop(frame, region["bbox"])
            if img is not None and img.size > 0:
                results[path] = img
    return results

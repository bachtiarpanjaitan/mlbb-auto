"""
ORB Matcher — Feature-based matching menggunakan ORB (Oriented FAST and Rotated BRIEF).

Cocok untuk matching hero portrait, skill icon, dan elemen yang
mungkin mengalami rotasi/scaling kecil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np

from ..core import layout as layout_mod


@dataclass
class FeatureMatchResult:
    """Hasil feature-based matching."""
    success: bool
    label: str | None = None
    confidence: float = 0.0
    keypoints: list[cv2.KeyPoint] | None = None
    matches: list[cv2.DMatch] | None = None
    homography: np.ndarray | None = None       # 3x3 transform matrix
    src_pts: np.ndarray | None = None
    dst_pts: np.ndarray | None = None


class ORBMatcher:
    """
    Feature-based matcher menggunakan ORB + Brute-Force Hamming.

    Args:
        nfeatures: Jumlah fitur yang diekstrak per image.
        min_good_matches: Minimal match point untuk dianggap sukses.
        ratio_threshold: Lowe's ratio test threshold.
    """

    def __init__(
        self,
        nfeatures: int = 500,
        min_good_matches: int = 10,
        ratio_threshold: float = 0.75,
        templates: dict[str, np.ndarray] | None = None,
    ):
        self.nfeatures = nfeatures
        self.min_good_matches = min_good_matches
        self.ratio_threshold = ratio_threshold
        self.templates: dict[str, tuple[np.ndarray, list[cv2.KeyPoint]]] = {}

        self._orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if templates:
            for name, img in templates.items():
                self.add_template(name, img)

    def add_template(self, name: str, image: np.ndarray):
        """Compute and store ORB features for a template."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is not None:
            self.templates[name] = (des, kp)

    def load_from_config(self):
        """Load config from layout.yaml."""
        config = layout_mod.matchers().get("orb", {})
        params = config.get("params", {})
        self.nfeatures = params.get("nfeatures", self.nfeatures)
        self.min_good_matches = config.get("min_good_matches", self.min_good_matches)
        self.ratio_threshold = config.get("ratio_threshold", self.ratio_threshold)
        self._orb = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=params.get("scale_factor", 1.2),
            nlevels=params.get("nlevels", 8),
            edgeThreshold=params.get("edge_threshold", 31),
        )
        return self

    def match(self, region: np.ndarray) -> FeatureMatchResult | None:
        """
        Match region terhadap semua template terdaftar.

        Returns:
            Best FeatureMatchResult atau None.
        """
        if region is None or region.size == 0:
            return None

        gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = self._orb.detectAndCompute(gray, None)
        if des_query is None or len(kp_query) < 3:
            return None

        best: FeatureMatchResult | None = None

        for name, (des_template, kp_template) in self.templates.items():
            if des_template is None:
                continue

            # KNN match with k=2 for Lowe's ratio test
            matches = self._bf.knnMatch(des_query, des_template, k=2)

            # Lowe's ratio test
            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_good_matches:
                continue

            confidence = min(1.0, len(good_matches) / 30.0)

            # Compute homography if enough matches
            homography = None
            src_pts = None
            dst_pts = None
            if len(good_matches) >= 4:
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inlier_ratio = mask.sum() / len(mask)
                    confidence *= float(inlier_ratio)

            if best is None or confidence > best.confidence:
                best = FeatureMatchResult(
                    success=True,
                    label=name,
                    confidence=confidence,
                    keypoints=kp_query,
                    matches=good_matches,
                    homography=homography,
                    src_pts=src_pts,
                    dst_pts=dst_pts,
                )

        return best

    def match_with_threshold(self, region: np.ndarray, threshold: float = 0.4) -> FeatureMatchResult | None:
        """Match dengan confidence threshold yang bisa diatur."""
        result = self.match(region)
        if result and result.confidence >= threshold:
            return result
        return None

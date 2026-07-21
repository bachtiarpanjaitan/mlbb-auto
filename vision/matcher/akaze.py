"""
AKAZE Matcher — Feature-based matching menggunakan AKAZE (Accelerated-KAZE).

Lebih robust terhadap perubahan lighting dibanding ORB.
Cocok untuk matching scene/background element yang pencahayaannya berubah.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from ..core import layout as layout_mod


@dataclass
class AKAZEMatchResult:
    success: bool
    label: str | None = None
    confidence: float = 0.0
    keypoints: list[cv2.KeyPoint] | None = None
    matches: list[cv2.DMatch] | None = None
    inliers: int = 0


class AKAZEMatcher:
    """
    Feature matcher menggunakan AKAZE (non-linear diffusion scaling).

    Args:
        threshold: Detector response threshold.
        min_good_matches: Minimal match points.
        ratio_threshold: Lowe's ratio test threshold.
    """

    def __init__(
        self,
        threshold: float = 0.001,
        min_good_matches: int = 8,
        ratio_threshold: float = 0.8,
        templates: dict[str, np.ndarray] | None = None,
    ):
        self.min_good_matches = min_good_matches
        self.ratio_threshold = ratio_threshold
        self.templates: dict[str, tuple[np.ndarray, list[cv2.KeyPoint]]] = {}

        self._akaze = cv2.AKAZE_create(
            threshold=threshold,
            nOctaves=4,
            nOctaveLayers=4,
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if templates:
            for name, img in templates.items():
                self.add_template(name, img)

    def add_template(self, name: str, image: np.ndarray):
        """Compute and store AKAZE features."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self._akaze.detectAndCompute(gray, None)
        if des is not None:
            self.templates[name] = (des, kp)

    def load_from_config(self):
        """Load config from layout.yaml."""
        config = layout_mod.matchers().get("akaze", {})
        params = config.get("params", {})
        self.min_good_matches = config.get("min_good_matches", self.min_good_matches)
        self.ratio_threshold = config.get("ratio_threshold", self.ratio_threshold)
        self._akaze = cv2.AKAZE_create(
            threshold=params.get("threshold", 0.001),
            nOctaves=params.get("nOctaves", 4),
            nOctaveLayers=params.get("nOctaveLayers", 4),
        )
        return self

    def match(self, region: np.ndarray) -> AKAZEMatchResult | None:
        """Match region terhadap semua template."""
        if region is None or region.size == 0:
            return None

        gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = self._akaze.detectAndCompute(gray, None)
        if des_query is None or len(kp_query) < 3:
            return None

        best: AKAZEMatchResult | None = None

        for name, (des_template, kp_template) in self.templates.items():
            if des_template is None:
                continue

            matches = self._bf.knnMatch(des_query, des_template, k=2)

            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_good_matches:
                continue

            # Compute inlier ratio via homography
            inliers = len(good_matches)
            confidence = min(1.0, inliers / 25.0)

            if len(good_matches) >= 4:
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    confidence *= float(inliers / len(good_matches))

            if best is None or confidence > best.confidence:
                best = AKAZEMatchResult(
                    success=True,
                    label=name,
                    confidence=confidence,
                    keypoints=kp_query,
                    matches=good_matches,
                    inliers=inliers,
                )

        return best

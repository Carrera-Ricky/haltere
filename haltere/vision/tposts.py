"""Baseline detector for the orange T-posts in the synthetic camera scene.

This is intentionally a simple image-processing checkpoint, not the final learned
vision model. It answers one narrow question first: can the framebuffer image be
turned into a stable object observation?
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TPostParams:
    """Thresholds for the controlled orange T-post render."""

    min_area: int = 20
    # The renderer leaves a small anti-aliased gap between the pole and bar.
    # Closing that gap makes one T-post one connected component while the two
    # posts remain far enough apart to stay separate.
    close_px: int = 9
    red_min: int = 70
    red_over_green: float = 1.20
    red_over_blue: float = 1.35


def orange_mask(rgb: np.ndarray, p: TPostParams = TPostParams()) -> np.ndarray:
    """Return a binary mask for orange/red pixels in an RGB uint8 image."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {a.shape}")
    r = a[..., 0].astype(np.float32)
    g = a[..., 1].astype(np.float32)
    b = a[..., 2].astype(np.float32)
    return ((r >= p.red_min)
            & (r >= p.red_over_green * g)
            & (r >= p.red_over_blue * b)).astype(np.uint8)


def detect_tposts(rgb: np.ndarray, p: TPostParams = TPostParams()) -> list[dict]:
    """Detect orange blobs and return image-space T-post observations.

    Each result contains pixel coordinates plus normalized offsets in [-1, 1].
    Distance is deliberately not returned: apparent size becomes distance only
    after camera calibration and a known physical post height are supplied.
    """
    mask = orange_mask(rgb, p)
    if not mask.any():
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p.close_px, p.close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    detections: list[dict] = []
    for label in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < p.min_area:
            continue
        cx, cy = (float(v) for v in centroids[label])
        detections.append({
            "bbox": [x, y, w, h],
            "center_px": [cx, cy],
            "center_x": 2.0 * cx / max(width - 1, 1) - 1.0,
            "center_y": 2.0 * cy / max(height - 1, 1) - 1.0,
            "pixel_area": area,
            "pixel_height": h,
        })
    detections.sort(key=lambda d: d["pixel_area"], reverse=True)
    return detections


def detect_image(path: str | Path, params: TPostParams = TPostParams(),
                 overlay: str | Path | None = None) -> list[dict]:
    """Read one image, detect T-posts, and optionally write an annotated copy."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detect_tposts(rgb, params)

    if overlay:
        for i, d in enumerate(detections, start=1):
            x, y, w, h = d["bbox"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"T-post {i}: x={d['center_x']:+.2f}"
            cv2.putText(image, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        out = Path(overlay)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), image):
            raise OSError(f"could not write overlay image: {out}")
    return detections


def observation_from_detections(detections: list[dict]) -> dict:
    """Convert raw detections into the compact observation used by navigation.

    The largest image-space post is treated as the nearest candidate. This is
    only an apparent-size ordering for now; it is not yet a metric distance.
    """
    if not detections:
        return {
            "visible": False,
            "center_x": 0.0,
            "center_y": 0.0,
            "pixel_height": 0.0,
            "pixel_width": 0.0,
            "side": "none",
        }

    nearest = max(detections, key=lambda d: d["pixel_height"])
    center_x = float(nearest["center_x"])
    if center_x < -0.05:
        side = "left"
    elif center_x > 0.05:
        side = "right"
    else:
        side = "center"
    _, _, width, _ = nearest["bbox"]
    return {
        "visible": True,
        "center_x": center_x,
        "center_y": float(nearest["center_y"]),
        "pixel_height": float(nearest["pixel_height"]),
        "pixel_width": float(width),
        "side": side,
    }


def params_as_dict(params: TPostParams) -> dict:
    return asdict(params)

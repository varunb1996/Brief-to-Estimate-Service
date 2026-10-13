import os
import tempfile

import cv2
import numpy as np
from pydantic import BaseModel

# Photos are resized to this so the parameters below behave the same for a phone photo and a thumbnail.
MAX_SIDE = 1200


class VisionResult(BaseModel):
    count_hough: int
    count_contour: int
    agreement: float  # smaller count / larger count, 1.0 when they match
    mean_diameter_px: float | None = None
    debug_path: str | None = None


def count_beads(image_bytes: bytes, debug: bool = False) -> VisionResult:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    min_r, max_r = radius_range(gray.shape)
    circles = _hough(gray, min_r, max_r)
    contours, n_c = _contours(gray, min_r, max_r)

    n_h = len(circles)
    return VisionResult(
        count_hough=n_h,
        count_contour=n_c,
        agreement=1.0 if n_h == n_c else round(min(n_h, n_c) / max(n_h, n_c), 3),
        mean_diameter_px=round(float(2 * circles[:, 2].mean()), 1) if n_h else None,
        debug_path=_draw(img, circles, contours, n_c) if debug else None,
    )


def radius_range(shape: tuple[int, ...]) -> tuple[int, int]:
    # Assumes a bead's diameter is between 1/50 and 1/6 of the short side. Seed beads in a
    # wide shot fall below this and are missed; that's a known limit, not something to tune away.
    short = min(shape[:2])
    return max(4, short // 100), short // 12


def _hough(gray: np.ndarray, min_r: int, max_r: int) -> np.ndarray:
    # Parameter values are the ones OpenCV's docs suggest for HOUGH_GRADIENT_ALT.
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=2 * min_r,
                             param1=300, param2=0.9, minRadius=min_r, maxRadius=max_r)
    return np.empty((0, 3)) if found is None else found[0]


def _contours(gray: np.ndarray, min_r: int, max_r: int) -> tuple[list, int]:
    """Returns (blobs kept, bead count). Merged blobs count as area / single-bead area."""
    # The block must be wider than a bead, or a bead's own interior becomes the "local background".
    block = 4 * max_r + 1
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5)
    # Opening with a kernel about the size of the smallest bead cuts the thread joining strung beads.
    k = min_r | 1
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area, max_area = 0.5 * np.pi * min_r**2, 1.5 * np.pi * max_r**2
    kept, round_areas, merged = [], [], []
    for c in found:
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
        if area < min_area or area > 0.25 * gray.size:
            continue
        kept.append(c)
        if area <= max_area and 4 * np.pi * area / perimeter**2 >= 0.7:
            round_areas.append(area)
        else:
            # A closed strand's outer contour encloses the empty middle, so measure only the
            # pixels the threshold marked inside it, not the contour's area.
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [c], -1, 255, -1)
            merged.append(cv2.bitwise_and(mask, binary))
    if not merged:
        return kept, len(round_areas)

    if round_areas:
        bead_area = float(np.median(round_areas))
    else:
        # No clean single bead to calibrate from (e.g. a whole strand merged into one ring).
        # The widest inscribed circle of a chain of touching beads is about one bead.
        radius = np.median([cv2.distanceTransform(m, cv2.DIST_L2, 5).max() for m in merged])
        bead_area = np.pi * float(radius) ** 2
    return kept, len(round_areas) + sum(round(cv2.countNonZero(m) / bead_area) for m in merged)


def _draw(img: np.ndarray, circles: np.ndarray, contours: list, n_contour: int) -> str:
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    for x, y, r in circles:
        cv2.circle(out, (int(x), int(y)), int(r), (0, 200, 0), 2)
    cv2.putText(out, f"hough={len(circles)} (green)  contour={n_contour} (red)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    fd, path = tempfile.mkstemp(prefix="beads_", suffix=".png")
    os.close(fd)
    cv2.imwrite(path, out)
    return path

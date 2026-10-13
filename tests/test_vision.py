from pathlib import Path

import cv2
import numpy as np
import pytest

from app.vision import count_beads


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def dark_discs(positions, r=30, size=(600, 800)):
    img = np.full((*size, 3), 230, np.uint8)
    for x, y in positions:
        cv2.circle(img, (x, y), r, (40, 40, 40), -1, cv2.LINE_AA)
    return png(cv2.GaussianBlur(img, (5, 5), 0))


def test_separated_discs_counted_by_both_methods():
    r = count_beads(dark_discs([(100 + 130 * i, 300) for i in range(5)]))
    assert (r.count_hough, r.count_contour, r.agreement) == (5, 5, 1.0)
    assert r.mean_diameter_px == pytest.approx(60, abs=6)


@pytest.mark.parametrize("positions", [
    [(150 + 60 * i, 300) for i in range(8)],  # touching chain
    [(400 + int(150 * np.cos(a)), 300 + int(150 * np.sin(a)))  # closed ring with an empty middle
     for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)],
], ids=["chain", "ring"])
def test_touching_discs_split_by_area(positions):
    assert count_beads(dark_discs(positions)).count_contour == len(positions)


def test_blank_image_counts_zero():
    r = count_beads(png(np.full((600, 800, 3), 230, np.uint8)))
    assert (r.count_hough, r.count_contour, r.agreement) == (0, 0, 1.0)


def test_debug_image_written_only_when_asked():
    image = dark_discs([(200, 300), (500, 300)])
    assert count_beads(image).debug_path is None
    path = Path(count_beads(image, debug=True).debug_path)
    assert path.exists() and cv2.imread(str(path)) is not None
    path.unlink()


def test_undecodable_bytes_raise():
    with pytest.raises(ValueError):
        count_beads(b"not an image")

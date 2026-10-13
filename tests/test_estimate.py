import pytest

from app.estimate import estimate, from_photo, from_spec
from app.match import load_catalog
from app.schemas import Spec
from app.vision import VisionResult


def bracelet(**fields):
    return Spec(**{"product_type": "bracelet", "bead_size_mm": 8, "length_cm": 18, **fields})


def vision(h, c):
    agreement = 1.0 if h == c else min(h, c) / max(h, c)
    return VisionResult(count_hough=h, count_contour=c, agreement=agreement, mean_diameter_px=50.0)


@pytest.mark.parametrize("item", load_catalog(), ids=lambda i: i.sku)
def test_matches_catalog_counts(item):
    # The catalog was generated with the same allowances, so this checks consistency, not accuracy.
    spec = Spec(product_type=item.product_type, bead_size_mm=item.bead_size_mm, length_cm=item.length_cm,
                string_material=item.string_material, clasp_type=item.clasp_type)
    assert from_spec(spec) == item.bead_count


@pytest.mark.parametrize("fields, expected", [
    ({"string_material": "stretch cord"}, 22),                          # 180 / 8, no clasp
    ({}, 20),                                                           # clasp assumed: 165 / 8
    ({"clasp_type": "lobster clasp"}, 20),
    ({"string_material": "elastic", "extras": ["heart charm"]}, 21),    # 170 / 8
    ({"string_material": "elastic", "bead_shape": "rondelle", "bead_size_mm": 6}, 50),  # 180 / 3.6
    ({"string_material": "elastic", "bead_shape": "heishi", "bead_size_mm": 6}, 100),   # 180 / 1.8
])
def test_shape_clasp_and_extras(fields, expected):
    assert from_spec(bracelet(**fields)) == expected


def test_missing_length_or_size_gives_none():
    assert from_spec(bracelet(length_cm=None)) is None
    assert from_spec(bracelet(bead_size_mm=None)) is None


@pytest.mark.parametrize("v, status, visible", [
    (None, "no_photo", None),
    (vision(20, 22), "ok", 21),
    (vision(14, 37), "failed", None),   # the pearl overcount case
    (vision(0, 10), "failed", None),    # cubes: Hough finds nothing
    (vision(0, 0), "failed", None),     # agreement is 1.0 here, but zero beads is still a failure
])
def test_photo_needs_both_methods_to_agree(v, status, visible):
    p = from_photo(v)
    assert (p.status, p.visible_beads) == (status, visible)


def test_spec_count_wins_and_photo_fills_in_without_length():
    assert estimate(bracelet(string_material="elastic"), vision(20, 20)).model_dump(
        include={"from_spec", "final", "source"}) == {"from_spec": 22, "final": 22, "source": "spec"}
    e = estimate(bracelet(length_cm=None), vision(20, 22))
    assert (e.from_spec, e.final, e.source) == (None, 21, "photo")
    e = estimate(bracelet(length_cm=None), None)
    assert (e.final, e.source) == (None, "none")

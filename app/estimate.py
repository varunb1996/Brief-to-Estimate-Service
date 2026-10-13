import math

from app.match import norm
from app.schemas import Estimate, PhotoEstimate, Spec
from app.vision import VisionResult

# How much of its diameter a bead takes up along the string. Rondelles and heishi are
# discs threaded through the flat face, so they are much thinner than they are wide.
STRING_FRACTION = {"round": 1.0, "cube": 1.0, "seed": 0.75, "rondelle": 0.6, "heishi": 0.3}
CLASP_MM = 15
KEYCHAIN_HARDWARE_MM = 30
EARRING_HOOK_MM = 15
EXTRA_MM = 10  # a guess at the string length a charm or pendant takes up
MIN_AGREEMENT = 0.8  # Hough and contour counts within 20% of each other


def estimate(spec: Spec, vision: VisionResult | None) -> Estimate:
    n = from_spec(spec)
    photo = from_photo(vision)
    if n is not None:
        return Estimate(from_spec=n, from_photo=photo, final=n, source="spec")
    if photo.status == "ok":
        return Estimate(from_photo=photo, final=photo.visible_beads, source="photo")
    return Estimate(from_photo=photo, source="none")


def from_spec(spec: Spec) -> int | None:
    """Beads per piece (per pair for earrings), from finished length and bead size."""
    if not spec.length_cm or not spec.bead_size_mm:
        return None
    bead_mm = spec.bead_size_mm * STRING_FRACTION.get(spec.bead_shape, 1.0)
    usable_mm = spec.length_cm * 10 - EXTRA_MM * len(spec.extras)
    if spec.product_type == "earrings":
        return max(0, 2 * _fit(usable_mm - EARRING_HOOK_MM, bead_mm))
    if spec.product_type == "keychain":
        usable_mm -= KEYCHAIN_HARDWARE_MM
    elif _has_clasp(spec):
        usable_mm -= CLASP_MM
    return max(0, _fit(usable_mm, bead_mm))


def from_photo(vision: VisionResult | None) -> PhotoEstimate:
    if vision is None:
        return PhotoEstimate(status="no_photo")
    h, c = vision.count_hough, vision.count_contour
    counts = f"hough={h}, contour={c}"
    if max(h, c) == 0:
        return PhotoEstimate(status="failed", note=f"no beads found ({counts})")
    if min(h, c) == 0 or vision.agreement < MIN_AGREEMENT:
        return PhotoEstimate(status="failed", mean_diameter_px=vision.mean_diameter_px,
                             note=f"methods disagree ({counts})")
    return PhotoEstimate(status="ok", visible_beads=round((h + c) / 2),
                         mean_diameter_px=vision.mean_diameter_px, note=counts)


def _has_clasp(spec: Spec) -> bool:
    # With no clasp mentioned, assume one unless the string is elastic.
    if spec.clasp_type:
        return norm("clasp", spec.clasp_type) not in ("none", "sliding knot")
    return not (spec.string_material and norm("string", spec.string_material) == "elastic")


def _fit(length_mm: float, bead_mm: float) -> int:
    # The epsilon stops float noise turning an exact fit like 180 / 7.5 into 23.999...
    return math.floor(length_mm / bead_mm + 1e-9)

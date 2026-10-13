import re

from app.estimate import STRING_FRACTION
from app.extract import ExtractionResult
from app.schemas import Estimate, FieldConfidence, Match, ReviewFlag, Spec

PLAUSIBLE = {"bead_size_mm": (2, 20), "length_cm": (5, 120), "bead_count": (1, 300)}
# An earring's length is its drop, so the necklace-scale range flagged every normal pair in the eval.
LENGTH_BY_PRODUCT = {"earrings": (1, 15)}
RANGE_OF = {"bead_size_mm": "bead_size_mm", "length_cm": "length_cm", "stated_bead_count": "bead_count"}
LENGTH_SLACK = 1.2  # stated beads may need up to 20% more string than the stated length
COUNT_TOLERANCE, COUNT_TOLERANCE_BEADS = 0.2, 5
MIN_MATCH_SCORE = 0.6


def field_confidence(brief: str, ext: ExtractionResult) -> dict[str, FieldConfidence]:
    return {field: _rate(field, value, brief, ext)
            for field, value in ext.spec.model_dump().items() if value not in (None, [])}


def review_flags(ext: ExtractionResult, confidence: dict[str, FieldConfidence],
                 matches: list[Match], est: Estimate) -> list[ReviewFlag]:
    flags = []
    if ext.status == "fallback":
        flags.append(_flag("EXTRACTION_FALLBACK", "Couldn't read a spec from the brief. Fill in the details by hand."))
    else:
        for field, c in confidence.items():
            if c.level == "low":
                code = "OUT_OF_RANGE" if out_of_range(field, getattr(ext.spec, field), ext.spec) else "LOW_CONFIDENCE_FIELD"
                flags.append(_flag(code, f"{field}: {c.reason}", field))

    lo, hi = PLAUSIBLE["bead_count"]
    if est.final is not None and not lo <= est.final <= hi:
        flags.append(_flag("OUT_OF_RANGE", f"Estimated {est.final} beads, outside the usual {lo}-{hi}."))

    photo = est.from_photo
    if photo.status == "failed":
        flags.append(_flag("CV_METHODS_DISAGREE", f"Couldn't count beads in the photo: {photo.note}."))
    elif photo.status == "ok" and ext.image_bead_count is not None:
        cv, llm = photo.visible_beads, ext.image_bead_count
        diff = abs(cv - llm)
        if diff > COUNT_TOLERANCE_BEADS or diff > COUNT_TOLERANCE * max(cv, llm):
            flags.append(_flag("CV_LLM_DISAGREE", f"Photo count is {cv} but the model counted {llm}."))

    if not matches or matches[0].score < MIN_MATCH_SCORE:
        best = f"best is {matches[0].sku} at {matches[0].score:.2f}" if matches else "catalog is empty"
        flags.append(_flag("NO_AUTO_MATCH", f"No close catalog match ({best}). Pick a product by hand."))
    return flags


def out_of_range(field: str, value, spec: Spec) -> str | None:
    if field in RANGE_OF:
        lo, hi = PLAUSIBLE[RANGE_OF[field]]
        if field == "length_cm":
            lo, hi = LENGTH_BY_PRODUCT.get(spec.product_type, (lo, hi))
        if not lo <= value <= hi:
            return f"{value:g} is outside the usual {lo}-{hi}"
    if field == "stated_bead_count" and spec.bead_size_mm and spec.length_cm:
        needed_cm = value * spec.bead_size_mm * STRING_FRACTION.get(spec.bead_shape, 1.0) / 10
        if needed_cm > spec.length_cm * LENGTH_SLACK:
            return f"{value} beads of {spec.bead_size_mm:g}mm need about {needed_cm:.0f}cm, longer than {spec.length_cm:g}cm"
    return None


def _rate(field: str, value, brief: str, ext: ExtractionResult) -> FieldConfidence:
    if ext.status == "fallback":
        return FieldConfidence(level="low", reason="placeholder, extraction failed")
    if field in ext.repaired_fields:
        return FieldConfidence(level="low", reason="the model's answer needed cleanup, check it")
    if problem := out_of_range(field, value, ext.spec):
        return FieldConfidence(level="low", reason=problem)
    if _in_brief(value, brief):
        return FieldConfidence(level="high", reason="stated in the brief")
    return FieldConfidence(level="medium", reason="not in the brief word for word, inferred by the model or read from the photo")


def _in_brief(value, brief: str) -> bool:
    if isinstance(value, list):
        return all(_in_brief(v, brief) for v in value)
    if isinstance(value, (int, float)):
        # "8" should match "8mm" but not "18cm"
        return re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?!\d)", brief) is not None
    return str(value).lower() in brief.lower()


def _flag(code: str, message: str, field: str | None = None) -> ReviewFlag:
    return ReviewFlag(code=code, message=message, field=field)

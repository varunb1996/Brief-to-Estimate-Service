import csv
from functools import cache
from pathlib import Path

from app.schemas import CatalogItem, Match, Spec

CATALOG_PATH = Path(__file__).parent.parent / "data" / "inventory.csv"

WEIGHTS = {"product_type": 0.25, "material": 0.20, "color": 0.15, "size": 0.15,
           "length": 0.10, "string": 0.10, "clasp": 0.05}

# Maps brief wording onto the catalog's vocabulary. Anything not listed is compared as-is
# after lowercasing, so a color we never mapped scores 0 instead of matching by accident.
SYNONYMS = {
    "color": {
        "navy": "blue", "royal blue": "blue", "sky blue": "blue", "teal": "turquoise", "aqua": "turquoise",
        "cream": "white", "ivory": "white", "pearl white": "white", "jet": "black", "jet black": "black",
        "rose gold": "gold", "champagne": "gold", "golden": "gold", "crimson": "red", "maroon": "red",
        "lilac": "purple", "lavender": "purple", "violet": "purple", "blush": "pink", "rose": "pink",
        "emerald": "green", "mint": "green", "olive": "green", "tan": "beige", "natural": "beige",
        "chocolate": "brown", "coffee": "brown", "tangerine": "orange",
    },
    "material": {
        "pearl": "freshwater pearl", "pearls": "freshwater pearl", "freshwater pearls": "freshwater pearl",
        "silver": "sterling silver", "925 silver": "sterling silver", "tigers eye": "tiger eye",
        "tiger's eye": "tiger eye", "wooden": "wood", "plastic": "acrylic", "gold-plated brass": "brass",
    },
    "string": {
        "stretch": "elastic", "stretch cord": "elastic", "elastic cord": "elastic", "elastic thread": "elastic",
        "nylon": "nylon thread", "nylon cord": "nylon thread", "silk": "silk thread",
        "wire": "beading wire", "tiger tail": "beading wire", "steel wire": "beading wire",
        "cotton": "cotton cord", "leather": "leather cord",
    },
    "clasp": {
        "lobster clasp": "lobster", "lobster claw": "lobster", "toggle clasp": "toggle",
        "magnetic clasp": "magnetic", "magnet": "magnetic", "no clasp": "none", "stretch": "none",
        "elastic": "none", "french hook": "fish hook", "hook": "fish hook", "ear wire": "fish hook",
        "leverback": "lever back", "lever-back": "lever back", "key ring": "split ring",
        "keyring": "split ring", "adjustable knot": "sliding knot",
    },
}
MATERIAL_GROUPS = [{"rose quartz", "amethyst", "tiger eye"}, {"sterling silver", "brass"}]
COLOR_WORDS = set(SYNONYMS["color"]) | set(SYNONYMS["color"].values()) | {"gray", "grey", "clear", "yellow"}


@cache
def load_catalog() -> tuple[CatalogItem, ...]:
    with CATALOG_PATH.open(newline="") as f:
        return tuple(CatalogItem(**row) for row in csv.DictReader(f))


def match(spec: Spec, catalog=None, k: int = 3) -> list[Match]:
    scored = [_score(spec, item) for item in (catalog or load_catalog())]
    return sorted(scored, key=lambda m: m.score, reverse=True)[:k]


def norm(kind: str, value: str) -> str:
    value = " ".join(value.lower().split())
    value = SYNONYMS[kind].get(value, value)
    if kind == "material":
        value = _strip_material(value)
    return value


def _strip_material(value: str) -> str:
    # The model often copies color and "beads" into the material ("gold brass", "white freshwater
    # pearls"). Try progressively stripped forms and keep the first one the catalog knows.
    known = {item.bead_material for item in load_catalog()}
    if value in known:
        return value
    no_beads = " ".join(w for w in value.split() if w not in ("bead", "beads"))
    no_color = " ".join(w for w in no_beads.split() if w not in COLOR_WORDS)
    candidates = [no_beads, no_color, no_color.removesuffix("s")]
    for c in candidates:
        c = SYNONYMS["material"].get(c, c)
        if c in known:
            return c
    return candidates[-1] or value


def _score(spec: Spec, item: CatalogItem) -> Match:
    parts = {
        "product_type": float(spec.product_type == item.product_type),
        "material": _material(spec.bead_material, item.bead_material),
        "color": float(item.bead_color in {norm("color", c) for c in spec.bead_color}),
        "size": _closeness(spec.bead_size_mm, item.bead_size_mm, tolerance=4),
        "length": _closeness(spec.length_cm, item.length_cm, tolerance=0.25 * (spec.length_cm or 1)),
        "string": _same("string", spec.string_material, item.string_material),
        "clasp": _same("clasp", spec.clasp_type, item.clasp_type),
    }
    breakdown = {name: round(WEIGHTS[name] * value, 3) for name, value in parts.items()}
    return Match(sku=item.sku, score=round(sum(breakdown.values()), 3), breakdown=breakdown)


def _closeness(wanted: float | None, actual: float, tolerance: float) -> float:
    if wanted is None:
        return 0.0
    return max(0.0, 1 - abs(wanted - actual) / tolerance)


def _same(kind: str, wanted: str | None, actual: str) -> float:
    return float(wanted is not None and norm(kind, wanted) == actual)


def _material(wanted: str | None, actual: str) -> float:
    if wanted is None:
        return 0.0
    wanted = norm("material", wanted)
    if wanted == actual:
        return 1.0
    return 0.5 if any(wanted in g and actual in g for g in MATERIAL_GROUPS) else 0.0

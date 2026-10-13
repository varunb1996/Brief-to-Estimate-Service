"""Draws synthetic bead photos with known counts into eval/images/ and writes eval/vision_cases.json.
(eval/cases.json holds the hand-labeled cases and is never written by a script.)

Each image targets one failure mode. They are cleaner than real photos (perfect spheres,
even lighting), so treat the vision numbers as an upper bound on real-world accuracy.
Run: python -m eval.make_images
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

EVAL = Path(__file__).parent
H, W = 900, 1200
rng = np.random.default_rng(7)

BGR = {"black": (35, 35, 35), "purple": (140, 60, 110), "pink": (180, 160, 235), "brown": (40, 80, 140),
       "turquoise": (200, 190, 60), "red": (40, 40, 200), "white": (240, 240, 245), "gold": (60, 170, 215)}


def scatter(n, r_lo, r_hi, spacing, beads=None):
    """Random beads. spacing > 0 keeps a gap, spacing < 0 allows that fraction of overlap."""
    beads = beads if beads is not None else []
    target = len(beads) + n
    while len(beads) < target:
        r = int(rng.integers(r_lo, r_hi + 1))
        x, y = int(rng.integers(r + 30, W - r - 30)), int(rng.integers(r + 30, H - r - 30))
        if all(math.hypot(x - bx, y - by) >= (r + br) * (1 + spacing) for bx, by, br in beads):
            beads.append((x, y, r))
    return beads


def strand(n, r, gap):
    """Beads laid out like a bracelet on a table, slightly foreshortened."""
    big_r = n * (2 * r + gap) / (2 * math.pi)
    return [(int(W / 2 + big_r * math.cos(a)), int(H / 2 + 0.85 * big_r * math.sin(a)), r)
            for a in np.linspace(0, 2 * math.pi, n, endpoint=False)], (int(big_r), int(0.85 * big_r))


def draw_bead(img, cx, cy, r, color, glare):
    x0, x1, y0, y1 = max(cx - r, 0), min(cx + r + 1, W), max(cy - r, 0), min(cy + r + 1, H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = (xx - cx) / r, (yy - cy) / r
    inside = dx**2 + dy**2 <= 1
    z = np.sqrt(np.clip(1 - dx**2 - dy**2, 0, 1))
    light = np.clip(-0.45 * dx - 0.45 * dy + 0.77 * z, 0, 1)  # lit from the upper left
    col = np.array(color, np.float32) * (0.35 + 0.65 * light)[..., None]
    if glare:
        col += 255 * (light**30)[..., None]
    img[y0:y1, x0:x1][inside] = col[inside]


def render(beads, color, *, glare=False, shadow=False, string=None, cubes=False, bg=232, noise=4, blur=1.0):
    img = np.full((H, W, 3), bg, np.float32) * np.linspace(1.0, 0.92, H)[:, None, None]
    if shadow:
        mask = np.zeros((H, W), np.float32)
        for x, y, r in beads:
            cv2.circle(mask, (x + r // 3, y + r // 3), r, 1.0, -1)
        img *= 1 - 0.4 * cv2.GaussianBlur(mask, (0, 0), 8)[..., None]
    if string:
        cv2.ellipse(img, (W // 2, H // 2), string, 0, 0, 360, (70, 70, 70), 3, cv2.LINE_AA)
    for x, y, r in beads:
        if cubes:
            box = cv2.boxPoints(((x, y), (2 * r, 2 * r), float(rng.integers(0, 90)))).astype(np.int32)
            cv2.fillPoly(img, [box], color, cv2.LINE_AA)
        else:
            draw_bead(img, x, y, r, color if color != "mixed" else list(BGR.values())[int(rng.integers(0, 6))], glare)
    img = cv2.GaussianBlur(img, (0, 0), blur) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def cases():
    s3, e3 = strand(18, 26, 10)
    s4, e4 = strand(24, 24, 0)
    s10, e10 = strand(20, 24, 6)
    s13, e13 = strand(60, 6, 2)
    yield "01_clean_separated", render(scatter(12, 28, 28, 0.6), BGR["black"]), 12, ["separated"]
    yield "02_small_separated", render(scatter(30, 13, 13, 0.6), BGR["purple"]), 30, ["separated", "small"]
    yield "03_strand_spaced", render(s3, BGR["pink"], string=e3), 18, ["strand", "string"]
    yield "04_strand_touching", render(s4, BGR["brown"], string=e4), 24, ["strand", "touching", "string"]
    yield "05_overlap", render(scatter(16, 30, 30, -0.3), BGR["turquoise"]), 16, ["overlap"]
    yield "06_glare", render(scatter(14, 30, 30, 0.5), BGR["black"], glare=True), 14, ["glare"]
    yield "07_mixed_sizes", render(scatter(16, 14, 14, 0.5, scatter(8, 40, 40, 0.5)), BGR["red"]), 24, ["mixed sizes"]
    yield "08_low_contrast_pearls", render(scatter(14, 28, 28, 0.5), BGR["white"], glare=True), 14, ["low contrast", "glare"]
    yield "09_shadows", render(scatter(14, 28, 28, 0.6), BGR["gold"], shadow=True), 14, ["shadow"]
    yield "10_noisy_blurry", render(s10, BGR["black"], string=e10, noise=18, blur=3), 20, ["strand", "noise", "blur"]
    yield "11_cubes", render(scatter(10, 25, 25, 0.8), BGR["red"], cubes=True), 10, ["not round"]
    yield "12_dense_pile", render(scatter(40, 22, 26, -0.35), "mixed"), 40, ["overlap", "dense", "mixed colors"]
    yield "13_seed_beads", render(s13, BGR["gold"], string=e13), 60, ["seed beads", "strand", "touching"]


if __name__ == "__main__":
    (EVAL / "images").mkdir(exist_ok=True)
    out = []
    for case_id, img, count, conditions in cases():
        path = f"images/{case_id}.png"
        cv2.imwrite(str(EVAL / path), img)
        out.append({"id": case_id, "image": path, "bead_count": count, "conditions": conditions})
    (EVAL / "vision_cases.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} images and vision_cases.json")

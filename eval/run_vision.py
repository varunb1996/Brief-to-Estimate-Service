"""Prints Hough and contour counts next to ground truth for every image in eval/vision_cases.json.
Run: python -m eval.run_vision [--debug]
"""
import json
import sys
from pathlib import Path

from app.vision import count_beads

EVAL = Path(__file__).parent

cases = json.loads((EVAL / "vision_cases.json").read_text())
debug = "--debug" in sys.argv
errors = {"hough": [], "contour": []}

print(f"{'case':<24}{'truth':>6}{'hough':>7}{'contour':>9}{'agree':>7}  conditions")
for c in cases:
    r = count_beads((EVAL / c["image"]).read_bytes(), debug=debug)
    truth = c["bead_count"]
    errors["hough"].append(abs(r.count_hough - truth) / truth)
    errors["contour"].append(abs(r.count_contour - truth) / truth)
    print(f"{c['id']:<24}{truth:>6}{r.count_hough:>7}{r.count_contour:>9}{r.agreement:>7.2f}  "
          f"{', '.join(c['conditions'])}" + (f"  {r.debug_path}" if debug else ""))

print()
for method, errs in errors.items():
    within = sum(e <= 0.10 for e in errs)
    print(f"{method:<8} mean abs error {100 * sum(errs) / len(errs):5.1f}%   within 10% of truth: {within}/{len(errs)}")

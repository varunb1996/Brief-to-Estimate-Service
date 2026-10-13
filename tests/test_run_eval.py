from itertools import cycle

import cv2
import numpy as np
import pytest

from app.extract import ExtractionResult
from app.schemas import Spec
from eval import run_eval

CASES = [
    {"id": "rq", "brief": "Pink rose quartz bracelet, 8mm, elastic, 18cm",
     "expected_spec": {"product_type": "bracelet", "bead_material": "rose quartz", "bead_size_mm": 8,
                       "bead_color": ["pink"], "length_cm": 18},
     "expected_sku": "BRC-008", "should_flag": False},
    {"id": "photo", "brief": "Black glass keychain like the photo", "image": None,  # set by twelve_beads
     "expected_spec": {"product_type": "keychain"}, "expected_sku": None,
     "expected_bead_count": 12, "should_flag": True},
]
GOOD = Spec(product_type="bracelet", bead_material="Rose Quartz", bead_size_mm=8.3, bead_color=["blush"],
            string_material="elastic", length_cm=18)


@pytest.fixture(autouse=True)
def twelve_beads(tmp_path, monkeypatch):
    # eval/images/ is generated and not in git, so the test draws its own photo.
    img = np.full((600, 900, 3), 230, np.uint8)
    for i in range(12):
        cv2.circle(img, (80 + 140 * (i % 6), 180 + 240 * (i // 6)), 30, (40, 40, 40), -1, cv2.LINE_AA)
    path = tmp_path / "twelve.png"
    cv2.imwrite(str(path), cv2.GaussianBlur(img, (5, 5), 0))
    monkeypatch.setitem(CASES[1], "image", str(path))


@pytest.fixture
def fake_extract(monkeypatch):
    def install(replies):
        it = iter(replies)
        monkeypatch.setattr(run_eval, "extract_spec", lambda brief, image: next(it))
    return install


def test_metrics_on_two_cases_and_two_runs(fake_extract):
    keychain = ExtractionResult(spec=Spec(product_type="keychain"), status="ok", image_bead_count=15)
    drifted = GOOD.model_copy(update={"length_cm": 20})  # second run gets length wrong
    # evaluate() runs every repeat of one case before moving to the next
    fake_extract([ExtractionResult(spec=GOOD, status="ok"), ExtractionResult(spec=drifted, status="ok"),
                  keychain, keychain])
    report = run_eval.evaluate(CASES, runs=2)
    s = report["summary"]

    # normalization and tolerance: "Rose Quartz", "blush" and 8.3mm all count as correct
    assert s["fields"]["bead_material"]["accuracy"] == 1.0
    assert s["fields"]["bead_color"]["accuracy"] == 1.0
    assert s["fields"]["bead_size_mm"]["accuracy"] == 1.0
    assert s["fields"]["length_cm"] == {"n": 1, "accuracy": 0.5, "std": 0.5, "stable": 0.0}
    assert s["matching"] == {"top1_hit": 1.0, "top3_hit": 1.0}  # null SKU case scores low, so it's a hit
    assert s["counts"]["hough"] == {"n": 1, "of": 1, "mae": 0, "mean_pct_error": 0.0}
    assert s["counts"]["vision-LLM"]["mae"] == 3
    assert s["flags"] == {"flag_rate": 0.5, "precision": 1.0, "recall": 1.0}
    assert s["cost_usd_total"] == 0.0
    assert len(report["cases"][0]["runs"]) == 2


def test_rescore_replays_saved_specs_and_excludes_cases_no_model_answered(fake_extract, monkeypatch):
    stretch_case = {**CASES[0], "brief": "Pink rose quartz stretch bracelet, 8mm, 18cm"}
    no_string = GOOD.model_copy(update={"string_material": None})
    fake_extract([ExtractionResult(spec=no_string, status="ok", model="m"),
                  ExtractionResult(spec=Spec(product_type="unknown"), status="fallback")])  # quota hit
    previous = run_eval.evaluate([stretch_case, CASES[1]], runs=1)

    monkeypatch.setattr(run_eval, "extract_spec", lambda *a: pytest.fail("rescore must not call the LLM"))
    report = run_eval.rescore([stretch_case, CASES[1]], previous)
    assert report["excluded_no_model_answer"] == ["photo"]
    assert report["summary_reached_model"]["cases"] == 1
    assert report["cases"][0]["runs"][0]["spec"]["string_material"] == "elastic"  # current code's stretch rule


def test_label_check_catches_unknown_fields_and_counts_without_images():
    problems = run_eval.check_labels([{"id": "x", "expected_spec": {"bead_count": 22}, "expected_bead_count": 22}])
    assert len(problems) == 2


def test_markdown_renders(fake_extract):
    fake_extract(cycle([ExtractionResult(spec=GOOD, status="ok")]))
    md = run_eval.to_markdown(run_eval.evaluate(CASES[:1], runs=1)["summary"])
    assert "| top-1 | 100% |" in md and "| bead_material | 1 | 100% |" in md

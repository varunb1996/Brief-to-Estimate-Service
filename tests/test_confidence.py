from app.confidence import field_confidence, review_flags
from app.estimate import estimate
from app.extract import ExtractionResult
from app.match import match
from app.schemas import Spec
from app.vision import VisionResult

BRIEF = "Pink rose quartz bracelet, 8mm round beads, elastic, 18cm"
CLEAN = Spec(product_type="bracelet", bead_material="rose quartz", bead_size_mm=8, bead_shape="round",
             bead_color=["pink"], string_material="elastic", length_cm=18)


def run(spec=CLEAN, brief=BRIEF, vision=None, **ext_fields):
    ext = ExtractionResult(spec=spec, **{"status": "ok", **ext_fields})
    conf = field_confidence(brief, ext)
    flags = review_flags(ext, conf, match(spec), estimate(spec, vision))
    return conf, [f.code for f in flags]


def vision(h, c):
    return VisionResult(count_hough=h, count_contour=c, agreement=1.0 if h == c else min(h, c) / max(h, c))


def test_clean_case_has_no_flags_and_high_confidence():
    conf, codes = run()
    assert codes == []
    assert {c.level for c in conf.values()} == {"high"}


def test_inferred_values_are_medium():
    conf, codes = run(spec=CLEAN.model_copy(update={"length_cm": 17.78}),
                      brief="Pink rose quartz bracelet, 8mm round beads, stretch, 7 inches")
    assert conf["length_cm"].level == "medium"       # converted from inches
    assert conf["string_material"].level == "medium"  # "stretch" became elastic
    assert conf["bead_size_mm"].level == "high"
    assert codes == []


def test_number_check_does_not_match_inside_other_numbers():
    conf, _ = run(spec=CLEAN.model_copy(update={"bead_size_mm": 8}), brief="rose quartz bracelet, 18cm")
    assert conf["bead_size_mm"].level == "medium"


def test_word_check_needs_whole_words_but_allows_plurals():
    conf, _ = run(spec=CLEAN.model_copy(update={"bead_color": ["red"]}), brief="hand-colored rose quartz bracelet")
    assert conf["bead_color"].level == "medium"
    conf, _ = run(spec=CLEAN.model_copy(update={"bead_material": "freshwater pearl"}), brief="freshwater pearls")
    assert conf["bead_material"].level == "high"


def test_repaired_field_is_low_confidence():
    conf, codes = run(repaired_fields=["bead_size_mm"], status="repaired")
    assert conf["bead_size_mm"].level == "low"
    assert codes == ["LOW_CONFIDENCE_FIELD"]


def test_out_of_range_size():
    conf, codes = run(spec=CLEAN.model_copy(update={"bead_size_mm": 25}), brief=BRIEF.replace("8mm", "25mm"))
    assert conf["bead_size_mm"].level == "low"
    assert "OUT_OF_RANGE" in codes and "LOW_CONFIDENCE_FIELD" not in codes


def test_earring_drop_length_is_not_out_of_range():
    spec = Spec(product_type="earrings", bead_material="freshwater pearl", bead_size_mm=8,
                bead_color=["white"], length_cm=4, clasp_type="fish hook")
    conf, codes = run(spec=spec, brief="8mm white freshwater pearl earrings, fish hook, 4cm drop")
    assert conf["length_cm"].level == "high" and codes == []
    conf, _ = run(spec=spec.model_copy(update={"length_cm": 40}), brief="earrings, 40cm drop")
    assert conf["length_cm"].level == "low"


def test_stated_count_inconsistent_with_length():
    spec = CLEAN.model_copy(update={"stated_bead_count": 40})  # 40 x 8mm = 32cm on an 18cm bracelet
    conf, codes = run(spec=spec, brief=BRIEF + ", 40 beads")
    assert conf["stated_bead_count"].level == "low" and "32cm" in conf["stated_bead_count"].reason
    assert codes == ["OUT_OF_RANGE"]


def test_cv_llm_disagree():
    _, codes = run(vision=vision(20, 20), image_bead_count=30)
    assert codes == ["CV_LLM_DISAGREE"]
    _, codes = run(vision=vision(20, 22), image_bead_count=23)  # 21 vs 23: within tolerance
    assert codes == []


def test_cv_methods_disagree():
    _, codes = run(vision=vision(14, 37), image_bead_count=14)
    assert codes == ["CV_METHODS_DISAGREE"]
    _, codes = run(vision=vision(0, 0))
    assert codes == ["CV_METHODS_DISAGREE"]


def test_no_auto_match_for_vague_spec():
    conf, codes = run(spec=Spec(product_type="bracelet"), brief="a bracelet please")
    assert codes == ["NO_AUTO_MATCH"]


def test_extraction_fallback_does_not_also_flag_each_field():
    conf, codes = run(spec=Spec(product_type="unknown"), brief="???", status="fallback")
    assert conf["product_type"].level == "low"
    assert codes == ["EXTRACTION_FALLBACK", "NO_AUTO_MATCH"]

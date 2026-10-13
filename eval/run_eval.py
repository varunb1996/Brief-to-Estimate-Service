"""Runs the hand-labeled cases in eval/cases.json through the service functions (not HTTP),
prints markdown tables, and writes per-case results to eval/results.json.
Run: python -m eval.run_eval [--runs N]

--rescore skips the LLM: it reloads the extracted specs from eval/results.json, reruns everything
downstream with the current code, and writes eval/results_rescored.json. It also reports the subset
of cases where a model actually answered, since a quota failure is not an extraction error.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from app.confidence import MIN_MATCH_SCORE, field_confidence, review_flags
from app.estimate import estimate
from app.extract import ExtractionResult, extract_spec, fill_from_brief
from app.match import norm
from app.match import match as match_catalog
from app.schemas import Spec
from app.vision import count_beads

EVAL = Path(__file__).parent
TOLERANCE = {"bead_size_mm": 0.5, "length_cm": 1.0}  # other numeric fields must match exactly
NORM_KIND = {"bead_color": "color", "bead_material": "material", "string_material": "string", "clasp_type": "clasp"}


def evaluate(cases: list[dict], runs: int) -> dict:
    def live_run(case, image, vision):
        start = time.perf_counter()
        ext = extract_spec(case["brief"], image)
        return _score_run(case, ext, vision, round((time.perf_counter() - start) * 1000))
    return _evaluate(cases, runs, live_run)


def rescore(cases: list[dict], previous: dict) -> dict:
    """Replays saved extractions through the current downstream code. No LLM calls."""
    runs = len(previous["cases"][0]["runs"])
    saved = {c["id"]: iter(c["runs"]) for c in previous["cases"]}

    def replayed_run(case, image, vision):
        run = next(saved[case["id"]])
        spec = Spec(**run["spec"])
        if run["status"] != "fallback":
            spec = fill_from_brief(spec, case["brief"])
        ext = ExtractionResult(spec=spec, status=run["status"], repaired_fields=run.get("repaired_fields", []),
                               image_bead_count=run["llm_count"], model=run["model"], cost_usd=run["cost_usd"])
        return _score_run(case, ext, vision, run["latency_ms"])

    report = _evaluate(cases, runs, replayed_run)
    reached = [r for r, c in zip(report["cases"], cases) if _reached_model(r, c)]
    reached_ids = {r["id"] for r in reached}
    report["summary_reached_model"] = _summarize(reached, runs)
    # Same subset as scored by the code that produced results.json, for a before/after comparison.
    report["summary_reached_model_before"] = _summarize([c for c in previous["cases"] if c["id"] in reached_ids], runs)
    report["excluded_no_model_answer"] = [r["id"] for r in report["cases"] if r["id"] not in reached_ids]
    return report


def _reached_model(result: dict, case: dict) -> bool:
    # An empty brief is meant to fall back without calling a model, so it stays in.
    return not case["brief"].strip() or any(run["model"] for run in result["runs"])


def _evaluate(cases: list[dict], runs: int, run_case) -> dict:
    results = []
    for case in cases:
        image = _image_bytes(case)
        vision = count_beads(image) if image else None
        results.append({
            "id": case["id"],
            "expected_sku": case.get("expected_sku"),
            "expected_bead_count": case.get("expected_bead_count"),
            "should_flag": case.get("should_flag"),
            "hough": vision.count_hough if vision else None,
            "contour": vision.count_contour if vision else None,
            "runs": [run_case(case, image, vision) for _ in range(runs)],
        })
    return {"summary": _summarize(results, runs), "cases": results}


def _score_run(case: dict, ext: ExtractionResult, vision, latency_ms: int) -> dict:
    est = estimate(ext.spec, vision)
    matches = match_catalog(ext.spec)
    flags = review_flags(ext, field_confidence(case["brief"], ext), matches, est)

    top3 = [m.sku for m in matches]
    expected_sku = case.get("expected_sku")
    if expected_sku is None:
        top1_hit = top3_hit = matches[0].score < MIN_MATCH_SCORE
    else:
        top1_hit, top3_hit = top3[:1] == [expected_sku], expected_sku in top3

    spec = ext.spec.model_dump()
    return {
        "status": ext.status, "model": ext.model, "spec": spec, "repaired_fields": ext.repaired_fields,
        "fields": {f: _field_ok(f, v, spec.get(f)) for f, v in case.get("expected_spec", {}).items()},
        "top3": top3, "top_score": matches[0].score, "top1_hit": top1_hit, "top3_hit": top3_hit,
        "llm_count": ext.image_bead_count, "final_count": est.final,
        "flags": [f.code for f in flags], "flagged": bool(flags),
        "latency_ms": latency_ms, "cost_usd": ext.cost_usd,
    }


def _field_ok(field: str, expected, got) -> bool:
    if expected is None or got is None:
        return expected == got
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(got, (int, float)) and abs(got - expected) <= TOLERANCE.get(field, 0)
    if isinstance(expected, list):
        return isinstance(got, list) and {_n(field, v) for v in expected} == {_n(field, v) for v in got}
    return _n(field, expected) == _n(field, got)


def _n(field: str, value) -> str:
    return norm(NORM_KIND[field], str(value)) if field in NORM_KIND else " ".join(str(value).lower().split())


def _summarize(results: list[dict], runs: int) -> dict:
    per_run = [[r["runs"][i] for r in results] for i in range(runs)]
    fields = sorted({f for r in results for f in r["runs"][0]["fields"]})

    field_stats = {}
    for f in fields:
        scored = [r for r in results if f in r["runs"][0]["fields"]]
        accs = [statistics.mean(r["runs"][i]["fields"][f] for r in scored) for i in range(runs)]
        stable = statistics.mean(len({json.dumps(run["spec"].get(f)) for run in r["runs"]}) == 1 for r in scored)
        field_stats[f] = {"n": len(scored), "accuracy": statistics.mean(accs),
                          "std": statistics.pstdev(accs) if runs > 1 else None, "stable": stable}

    def flag_scores(run_results):
        labeled = [(c["should_flag"], run["flagged"]) for c, run in zip(results, run_results)
                   if c["should_flag"] is not None]
        tp = sum(s and f for s, f in labeled)
        flagged, should = sum(f for _, f in labeled), sum(s for s, _ in labeled)
        return {"flag_rate": flagged / len(labeled) if labeled else None,
                "precision": tp / flagged if flagged else None, "recall": tp / should if should else None}

    counted = [r for r in results if r["expected_bead_count"]]
    counts = {"hough": _count_error([(r["hough"], r["expected_bead_count"]) for r in counted]),
              "contour": _count_error([(r["contour"], r["expected_bead_count"]) for r in counted]),
              "vision-LLM": _count_error([(r["runs"][i]["llm_count"], r["expected_bead_count"])
                                          for r in counted for i in range(runs)])}

    costs = [run["cost_usd"] for rr in per_run for run in rr]
    return {
        "cases": len(results), "runs": runs, "fields": field_stats,
        "matching": {k: statistics.mean(run[k] for rr in per_run for run in rr) for k in ("top1_hit", "top3_hit")},
        "counts": counts,
        "flags": {k: _mean_or_none([flag_scores(rr)[k] for rr in per_run]) for k in ("flag_rate", "precision", "recall")},
        "latency_s_total": sum(run["latency_ms"] for rr in per_run for run in rr) / 1000,
        "latency_s_mean": statistics.mean(run["latency_ms"] for rr in per_run for run in rr) / 1000,
        "cost_usd_total": None if None in costs else sum(costs),
        "statuses": {s: sum(run["status"] == s for rr in per_run for run in rr)
                     for s in sorted({run["status"] for rr in per_run for run in rr})},
    }


def _count_error(pairs: list[tuple]) -> dict:
    scored = [(p, t) for p, t in pairs if p is not None]
    if not scored:
        return {"n": 0, "of": len(pairs), "mae": None, "mean_pct_error": None}
    return {"n": len(scored), "of": len(pairs), "mae": statistics.mean(abs(p - t) for p, t in scored),
            "mean_pct_error": 100 * statistics.mean(abs(p - t) / t for p, t in scored)}


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _image_bytes(case: dict) -> bytes | None:
    if not case.get("image"):
        return None
    for path in (EVAL / "images" / case["image"], EVAL / case["image"]):
        if path.is_file():
            return path.read_bytes()
    raise FileNotFoundError(f"{case['id']}: image {case['image']} not found under eval/images/ or eval/")


def check_labels(cases: list[dict]) -> list[str]:
    known = set(Spec.model_fields)
    problems = []
    for c in cases:
        for f in c.get("expected_spec", {}):
            if f not in known:
                problems.append(f"{c['id']}: expected_spec has '{f}', which is not a spec field")
        if c.get("expected_bead_count") and not c.get("image"):
            problems.append(f"{c['id']}: expected_bead_count without an image")
    return problems


def to_markdown(s: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"
    num = lambda v, fmt=".1f": "n/a" if v is None else format(v, fmt)
    out = [f"Cases: {s['cases']}, runs per case: {s['runs']}, extraction statuses: {s['statuses']}", "",
           "| field | n | accuracy | std across runs | same value every run |", "|---|---|---|---|---|"]
    for f, v in s["fields"].items():
        std = "n/a" if v["std"] is None else f"{100 * v['std']:.1f} pts"
        out.append(f"| {f} | {v['n']} | {pct(v['accuracy'])} | {std} | {pct(v['stable']) if s['runs'] > 1 else 'n/a'} |")
    out += ["", "| matching | hit rate |", "|---|---|",
            f"| top-1 | {pct(s['matching']['top1_hit'])} |", f"| top-3 | {pct(s['matching']['top3_hit'])} |",
            "", "| bead count | scored | MAE | mean % error |", "|---|---|---|---|"]
    for m, v in s["counts"].items():
        out.append(f"| {m} | {v['n']}/{v['of']} | {num(v['mae'])} | {num(v['mean_pct_error'])}% |")
    f = s["flags"]
    cost = "n/a (unpriced model)" if s["cost_usd_total"] is None else f"${s['cost_usd_total']:.4f}"
    out += ["", "| flag rate | precision | recall |", "|---|---|---|",
            f"| {pct(f['flag_rate'])} | {pct(f['precision'])} | {pct(f['recall'])} |",
            "", "| total latency | mean per run | total cost |", "|---|---|---|",
            f"| {s['latency_s_total']:.1f}s | {s['latency_s_mean']:.1f}s | {cost} |"]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="repeat extraction N times per case")
    parser.add_argument("--rescore", action="store_true", help="replay eval/results.json without calling the LLM")
    args = parser.parse_args()
    load_dotenv(EVAL.parent / ".env")

    cases = json.loads((EVAL / "cases.json").read_text())
    for problem in check_labels(cases):
        print(f"label check: {problem}")
    if args.rescore:
        report = rescore(cases, json.loads((EVAL / "results.json").read_text()))
        (EVAL / "results_rescored.json").write_text(json.dumps(report, indent=2))
        print("## All cases, current code\n\n" + to_markdown(report["summary"]))
        print(f"\n## Cases where a model answered (excluded: {', '.join(report['excluded_no_model_answer'])})")
        print("\n### Before (code that produced results.json)\n\n" + to_markdown(report["summary_reached_model_before"]))
        print("\n### After (current code)\n\n" + to_markdown(report["summary_reached_model"]))
    else:
        report = evaluate(cases, args.runs)
        (EVAL / "results.json").write_text(json.dumps(report, indent=2))
        print(to_markdown(report["summary"]))

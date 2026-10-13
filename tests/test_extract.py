import httpx
import pytest

from app import extract
from app.schemas import Spec

VALID = '{"product_type": "bracelet", "bead_material": "rose quartz", "bead_size_mm": 8, "bead_color": ["pink"], "length_cm": 18}'


@pytest.fixture(autouse=True)
def models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "primary")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "backup")


@pytest.fixture
def llm(monkeypatch):
    """Replaces the HTTP call. Queue replies (str) or exceptions; read back the calls made."""
    replies, calls = [], []

    def fake_call(model, prompt, image):
        calls.append((model, prompt))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 100, 50

    monkeypatch.setattr(extract, "_call", fake_call)
    return replies, calls


def test_valid_json(llm):
    replies, calls = llm
    replies.append(VALID)
    r = extract.extract_spec("8mm rose quartz bracelet, 18cm", None)
    assert r.status == "ok" and r.model == "primary"
    assert r.spec.bead_size_mm == 8.0 and r.spec.bead_color == ["pink"]
    assert len(calls) == 1


def test_markdown_fences_are_repaired(llm):
    replies, _ = llm
    replies.append(f"Here you go:\n```json\n{VALID}\n```")
    r = extract.extract_spec("8mm rose quartz bracelet", None)
    assert r.status == "repaired" and r.spec.bead_material == "rose quartz"


def test_unit_strings_and_unknown_keys_are_repaired(llm):
    replies, _ = llm
    replies.append('{"product_type": "necklace", "bead_size_mm": "8mm", "length_cm": "18 inches", '
                   '"bead_color": "white", "vibe": "boho"}')
    r = extract.extract_spec("8mm white necklace, 18 inches", None)
    assert r.status == "repaired"
    assert r.spec.bead_size_mm == 8.0 and r.spec.length_cm == pytest.approx(45.72)
    assert r.spec.bead_color == ["white"]
    assert set(r.repaired_fields) == {"bead_size_mm", "length_cm", "bead_color"}


def test_truncated_json_triggers_retry(llm):
    replies, calls = llm
    replies.extend([VALID[:40], VALID])
    r = extract.extract_spec("8mm rose quartz bracelet", None)
    assert r.status == "retried"
    assert "not valid" in calls[1][1]


def test_wrong_type_retries_with_validation_error(llm):
    replies, calls = llm
    replies.extend(['{"product_type": "bracelet", "bead_size_mm": "eight"}', VALID])
    r = extract.extract_spec("eight millimetre beads", None)
    assert r.status == "retried" and r.spec.bead_size_mm == 8.0
    assert "bead_size_mm" in calls[1][1]


@pytest.mark.parametrize("brief, string", [
    ("pink rose quartz stretch bracelet, 8mm", "elastic"),   # model left string empty
    ("rose quartz bracelet on silk, 8mm", None),             # no "stretch": nothing filled
])
def test_stretch_in_brief_fills_missing_string(llm, brief, string):
    replies, _ = llm
    replies.append('{"product_type": "bracelet", "bead_size_mm": 8}')
    assert extract.extract_spec(brief, None).spec.string_material == string


def test_stretch_rule_does_not_override_a_string_the_model_found(llm):
    replies, _ = llm
    replies.append('{"product_type": "bracelet", "string_material": "silk thread"}')
    assert extract.extract_spec("stretch bracelet on silk", None).spec.string_material == "silk thread"


def test_empty_brief_skips_llm(llm):
    _, calls = llm
    r = extract.extract_spec("   ", b"\x89PNG fake")
    assert r.status == "fallback" and r.spec.product_type == "unknown"
    assert calls == []


def test_primary_error_uses_fallback_model(llm):
    replies, calls = llm
    replies.extend([httpx.ConnectTimeout("slow"), VALID])
    r = extract.extract_spec("8mm rose quartz bracelet", None)
    assert r.status == "fallback_model" and r.model == "backup"
    assert [m for m, _ in calls] == ["primary", "backup"]


def test_total_failure_returns_unknown_spec(llm):
    replies, _ = llm
    replies.extend([httpx.ConnectError("down"), httpx.ConnectError("down")])
    r = extract.extract_spec("8mm rose quartz bracelet", None)
    assert r.status == "fallback" and r.model is None
    assert r.spec == Spec(product_type="unknown")

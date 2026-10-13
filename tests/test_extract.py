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


def test_cm_value_mentioning_in_is_not_converted(llm):
    replies, _ = llm
    replies.append('{"product_type": "necklace", "length_cm": "45 cm in length"}')
    assert extract.extract_spec("necklace, 45 cm", None).spec.length_cm == 45.0


@pytest.mark.parametrize("provider, url, reply, text", [
    ("openai", "https://api.example.test/v1/chat/completions",
     {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}, "{}"),
    ("anthropic", "https://api.example.test/v1/messages",
     {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "{}"}],
      "usage": {"input_tokens": 7, "output_tokens": 3}}, "{}"),
])
def test_call_request_and_response_shape(monkeypatch, provider, url, reply, text):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    sent = {}

    def fake_post(u, headers, json, timeout):
        sent.update(url=u, headers=headers, body=json)
        return httpx.Response(200, json=reply, request=httpx.Request("POST", u))

    monkeypatch.setattr(extract.httpx, "post", fake_post)
    assert extract._call("some-model", "PROMPT", b"\x89PNG fake") == (text, 7, 3)
    assert sent["url"] == url and sent["body"]["model"] == "some-model"
    content = sent["body"]["messages"][0]["content"]
    image_block = "image_url" if provider == "openai" else "image"
    assert {block["type"] for block in content} == {"text", image_block}
    if provider == "anthropic":
        assert sent["headers"]["x-api-key"] == "test-key" and "temperature" not in sent["body"]
    else:
        assert sent["headers"]["Authorization"] == "Bearer test-key" and sent["body"]["temperature"] == 0


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

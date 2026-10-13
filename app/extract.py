import base64
import json
import logging
import os
import re
import time

import httpx
from pydantic import BaseModel

from app.schemas import ExtractionStatus, Spec

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Models missing here log cost_usd=None.
PRICES = {
    "dots-studio/dots-3-note-preview:free": (0.0, 0.0),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": (0.0, 0.0),
}
DEFAULT_BASE_URL = {"anthropic": "https://api.anthropic.com/v1", "openai": "https://api.openai.com/v1"}
TIMEOUT_S = 60
NUMERIC_FIELDS = {"bead_size_mm": float, "length_cm": float, "stated_bead_count": int, "quantity": int}
LIST_FIELDS = ("bead_color", "extras")

PROMPT = """Extract a spec from this sales brief for a custom beaded jewelry piece.
Reply with a single JSON object and nothing else. Its keys follow this JSON schema:
{schema}
Add one more key, "image_bead_count": the number of beads you can count in the attached photo, or null if there is no photo.

Use null for anything the brief does not say. length_cm is the finished length of the piece; convert inches to cm.
If the brief gives a wrist or ankle measurement instead of a finished length, length_cm is that measurement plus 2 cm.
If the product type is not clear, use "unknown".

Brief:
{brief}"""


class ExtractionResult(BaseModel):
    spec: Spec
    status: ExtractionStatus
    repaired_fields: list[str] = []
    image_bead_count: int | None = None
    model: str | None = None
    cost_usd: float | None = 0.0  # None when a model that answered has no entry in PRICES


def extract_spec(brief: str, image_bytes: bytes | None) -> ExtractionResult:
    usage = {"cost_usd": 0.0}
    if not brief.strip():
        return _fallback(usage)

    prompt = PROMPT.format(schema=json.dumps(Spec.model_json_schema()), brief=brief)
    text, model = _complete(prompt, image_bytes, usage)
    if text is None:
        return _fallback(usage)

    try:
        spec, count, fixed, repaired = _parse(text)
        status = "repaired" if repaired else "ok"
    except ValueError as e:
        retry = f"{prompt}\n\nYour previous reply was not valid:\n{e}\nReply again with corrected JSON only."
        text, model = _complete(retry, image_bytes, usage)
        if text is None:
            return _fallback(usage)
        try:
            spec, count, fixed, _ = _parse(text)
        except ValueError as e:
            log.warning("extraction failed after retry: %s", e)
            return _fallback(usage)
        status = "retried"

    if model != os.environ.get("LLM_MODEL"):
        status = "fallback_model"
    return ExtractionResult(spec=fill_from_brief(spec, brief), status=status, repaired_fields=fixed,
                            image_bead_count=count, model=model, **usage)


def fill_from_brief(spec: Spec, brief: str) -> Spec:
    # In the eval the model read "stretch bracelet" as a style, not the string, in 9 of 9 runs.
    # A missing string makes the estimate assume a clasp and come out two beads short.
    if spec.string_material is None and re.search(r"\bstretch", brief, re.IGNORECASE):
        return spec.model_copy(update={"string_material": "elastic"})
    return spec


def _fallback(usage: dict) -> ExtractionResult:
    return ExtractionResult(spec=Spec(product_type="unknown"), status="fallback", **usage)


def _complete(prompt: str, image: bytes | None, usage: dict) -> tuple[str | None, str | None]:
    for model in (os.environ.get("LLM_MODEL"), os.environ.get("LLM_FALLBACK_MODEL")):
        if not model:
            continue
        start = time.perf_counter()
        try:
            text, tokens_in, tokens_out = _call(model, prompt, image)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            log.warning("llm call failed model=%s error=%r", model, e)
            continue
        price = PRICES.get(model)
        cost = (tokens_in * price[0] + tokens_out * price[1]) / 1e6 if price else None
        log.info("llm model=%s latency_ms=%d input_tokens=%d output_tokens=%d cost_usd=%s",
                 model, (time.perf_counter() - start) * 1000, tokens_in, tokens_out, cost)
        usage["cost_usd"] = None if cost is None or usage["cost_usd"] is None else usage["cost_usd"] + cost
        return text, model
    return None, None


def _call(model: str, prompt: str, image: bytes | None) -> tuple[str, int, int]:
    provider = os.environ.get("LLM_PROVIDER", "openai")
    base = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL[provider]
    key = os.environ["LLM_API_KEY"]
    content = [{"type": "text", "text": prompt}]
    if image:
        media_type = "image/png" if image.startswith(b"\x89PNG") else "image/jpeg"
        data = base64.b64encode(image).decode()

    if provider == "anthropic":
        if image:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        r = httpx.post(f"{base}/messages", timeout=TIMEOUT_S,
                       headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                       json={"model": model, "max_tokens": 16000, "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        body = r.json()
        text = "".join(b["text"] for b in body["content"] if b["type"] == "text")
        return text, body["usage"]["input_tokens"], body["usage"]["output_tokens"]

    if image:
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
    r = httpx.post(f"{base}/chat/completions", timeout=TIMEOUT_S,
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}]})
    r.raise_for_status()
    body = r.json()
    usage = body.get("usage") or {}
    return body["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _parse(text: str) -> tuple[Spec, int | None, list[str], bool]:
    """Returns (spec, image_bead_count, repaired_fields, repaired). Raises ValueError if unusable."""
    try:
        return *_validate(json.loads(text)), [], False
    except ValueError:  # pydantic's ValidationError is a ValueError too
        pass
    data, fixed = _repair(text)
    return *_validate(data), fixed, True


def _validate(data) -> tuple[Spec, int | None]:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    data = dict(data)
    count = data.pop("image_bead_count", None)
    if not isinstance(count, int) or isinstance(count, bool):
        count = None
    return Spec.model_validate(data), count


def _repair(text: str) -> tuple[dict, list[str]]:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    data, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")

    allowed = set(Spec.model_fields) | {"image_bead_count"}
    if dropped := [k for k in data if k not in allowed]:
        log.info("dropped unknown keys: %s", dropped)
    data = {k: v for k, v in data.items() if k in allowed}

    fixed = []
    for field, cast in NUMERIC_FIELDS.items():
        value = data.get(field)
        if isinstance(value, str) and (m := re.search(r"\d+(?:\.\d+)?", value)):
            number = float(m.group())
            # "7 inches" must not silently become 7 cm, and "18 cm in length" must not become inches
            if field == "length_cm" and "cm" not in value and re.search(r'\d\s*(in|inch|inches)\b|\d\s*"', value):
                number *= 2.54
            data[field] = cast(number)
            fixed.append(field)
    for field in LIST_FIELDS:
        if isinstance(data.get(field), str):
            data[field] = [data[field]]
            fixed.append(field)
    return data, fixed


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    image = Path(sys.argv[2]).read_bytes() if len(sys.argv) > 2 else None
    print(extract_spec(sys.argv[1], image).model_dump_json(indent=2))

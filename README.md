# bead-quote

A FastAPI service that reads a sales brief for a custom beaded jewelry piece, plus an optional reference photo, and returns a structured spec, the three closest products in a 40-item catalog, a bead count, and the reasons a sales rep should check the result before quoting. Only the spec extraction uses an LLM. Matching, counting, confidence and review flags are plain Python and OpenCV.

Measured results and failure analysis are in [EVALUATION.md](EVALUATION.md).

## Quickstart

Python 3.11 on Windows, from the project folder:

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Put an OpenRouter key after `LLM_API_KEY=` in `.env`. The file is gitignored and nothing else in the repo holds a key.

| task | command |
|---|---|
| run | `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000` |
| run in Docker | `docker compose up --build` |
| test (LLM mocked, no network) | `.venv\Scripts\python.exe -m pytest -q` |
| generate the eval photos (run once, before any eval) | `.venv\Scripts\python.exe -m eval.make_images` |
| eval, calls the LLM | `.venv\Scripts\python.exe -m eval.run_eval --runs 3` |
| eval replay, no LLM calls | `.venv\Scripts\python.exe -m eval.run_eval --rescore` |
| CV eval on 13 synthetic photos | `.venv\Scripts\python.exe -m eval.run_vision --debug` |

The Docker image is python:3.11-slim-bookworm with headless OpenCV (573MB). It runs as uid 10001, has no `.env` inside, and gets its configuration through `env_file`. I built and ran it with Docker 29.1.3 under WSL Ubuntu.

## API example

```
curl.exe -s -X POST http://127.0.0.1:8000/analyze -F "brief=Looking for a pink rose quartz stretch bracelet, 8mm round beads, 18cm."
```

Add `-F "photo=@path\to\photo.png"` to include a photo. In Windows PowerShell, plain `curl` is an alias for `Invoke-WebRequest` and rejects these flags, so use `curl.exe`.

This is the response for eval case c01, with the real model extraction (13.3s) replayed through the current code and trimmed to the first match:

```json
{
  "extraction_status": "ok",
  "spec": {"product_type": "bracelet", "bead_material": "rose quartz", "bead_size_mm": 8.0,
           "bead_shape": "round", "bead_color": ["pink"], "string_material": "elastic", "length_cm": 18.0},
  "confidence": {
    "bead_size_mm": {"level": "high", "reason": "stated in the brief"},
    "string_material": {"level": "medium", "reason": "not in the brief word for word, inferred by the model or read from the photo"}
  },
  "matches": [
    {"sku": "BRC-008", "score": 0.95,
     "breakdown": {"product_type": 0.25, "material": 0.2, "color": 0.15, "size": 0.15,
                   "length": 0.1, "string": 0.1, "clasp": 0.0}}
  ],
  "estimate": {"from_spec": 22, "from_photo": {"status": "no_photo"}, "final": 22, "source": "spec"},
  "review_flags": [],
  "needs_review": false
}
```

`string_material` is medium because the brief says "stretch", not "elastic". Clasp scores 0 because the brief doesn't mention one. The count is 180mm divided by 8mm, with no clasp allowance because the string is elastic. The runner-up, BRC-007, is the 6mm version of the same bracelet and loses 0.075 on size.

| `extraction_status` | meaning |
|---|---|
| `ok` | the first reply validated |
| `repaired` | invalid JSON fixed locally |
| `retried` | a second call, with the validation error, validated |
| `fallback_model` | the primary model failed and the fallback model answered |
| `fallback` | nothing worked, and the spec is `product_type: unknown` |

The six review flags are `LOW_CONFIDENCE_FIELD`, `OUT_OF_RANGE`, `CV_LLM_DISAGREE`, `CV_METHODS_DISAGREE`, `NO_AUTO_MATCH` and `EXTRACTION_FALLBACK`.

## Design decisions

**Matching is deterministic.** A quote has to be reproducible: the same spec gives the same three SKUs with the same scores every time, so a quote can be traced back to why it was made. Pricing also needs an audit trail. Each match carries a breakdown that adds up to its score, so a manager can see that BRC-008 beat BRC-007 by 0.075 on size and nothing else. An LLM ranking products would add run-to-run noise to the product choice. I saw that noise in extraction even at temperature 0: 3 of the 16 cases that reached a model gave different values across three runs (c06 returned 25cm, 27cm, 25cm). The weights live in one dict in `app/match.py`. If the business decides material matters more than color, that's a one-line change, and `--rescore` shows the effect on the eval without any API calls. The cost is the synonym tables, which someone has to maintain. A color with no catalog entry scores 0 instead of being understood.

**Confidence is rule-based.** I don't ask the model how sure it is, because that answer can't be tested and changes between runs. Each level has one rule: `high` if the value appears in the brief (numbers are matched so "8" doesn't hit "18cm"), `medium` if inferred, `low` if repaired, out of a plausible range, or inconsistent (40 beads of 8mm don't fit on an 18cm bracelet). Every level comes with a reason a rep can read, and every flag has a test. The rules are blunt. "stretch" means elastic every time, but it still gets medium, because the word "elastic" isn't in the brief.

**Why bead jewelry.** The bead count follows from geometry: finished length divided by bead size, adjusted for shape, gives a number I can compute and check. Beads are close to circles, which gives classical CV a fair chance. Count and material set the price directly, so an error has a cost you can reason about. And the vocabulary is small (5 bead shapes in the spec, 12 colors and 9 materials in the catalog), which keeps a deterministic matcher workable.

**Classical CV first.** I had no labeled photos to train or fine-tune on. HoughCircles and contour counting run in milliseconds, give the same answer every time, and draw every detection on a debug image. I wrote down the failure modes I expected before tuning anything, and the eval matched them. I run both methods because they fail differently. On low-contrast pearls, Hough found 14 and contours found 37, and that disagreement raises a flag instead of putting a wrong count into the quote. The CV count is also the baseline the vision model's own count is compared against.

## Deploying on AWS

I'd run the existing container on ECS Fargate behind an Application Load Balancer. The service is stateless and a request takes 12 to 20 seconds while the LLM reasons. The 573MB OpenCV image fits a long-running container better than Lambda's packaging and cold starts. The API key would move to Secrets Manager and reach the task as an environment variable, the same way `env_file` works now. For the model I'd switch from the OpenRouter free tier to Claude on Bedrock. Briefs and customer photos would stay inside the AWS account, authentication would go through IAM instead of a key, and the 50-requests-per-day cap that cut my eval short would go away. That needs a small change in `_call` in `app/extract.py`, because Bedrock signs requests differently from the Anthropic API. The per-call log lines (model, latency, tokens, cost) go to CloudWatch. At this volume, a metric filter on cost and on the `fallback` rate is enough alerting. Photos aren't stored today. If the business wants them for review, S3 with a lifecycle rule would hold them.

## Limitations

Everything in the eval is synthetic: the 40-item catalog, the 13 photos (perfect spheres on an even background), and the 25 labeled briefs. The labels were written in the same session that built the system, so they share its assumptions.

A free OpenRouter account allows 50 free-model requests per day, and the eval used them up after 52 calls. Nine cases, including all six photo cases, never reached a model, so photo-case extraction and the vision model's own bead count are unmeasured.

Four known misses aren't fixed. A color the catalog doesn't stock still matches at 0.80 without a flag. Nothing compares the length-based count with the photo count. Missing quote fields such as length aren't flagged. And a prompt line meant for wrist sizes made the model add 2cm to a stated anklet length in 1 of 3 runs.

The heishi and charm allowances in the bead count are guesses, not measurements.

from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.confidence import field_confidence, review_flags
from app.estimate import estimate
from app.extract import extract_spec
from app.match import match
from app.schemas import AnalyzeResponse
from app.vision import count_beads

load_dotenv()
app = FastAPI(title="bead-quote")


@app.get("/health")
def health():
    return {"status": "ok"}


# Plain def, not async: the LLM call blocks, so FastAPI runs this in its thread pool.
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    brief: Annotated[str, Form()],
    photo: Annotated[UploadFile | None, File()] = None,
) -> AnalyzeResponse:
    image = photo.file.read() if photo else None
    vision = None
    if image:
        try:
            vision = count_beads(image)
        except ValueError:
            raise HTTPException(400, "photo could not be read as an image")

    ext = extract_spec(brief, image)
    est = estimate(ext.spec, vision)
    matches = match(ext.spec)
    confidence = field_confidence(brief, ext)
    flags = review_flags(ext, confidence, matches, est)
    return AnalyzeResponse(
        extraction_status=ext.status, spec=ext.spec, confidence=confidence, matches=matches,
        estimate=est, review_flags=flags, needs_review=bool(flags),
    )

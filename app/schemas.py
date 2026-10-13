from typing import Literal

from pydantic import BaseModel, Field

ProductType = Literal["bracelet", "necklace", "anklet", "keychain", "earrings", "unknown"]
ExtractionStatus = Literal["ok", "repaired", "retried", "fallback_model", "fallback"]
BeadShape = Literal["round", "rondelle", "cube", "seed", "heishi"]
Level = Literal["high", "medium", "low"]
SpecField = Literal[
    "product_type", "bead_material", "bead_size_mm", "bead_shape", "bead_color",
    "pattern", "string_material", "length_cm", "stated_bead_count", "clasp_type",
    "quantity", "extras", "notes",
]


class Spec(BaseModel):
    product_type: ProductType
    bead_material: str | None = None
    bead_size_mm: float | None = Field(None, gt=0)
    bead_shape: BeadShape | None = None
    bead_color: list[str] = []
    pattern: str | None = None
    string_material: str | None = None
    # finished length of the piece, not the wrist or neck measurement
    length_cm: float | None = Field(None, gt=0)
    # what the customer asked for; our own count lives in Estimate
    stated_bead_count: int | None = Field(None, gt=0)
    clasp_type: str | None = None
    quantity: int | None = Field(None, gt=0)
    extras: list[str] = []
    notes: str | None = None


class CatalogItem(BaseModel):
    sku: str
    product_type: ProductType
    bead_material: str
    bead_size_mm: float = Field(gt=0)
    bead_color: str
    string_material: str
    length_cm: float = Field(gt=0)
    bead_count: int = Field(gt=0)
    clasp_type: str
    unit_price_inr: int = Field(gt=0)


class FieldConfidence(BaseModel):
    level: Level
    reason: str


class Match(BaseModel):
    sku: str
    score: float = Field(ge=0, le=1)
    breakdown: dict[str, float]


class PhotoEstimate(BaseModel):
    status: Literal["ok", "no_photo", "failed"]
    visible_beads: int | None = None
    mean_diameter_px: float | None = None
    note: str | None = None


class Estimate(BaseModel):
    from_spec: int | None = None
    from_photo: PhotoEstimate
    final: int | None = None
    source: Literal["spec", "photo", "none"]


class ReviewFlag(BaseModel):
    code: Literal[
        "LOW_CONFIDENCE_FIELD", "OUT_OF_RANGE", "CV_LLM_DISAGREE",
        "CV_METHODS_DISAGREE", "NO_AUTO_MATCH", "EXTRACTION_FALLBACK",
    ]
    field: SpecField | None = None
    message: str


class AnalyzeResponse(BaseModel):
    extraction_status: ExtractionStatus
    spec: Spec
    confidence: dict[SpecField, FieldConfidence]
    matches: list[Match] = Field(max_length=3)
    estimate: Estimate
    review_flags: list[ReviewFlag]
    needs_review: bool

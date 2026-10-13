import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.extract import ExtractionResult
from app.schemas import Spec

BRIEF = "Pink rose quartz bracelet, 8mm round beads, elastic, 18cm"
CLEAN = Spec(product_type="bracelet", bead_material="rose quartz", bead_size_mm=8, bead_shape="round",
             bead_color=["pink"], string_material="elastic", length_cm=18)


def test_analyze_end_to_end_with_photo(monkeypatch):
    monkeypatch.setattr(main, "extract_spec", lambda brief, image: ExtractionResult(
        spec=CLEAN, status="ok", image_bead_count=5))
    img = np.full((600, 800, 3), 230, np.uint8)
    for i in range(5):
        cv2.circle(img, (100 + 130 * i, 300), 30, (40, 40, 40), -1, cv2.LINE_AA)
    photo = cv2.imencode(".png", cv2.GaussianBlur(img, (5, 5), 0))[1].tobytes()

    client = TestClient(main.app)
    r = client.post("/analyze", data={"brief": BRIEF}, files={"photo": ("p.png", photo, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["matches"][0]["sku"] == "BRC-008"
    assert body["estimate"]["from_spec"] == 22 and body["estimate"]["from_photo"]["visible_beads"] == 5
    assert body["needs_review"] is False and body["review_flags"] == []


def test_undecodable_photo_is_a_400():
    r = TestClient(main.app).post("/analyze", data={"brief": BRIEF},
                                  files={"photo": ("p.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_health():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}

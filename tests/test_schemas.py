import csv
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from app.schemas import CatalogItem, Spec, SpecField

CATALOG = Path(__file__).parent.parent / "data" / "inventory.csv"


def test_catalog_loads_with_40_unique_skus():
    with CATALOG.open(newline="") as f:
        items = [CatalogItem(**row) for row in csv.DictReader(f)]
    assert len(items) == 40
    assert len({i.sku for i in items}) == 40


def test_spec_field_literal_matches_spec_model():
    # SpecField is typed by hand for the confidence dict keys, so catch drift
    assert set(get_args(SpecField)) == set(Spec.model_fields)


def test_spec_needs_only_product_type():
    spec = Spec(product_type="bracelet")
    assert spec.bead_color == [] and spec.length_cm is None


@pytest.mark.parametrize("bad", [
    {},
    {"product_type": "ring"},
    {"product_type": "bracelet", "bead_shape": "teardrop"},
    {"product_type": "bracelet", "length_cm": -3},
])
def test_spec_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        Spec(**bad)

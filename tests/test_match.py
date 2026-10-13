import pytest

from app.match import load_catalog, match, norm
from app.schemas import Spec


def spec(**fields):
    return Spec(**{"product_type": "bracelet", **fields})


def test_exact_spec_ranks_its_sku_first_and_size_twin_second():
    top = match(spec(bead_material="rose quartz", bead_size_mm=8, bead_color=["pink"],
                     string_material="elastic", length_cm=18, clasp_type="no clasp"))
    assert [m.sku for m in top[:2]] == ["BRC-008", "BRC-007"]
    assert top[0].score == 1.0
    assert top[1].breakdown["size"] == 0.075  # 6mm vs 8mm is half credit


def test_color_twin_ranks_below_exact_color():
    top = match(spec(bead_material="glass", bead_size_mm=8, bead_color=["black"],
                     string_material="elastic", length_cm=18))
    assert [m.sku for m in top[:2]] == ["BRC-001", "BRC-002"]
    assert top[0].score - top[1].score == pytest.approx(0.15)


def test_synonyms_map_brief_words_to_catalog_vocabulary():
    top = match(spec(bead_material="Glass", bead_size_mm=8, bead_color=["navy"],
                     string_material="stretch cord", length_cm=18))
    assert top[0].sku == "BRC-002"
    assert top[0].breakdown["color"] == 0.15 and top[0].breakdown["string"] == 0.10


def test_clasp_wording_is_normalized():
    top = match(Spec(product_type="necklace", bead_material="pearls", bead_size_mm=6,
                     bead_color=["white"], length_cm=45, clasp_type="lobster clasp"))
    assert top[0].sku == "NCK-001"
    assert top[0].breakdown["clasp"] == 0.05 and top[0].breakdown["material"] == 0.20


def test_same_stone_group_gets_half_material_credit():
    top = match(spec(bead_material="tiger's eye", bead_size_mm=8), catalog=[
        item for item in load_catalog() if item.sku in ("BRC-009", "BRC-011")])
    by_sku = {m.sku: m for m in top}
    assert by_sku["BRC-011"].breakdown["material"] == 0.20
    assert by_sku["BRC-009"].breakdown["material"] == 0.10  # amethyst


@pytest.mark.parametrize("said, expected", [
    ("gold brass", "brass"), ("white freshwater pearls", "freshwater pearl"), ("black onyx", "onyx"),
    ("rose quartz beads", "rose quartz"), ("Rose Quartz", "rose quartz"), ("natural wood", "wood"),
    ("pink rose quartz", "rose quartz"),  # "rose" is also a color word; it must not be stripped here
    ("swarovski glass", "glass"), ("lava stone", "lava stone"),
])
def test_material_drops_color_and_bead_words(said, expected):
    assert norm("material", said) == expected


def test_unmapped_color_never_scores():
    for m in match(spec(bead_color=["chartreuse"])):
        assert m.breakdown["color"] == 0


def test_vague_spec_scores_low():
    assert all(m.score == 0 for m in match(Spec(product_type="unknown")))
    assert match(spec())[0].score == 0.25  # product type alone


def test_returns_three_and_breakdown_sums_to_score():
    top = match(spec(bead_material="amethyst", bead_size_mm=10, length_cm=19))
    assert len(top) == 3
    for m in top:
        assert sum(m.breakdown.values()) == pytest.approx(m.score, abs=1e-6)
    assert [m.score for m in top] == sorted((m.score for m in top), reverse=True)

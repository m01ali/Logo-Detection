"""21_test_plan.md — MM0."""

from __future__ import annotations

from pathlib import Path

import pytest

from logo_pipeline.brands import (BrandEntry, BrandListError, build_brand_list, load_brand_list, normalise,
                                   rewrite_hash)


def test_brands_load_hash_stable(brands_small_path: Path):
    bl1 = load_brand_list(brands_small_path)
    bl2 = load_brand_list(brands_small_path)
    assert bl1.content_hash == bl2.content_hash
    assert bl1.content_hash.startswith("sha256:")


def test_alias_collision_rejected():
    with pytest.raises(BrandListError, match="collides"):
        build_brand_list("2026-01-01", [
            BrandEntry(canonical="rai", aliases=("rai uno",)),
            BrandEntry(canonical="rai sport", aliases=("rai uno",)),  # collides with rai's alias
        ])


def test_alias_equal_to_other_canonical_rejected():
    with pytest.raises(BrandListError, match="collides with canonical"):
        build_brand_list("2026-01-01", [
            BrandEntry(canonical="rai", aliases=()),
            BrandEntry(canonical="rai sport", aliases=("rai",)),
        ])


def test_duplicate_canonical_rejected():
    with pytest.raises(BrandListError, match="duplicate canonical"):
        build_brand_list("2026-01-01", [
            BrandEntry(canonical="rai", aliases=()),
            BrandEntry(canonical="rai", aliases=()),
        ])


def test_normalise_accents():
    assert normalise("L'Oréal") == "loreal"
    assert normalise("  Café   Central  ") == "cafe central"
    assert normalise(None) == ""


def test_short_exact_rule(brands_small_path: Path):
    bl = load_brand_list(brands_small_path)
    # "hb" is not "hp": below the short cutoff, only an exact normalised match counts.
    canon, score, mode = bl.map_fuzzy("hb", threshold=90, short_cutoff=4)
    assert canon is None
    assert score == 0
    assert mode == "exact_short"
    canon, score, mode = bl.map_fuzzy("hp", threshold=90, short_cutoff=4)
    assert canon == "hp"
    assert score == 100


def test_map_fuzzy_long():
    bl = build_brand_list("2026-01-01", [BrandEntry(canonical="nurofen", aliases=())])
    canon, score, mode = bl.map_fuzzy("nurofenn", threshold=90, short_cutoff=4)
    assert canon == "nurofen"
    assert score >= 90
    assert mode == "fuzzy"
    # A poor match below threshold returns None but still reports the score.
    canon2, score2, _ = bl.map_fuzzy("zzzzzzzz", threshold=90, short_cutoff=4)
    assert canon2 is None


def test_channel_set(brands_small_path: Path):
    bl = load_brand_list(brands_small_path)
    assert bl.channel_brands == frozenset({"rai", "rai sport"})
    assert bl.is_channel("rai")
    assert not bl.is_channel("hp")


def test_alias_matches_via_map_exact(brands_small_path: Path):
    bl = load_brand_list(brands_small_path)
    assert bl.map_exact("Very Sure") == "verisure"
    assert bl.map_exact("VOLKSWAGEN") == "wolkswagen"
    assert bl.map_exact("nonexistent brand") is None


def test_rewrite_hash_matches_reload(tmp_path: Path, brands_small_path: Path):
    dst = tmp_path / "brands.yaml"
    dst.write_text(brands_small_path.read_text(encoding="utf-8"))
    bl = rewrite_hash(dst)
    reloaded = load_brand_list(dst)
    assert bl.content_hash == reloaded.content_hash
    assert reloaded.hash_matches is True


def test_snapshot_has_321_entries(repo_root: Path):
    snapshot = repo_root / "configs" / "brands" / "roboflow_2026-04-28.yaml"
    if not snapshot.exists():
        pytest.skip("brand snapshot not generated (run scripts/make_brand_snapshot.py)")
    bl = load_brand_list(snapshot)
    assert len(bl.canonical) == 321
    assert len(bl.short_names) == 37  # 06 §6
    assert bl.hash_matches is not False

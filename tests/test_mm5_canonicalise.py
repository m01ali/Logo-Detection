"""MM5 Part A (T37): the deterministic canonicalisation map and its stage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

from logo_pipeline.stages.canonicalise import (CanonicaliseStage, canonicalise, canonicalise_audio_brands)
from helpers import make_ctx  # noqa: E402
from mm5_helpers import BRANDS, make_cfg  # noqa: E402


def canon(s, audio=()):
    return canonicalise(s, BRANDS, make_cfg().canonicalisation, list(audio))


def test_none_and_unknown():
    assert (canon(None).canonical, canon(None).mode, canon(None).score) == (None, None, None)
    c = canon("UNKNOWN")
    assert (c.canonical, c.mode, c.score) == ("UNKNOWN", "unknown", None)


def test_alias_table_comes_first_then_exact():
    c = canon("RAI SPORT HD")
    assert (c.canonical, c.mode, c.score) == ("rai sport", "alias", 100)
    c = canon("Nike")
    assert (c.canonical, c.mode) == ("nike", "exact")
    assert canon("Volkswagen").canonical == "wolkswagen"


def test_fuzzy_onto_canonical_list_with_length_rule():
    c = canon("Verisur")
    assert (c.canonical, c.mode) == ("verisure", "fuzzy") and c.score >= 90
    # A short string only matches exactly: "hb" is not "hp".
    assert canon("hb").mode == "unmatched"
    assert canon("hp").canonical == "hp"


def test_audio_brands_are_canonicalised_first_and_then_targets():
    audio = canonicalise_audio_brands(["very sure", "Acme Corp"], BRANDS, make_cfg().canonicalisation)
    assert audio == ["verisure", "acme corp"]                 # 'very sure' -> verisure before it can be a target
    c = canon("ACME Corp.", audio)
    assert (c.canonical, c.mode, c.score) == ("acme corp", "audio", 100)
    assert canon("Acme Corporation", audio).mode == "audio"     # fuzzy onto the audio target
    assert canon("ACME Corp.").mode == "unmatched"             # not a target without the audio list


def test_list_wins_over_audio_target():
    assert canon("very sure", ["verisure"]).mode == "alias"


def test_unmatched_keeps_normalised_string_and_best_score():
    c = canon("Foo  Bär!!")
    assert (c.canonical, c.mode) == ("foo bar", "unmatched") and isinstance(c.score, int)
    assert canon("FOO BAR").canonical == c.canonical           # repeated spellings group exactly


def test_deterministic():
    assert canon("Rai  Sport HD!") == canon("Rai  Sport HD!")


def test_audio_targets_can_be_switched_off():
    cfg = make_cfg({"canonicalisation": {"audio_brands_as_targets": False}}).canonicalisation
    assert canonicalise("acme corp", BRANDS, cfg, ["acme corp"]).mode == "unmatched"


# -- the stage ---------------------------------------------------------------------

def _stage_ctx(tmp_path, repo_root, hinted, faiss=None, blind=None, audio=None, overrides=None):
    ctx = make_ctx(tmp_path, repo_root, overrides=overrides)
    reps = [{"det_id": i, "frame_idx": 0} for i in range(len(hinted))]
    ctx.write_jsonl("merged_detections.jsonl", reps)
    ctx.write_jsonl("vlm/hinted.jsonl", [{"det_id": i, "parsed": h, "vlm_parse_ok": h is not None}
                                          for i, h in enumerate(hinted)])
    if blind:
        ctx.write_jsonl("vlm/blind.jsonl", [{"det_id": i, "parsed": b} for i, b in enumerate(blind)])
    if faiss:
        ctx.write_jsonl("faiss_hints.jsonl", [{"det_id": i, **f} for i, f in enumerate(faiss)])
    if audio is not None:
        ctx.write_json("audio/audio_hints.json", {"status": "ok", "brands": audio})
    return ctx


def test_stage_writes_derived_fields_and_report(tmp_path, repo_root):
    ctx = _stage_ctx(
        tmp_path, repo_root,
        hinted=[{"is_logo": "logo", "brand": "Rai News 24"}, {"is_logo": "logo", "brand": "Zzyzx Cola"},
                {"is_logo": "not_logo", "brand": None}, {"is_logo": "logo", "brand": "UNKNOWN"}],
        blind=[{"brand": "RAI"}, {"brand": "Other"}, {"brand": None}, {"brand": None}],
        faiss=[{"faiss_top1": "Rai", "faiss_top5_brands": ["Rai", "Nike"]},
               {"faiss_top1": "Nike", "faiss_top5_brands": ["Nike"]}, {}, {}],
        audio=["zzyzx cola", "very sure"])
    CanonicaliseStage().run(ctx)
    rows = {r["det_id"]: r for r in ctx.read_jsonl("canonical.jsonl")}

    r0 = rows[0]
    assert (r0["canonical_brand"], r0["canonical_match_mode"]) == ("rai", "alias")
    assert r0["faiss_top1_brand_canonical"] == "rai" and r0["faiss_agrees_with_vlm"] is True
    assert r0["hinted_brand_in_faiss_hints"] is True and r0["blind_hinted_brand_agree"] is True
    assert r0["brand_from_audio"] is False
    assert r0["audio_brands_canonical"] == ["zzyzx cola", "verisure"]

    r1 = rows[1]
    assert (r1["canonical_brand"], r1["canonical_match_mode"]) == ("zzyzx cola", "audio")
    assert r1["brand_from_audio"] is True and r1["faiss_agrees_with_vlm"] is False
    assert r1["hinted_brand_in_faiss_hints"] is False and r1["blind_hinted_brand_agree"] is False

    assert rows[2]["canonical_brand"] is None and rows[2]["canonical_match_mode"] is None
    assert (rows[3]["canonical_brand"], rows[3]["canonical_match_mode"]) == ("UNKNOWN", "unknown")
    assert rows[3]["brand_from_audio"] is False and rows[3]["hinted_brand_in_faiss_hints"] is False
    assert all(r["brand_list_version"] == ctx.brands.version for r in rows.values())
    assert "canonical_match_mode" in ctx.path("reports/canonicalisation.md").read_text()


def test_stage_is_deterministic(tmp_path, repo_root):
    ctx = _stage_ctx(tmp_path, repo_root, hinted=[{"is_logo": "logo", "brand": "Foo"}], audio=[])
    CanonicaliseStage().run(ctx)
    first = ctx.path("canonical.jsonl").read_text()
    CanonicaliseStage().run(ctx)
    assert ctx.path("canonical.jsonl").read_text().split("\n", 1)[1] == first.split("\n", 1)[1]


def test_llm_merge_is_off_by_default(tmp_path, repo_root):
    def boom(_):
        raise AssertionError("LLM merge must not run when canonicalisation.llm_merge.enabled is false")
    ctx = _stage_ctx(tmp_path, repo_root, hinted=[{"is_logo": "logo", "brand": "Foo Bar"}])
    CanonicaliseStage(llm_merge=boom).run(ctx)
    assert not ctx.path("canonical_llm_merge.json").exists()


def test_llm_merge_only_sees_unmatched_strings_and_is_logged(tmp_path, repo_root):
    seen = []

    def fake(strings):
        seen.append(strings)
        return {s: ("foo bar" if s == "foo baar" else s) for s in strings}, "raw"

    ctx = _stage_ctx(tmp_path, repo_root, overrides={"canonicalisation": {"llm_merge": {"enabled": True}}},
                     hinted=[{"is_logo": "logo", "brand": "Nike"}, {"is_logo": "logo", "brand": "Foo Bar"},
                             {"is_logo": "logo", "brand": "Foo Baar"}])
    CanonicaliseStage(llm_merge=fake).run(ctx)
    assert seen == [["foo baar", "foo bar"]] or seen == [["foo bar", "foo baar"]]
    rows = {r["det_id"]: r for r in ctx.read_jsonl("canonical.jsonl")}
    assert rows[0]["canonical_match_mode"] == "exact"
    assert (rows[2]["canonical_brand"], rows[2]["canonical_match_mode"]) == ("foo bar", "llm_merge")
    assert rows[1]["canonical_match_mode"] == "unmatched"
    saved = json.loads(ctx.path("canonical_llm_merge.json").read_text())
    assert saved["changed"] == {"foo baar": "foo bar"}

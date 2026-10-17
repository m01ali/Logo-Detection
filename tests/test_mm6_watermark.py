"""MM6 (spec docs/12, T47-T50): the overlap measure, the keyed region store and
its validation, the watermark stage, and Rule 5 reading real overlaps."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402
from mm5_helpers import decide, row  # noqa: E402

from logo_pipeline.config import WatermarkCfg
from logo_pipeline.stages.watermark import WatermarkStage
from logo_pipeline.watermark_regions import (RegionsError, load_regions_store, parse_regions_store,
                                             region_overlap, regions_for_video)

W, H = 1000, 500
REGION = [(0.10, 0.10, 0.20, 0.20)]          # pixels x 100-200, y 50-100


def ov(box, regions=REGION, margin=0.02, t=0.5):
    return region_overlap(box, regions, W, H, margin, t)


# -- T48 overlap ------------------------------------------------------------------------

def test_box_fully_inside_region_overlaps():
    assert ov([120, 60, 180, 90]) == (True, 1.0, True)


def test_box_just_outside_but_within_margin_overlaps():
    # Region grows by 0.02*1000 = 20 px in x; a box starting 10 px right of the region edge is inside the margin.
    overlap, ratio, available = ov([205, 60, 215, 90])
    assert overlap is True and available
    assert ov([205, 60, 215, 90], margin=0.0)[0] is False


def test_box_elsewhere_does_not_overlap():
    assert ov([600, 300, 700, 400]) == (False, 0.0, True)


def test_no_regions_is_null_not_false():
    assert ov([120, 60, 180, 90], regions=[]) == (None, None, False)


def test_containment_uses_the_smaller_area_not_iou():
    # A large box fully containing the region: IoU would be small, containment is 1.
    overlap, ratio, _ = ov([0, 0, 1000, 500])
    assert overlap is True and ratio == 1.0


def test_best_region_counts_when_several():
    overlap, _, _ = ov([600, 300, 700, 400], regions=REGION + [(0.55, 0.55, 0.75, 0.85)])
    assert overlap is True


# -- T50 store validation -------------------------------------------------------------

def test_committed_store_loads_and_has_video_1_regions(repo_root):
    store = load_regions_store(repo_root / "configs" / "watermark_regions.yaml")
    e = store["rai_sport_default"]
    assert e.regions and all(0 <= v <= 1 for r in e.regions for v in r)
    assert "rai sport" in e.channel_brands


@pytest.mark.parametrize("bad", [
    {"k": {"regions": [[0.1, 0.1, 1.2, 0.2]]}},                  # outside [0, 1]
    {"k": {"regions": [[0.3, 0.1, 0.2, 0.2]]}},                  # x1 >= x2
    {"k": {"regions": [[0.1, 0.4, 0.2, 0.2]]}},                  # y1 >= y2
    {"k": {"regions": [[0.1, 0.1, 0.2]]}},                       # not four numbers
    {"k": {"regions": [], "colour": "red"}},                     # unknown key
    {"k": {"channel_brands": "rai", "regions": []}},             # brands not a list
])
def test_bad_store_entries_are_rejected(bad):
    with pytest.raises(RegionsError):
        parse_regions_store(bad)


def test_reference_to_missing_key_is_rejected(tmp_path):
    store = tmp_path / "regions.yaml"
    store.write_text("a: {channel_brands: [], regions: []}\n")
    video = SimpleNamespace(video_id="v", watermark_regions_ref="nope")
    with pytest.raises(RegionsError):
        regions_for_video(tmp_path, video, WatermarkCfg(), store_path=store)


def test_missing_reference_allowed_unless_required(tmp_path):
    video = SimpleNamespace(video_id="v", watermark_regions_ref=None)
    assert regions_for_video(tmp_path, video, WatermarkCfg()) is None
    with pytest.raises(RegionsError):
        regions_for_video(tmp_path, video, WatermarkCfg(regions_ref_required=True))


# -- T48 stage --------------------------------------------------------------------------

def _rep(det_id, box, w=200, h=100):
    return {"det_id": det_id, "frame_idx": 0, "frame_width": w, "frame_height": h, "loose_bbox": box,
            "box_xyxy": box}


def test_stage_without_regions_writes_nulls(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    ctx.write_jsonl("merged_detections.jsonl", [_rep(0, [1, 2, 3, 4])])
    WatermarkStage().run(ctx)
    r = ctx.read_jsonl("watermark.jsonl")[0]
    assert r == {"det_id": 0, "watermark_regions_available": False, "watermark_overlap": None,
                 "watermark_overlap_ratio": None}


def test_stage_with_video_1_regions(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    ctx.video.watermark_regions_ref = "rai_sport_default"
    # 960x560 like video_1: the bug sits at x 45-160, y 33-70.
    ctx.write_jsonl("merged_detections.jsonl", [_rep(0, [44, 30, 162, 72], 960, 560),
                                                _rep(1, [600, 300, 700, 400], 960, 560)])
    WatermarkStage().run(ctx)
    rows = {r["det_id"]: r for r in ctx.read_jsonl("watermark.jsonl")}
    assert rows[0]["watermark_overlap"] is True and rows[0]["watermark_regions_available"] is True
    assert rows[1]["watermark_overlap"] is False
    header, _ = ctx.read_jsonl("watermark.jsonl", with_header=True)
    assert header["watermark_regions_ref"] == "rai_sport_default"


# -- T49: Rule 5 on real overlaps (the five-row table; REJECT inert when disabled) ------

CH = frozenset({"rai sport", "rai"})
CH_ROW = dict(canonical_brand="rai sport", brand_in_database=False)


def test_rule5_no_regions_weak_ocr_needs_check():
    assert decide(row(**CH_ROW, watermark_overlap=None, ocr_fuzzy_score=10), known=CH)["reason_code"] == \
        "channel_brand_needs_watermark_check"


def test_rule5_inside_region_makes_no_decision():
    d = decide(row(**CH_ROW, watermark_overlap=True, ocr_fuzzy_score=10), known=CH)
    assert d["router_rule"] != 5 and d["firing"]["rule5_branch"] == "inside"


def test_rule5_outside_with_strong_ocr_makes_no_decision():
    d = decide(row(**CH_ROW, vlm_decision="uncertain", watermark_overlap=False, ocr_text="rai sport",
                   ocr_fuzzy_score=95, ocr_brand_match="rai sport"), known=CH)
    assert d["router_rule"] != 5 and d["firing"]["rule5_branch"] == "outside_ocr_supported"


def test_rule5_outside_with_weak_ocr_reviews():
    d = decide(row(**CH_ROW, watermark_overlap=False, ocr_fuzzy_score=20), known=CH)
    assert (d["router_decision"], d["reason_code"]) == ("REVIEW", "channel_brand_outside_watermark_region")


def test_rule5_reject_never_fires_when_disabled():
    d = decide(row(**CH_ROW, watermark_overlap=False, brand_from_audio=True, audio_brands_canonical=["rai sport"],
                   ocr_fuzzy_score=10), known=CH)
    assert d["router_decision"] != "REJECT"


def test_known_channel_brands_come_from_the_region_entry(repo_root):
    from mm5_helpers import BRANDS
    from logo_pipeline.stages.router.inputs import known_channel_brands
    video = SimpleNamespace(video_id="video_1", watermark_regions_ref="rai_sport_default")
    assert known_channel_brands(repo_root, video, BRANDS, WatermarkCfg()) == {"rai sport", "rai"}
    no_ref = SimpleNamespace(video_id="x", watermark_regions_ref=None)
    assert known_channel_brands(repo_root, no_ref, BRANDS, WatermarkCfg()) == BRANDS.channel_brands


def test_channel_brand_canonical_is_written(tmp_path, repo_root):
    from logo_pipeline.stages.ingest import channel_brand_canonical
    ctx = make_ctx(tmp_path, repo_root)
    assert channel_brand_canonical("Rai News 24", ctx) == "rai"
    assert channel_brand_canonical("rai sport", ctx) == "rai sport"     # not in the list: normalised spelling
    assert channel_brand_canonical(None, ctx) is None

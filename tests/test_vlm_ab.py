"""19 A12: the two VLM settings behind the route-three finding (crop upscaling, the
wordmark paragraph), their hash neutrality at the defaults, and the ``vlm-ab`` runner."""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402
from test_mm1_vlm_stage import _prep  # noqa: E402

import generate_hints as gh
from logo_pipeline.config import ConfigError, compute_config_hash, resolve_config, to_dict
from logo_pipeline.stages.merge import MergeStage
from logo_pipeline.stages.vlm import DetectionInputs
from logo_pipeline.vlm_ab import parse_sets, run_ab, sample_groups


# -- settings ----------------------------------------------------------------------------

def test_new_vlm_settings_leave_the_hash_alone_at_their_defaults(repo_root):
    video = {"video_id": "v", "source": "v.mp4"}
    r = resolve_config(str(repo_root / "configs" / "pipeline.yaml"), video)
    before = copy.deepcopy(r.resolved)                       # the resolved dict as older code built it
    del before["vlm"]["crop_min_pixels"], before["vlm"]["wordmark_rule"], before["vlm"]["wordmark_exclusions"]
    vd = {k: v for k, v in to_dict(r.video).items() if k != "overrides"}
    assert compute_config_hash(before, vd) == r.config_hash
    for o in ({"crop_min_pixels": 50176}, {"wordmark_rule": True}, {"wordmark_rule": True, "wordmark_exclusions": True}):
        assert resolve_config(str(repo_root / "configs" / "pipeline.yaml"),
                              {**video, "overrides": {"vlm": o}}).config_hash != r.config_hash


def test_crop_min_pixels_may_not_exceed_max_pixels(repo_root):
    with pytest.raises(ConfigError):
        resolve_config(str(repo_root / "configs" / "pipeline.yaml"),
                       {"video_id": "v", "source": "v.mp4", "overrides": {"vlm": {"crop_min_pixels": 200000}}})


def test_wordmark_paragraph_is_opt_in_and_has_no_hint_reference_when_blind():
    default = gh.build_final_instruction([], "open", "")
    assert "wordmarks" not in default
    hinted = gh.build_final_instruction([], "open", "", wordmark_rule=True)
    assert "ARE logos (wordmarks)" in hinted and "OCR hint" in hinted
    assert hinted.index("These are \"not_logo\"") < hinted.index("ARE logos") < hinted.index("Answer:")
    blind = gh.build_final_instruction([], "open", "", blind=True, wordmark_rule=True)
    assert "ARE logos (wordmarks)" in blind and "hint" not in blind.lower()


def test_tightened_wordmark_paragraph_keeps_text_graphics_out():
    plain = gh.wordmark_paragraph(blind=False)
    tight = gh.wordmark_paragraph(blind=False, exclusions=True)
    assert "on-screen graphics" in plain and "on-screen graphics" not in tight      # the first round's text is kept
    assert "scoreboards, captions, headlines and news tickers stay \"not_logo\"" in tight
    assert "never the source of the brand" in tight and "readable OCR text alone does not make it one" in tight
    assert "hint" not in gh.wordmark_paragraph(blind=True, exclusions=True).lower()
    text = gh.build_final_instruction([], "open", "", wordmark_rule=True, wordmark_exclusions=True)
    assert tight.strip() in text


def test_wordmark_exclusions_needs_wordmark_rule(repo_root):
    with pytest.raises(ConfigError):
        resolve_config(str(repo_root / "configs" / "pipeline.yaml"),
                       {"video_id": "v", "source": "v.mp4", "overrides": {"vlm": {"wordmark_exclusions": True}}})


def test_crop_min_pixels_reaches_the_two_crops_only(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, overrides={"vlm": {"crop_min_pixels": 4000,
                                                                       "wordmark_rule": True}})
    _prep(ctx)
    content, _, _ = DetectionInputs(ctx).content(0, ctx.open_frame_image(0), blind=False)
    images = [c for c in content if c["type"] == "image"]
    assert [c.get("min_pixels") for c in images] == [4000, 4000, None]      # tight, enlarged, full frame
    assert "ARE logos (wordmarks)" in "".join(c["text"] for c in content if c["type"] == "text")


def test_default_content_carries_no_min_pixels(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    _prep(ctx)
    content, _, _ = DetectionInputs(ctx).content(0, ctx.open_frame_image(0), blind=False)
    assert all("min_pixels" not in c for c in content)


# -- vlm-ab ------------------------------------------------------------------------------

def test_parse_sets_builds_nested_overrides():
    assert parse_sets(["vlm.crop_min_pixels=50176", "vlm.wordmark_rule=true"]) == \
        {"vlm": {"crop_min_pixels": 50176, "wordmark_rule": True}}
    with pytest.raises(ValueError):
        parse_sets(["vlm.crop_min_pixels"])


def _hinted(is_logo):
    return {"parsed": {"is_logo": is_logo}}


def test_sample_groups_pools():
    matched = [{"det_id": "0"}, {"det_id": "1"}, {"det_id": "2"}]
    unmatched = [{"det_id": "3", "max_iou_with_gold": "0.0"}, {"det_id": "4", "max_iou_with_gold": "0.3"},
                 {"det_id": "5", "max_iou_with_gold": "0.05"}]
    hinted = {0: _hinted("not_logo"), 1: _hinted("logo"), 2: _hinted("partial_logo"), 3: _hinted("not_logo"),
              4: _hinted("not_logo"), 5: _hinted("logo")}
    groups, pools = sample_groups(matched, unmatched, hinted, {"rescue": 5, "held": 5, "negative": 5}, 42)
    assert groups == {"rescue": [0], "held": [1, 2], "negative": [3]}      # 4 overlaps gold, 5 was a logo
    assert pools == {"rescue": 1, "held": 2, "negative": 1}


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _ab_run(tmp_path, repo_root):
    """Five detections on one frame: 0-1 rejected true logos, 2 a kept one, 3-4 unlabelled boxes."""
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    boxes = [[5 + 38 * i, 10, 30 + 38 * i, 35] for i in range(5)]
    ctx.write_jsonl("raw_detections.jsonl", [
        {"raw_det_id": i, "frame_idx": 0, "source": "yolo", "box_xyxy": b, "score": 0.8, "label": "logo",
         "class_conf": None, "frame_width": 200, "frame_height": 100} for i, b in enumerate(boxes)],
        header=ctx.header("detect_yolo"))
    MergeStage().run(ctx)
    ids = sorted(r["det_id"] for r in ctx.read_jsonl("merged_detections.jsonl"))
    assert len(ids) == 5
    reps = {r["det_id"]: r for r in ctx.read_jsonl("merged_detections.jsonl")}
    ctx.write_jsonl("refinement.jsonl", [{"det_id": d, "selected_bbox": reps[d]["loose_bbox"]} for d in ids],
                    header=ctx.header("refine"))
    ctx.write_jsonl("ocr.jsonl", [{"det_id": d, "ocr_text": "", "ocr_source": None, "ocr_lines": []} for d in ids],
                    header=ctx.header("ocr"))
    ctx.write_jsonl("faiss_hints.jsonl", [{"det_id": d, "faiss_top5_brands": [], "faiss_top5_scores": []}
                                          for d in ids], header=ctx.header("faiss_query"))
    ctx.write_json("audio/audio_hints.json", {"status": "empty", "brands": []})
    ctx.write_jsonl("vlm/hinted.jsonl", [
        {"det_id": d, "parsed": {"is_logo": "logo" if d == ids[2] else "not_logo",
                                 "brand": "verisure" if d == ids[2] else None}, "elapsed_s": 2.0} for d in ids],
        header=ctx.header("vlm"))
    (ctx.run_dir / "reports").mkdir(exist_ok=True)
    _write_csv(ctx.run_dir / "reports" / "iou_match_matched.csv",
               ["frame_idx", "det_id", "gold_class", "canonical_brand", "brand_correct", "router_decision",
                "reason_code", "vlm_decision"],
               [[0, ids[0], "verisure", "", False, "REJECT", "high_confidence_not_logo", "not_logo"],
                [0, ids[1], "verisure", "", False, "REJECT", "high_confidence_not_logo", "not_logo"],
                [0, ids[2], "verisure", "verisure", True, "ACCEPT", "x", "logo"]])
    _write_csv(ctx.run_dir / "reports" / "iou_match_unmatched.csv",
               ["frame_idx", "det_id", "max_iou_with_gold", "router_decision", "reason_code", "vlm_decision"],
               [[0, ids[3], 0.0, "REJECT", "high_confidence_not_logo", "not_logo"],
                [0, ids[4], 0.0, "REJECT", "high_confidence_not_logo", "not_logo"]])
    return ctx, ids


def _fake_generate(calls):
    def gen(content):
        calls.append(content)
        upscaled = any(c.get("min_pixels") for c in content if c["type"] == "image")
        return json.dumps({"is_logo": "logo" if upscaled else "not_logo", "brand": "verisure" if upscaled else None,
                           "confidence": 90, "reasoning": "r"})
    return gen


def test_vlm_ab_applies_the_overrides_reports_flips_and_resumes(tmp_path, repo_root):
    ctx, ids = _ab_run(tmp_path, repo_root)
    hinted_before = ctx.path("vlm/hinted.jsonl").read_text(encoding="utf-8")
    calls = []
    n = {"rescue": 10, "held": 10, "negative": 10}
    path = run_ab(ctx.run_dir, ["vlm.crop_min_pixels=4000"], "up", n, 42, repo_root,
                  generate_fn=_fake_generate(calls))
    assert len(calls) == 5
    text = path.read_text(encoding="utf-8")
    assert "| rescue | true logos the run REJECTed (hinted `not_logo`) | 2 / 2 | 2 now logo or partial | 100.0% |" in text
    assert "| held | true logos the run kept as logo | 1 / 1 | 0 now not_logo | 0.0% |" in text
    assert "2 / 2 | 2 now logo or partial | 100.0% |" in text.split("| negative |")[1]
    assert "Brand correct among rescued boxes: 2 of 2" in text
    with open(ctx.run_dir / "reports" / "vlm_ab_up.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and {r["group"] for r in rows} == {"rescue", "held", "negative"}
    assert ctx.path("vlm/hinted.jsonl").read_text(encoding="utf-8") == hinted_before   # the run is untouched

    calls.clear()
    run_ab(ctx.run_dir, ["vlm.crop_min_pixels=4000"], "up", n, 42, repo_root, generate_fn=_fake_generate(calls))
    assert calls == []                                                                 # resumed: nothing to do
    with pytest.raises(ValueError):
        run_ab(ctx.run_dir, ["vlm.crop_min_pixels=8000"], "up", n, 42, repo_root, generate_fn=_fake_generate(calls))


def test_vlm_ab_baseline_tag_reproduces_the_run(tmp_path, repo_root):
    ctx, _ = _ab_run(tmp_path, repo_root)
    calls = []
    path = run_ab(ctx.run_dir, [], "baseline", {"rescue": 10, "held": 10, "negative": 10}, 42, repo_root,
                  generate_fn=_fake_generate(calls))
    text = path.read_text(encoding="utf-8")
    assert "| 0 now logo or partial | 0.0% |" in text                    # no min_pixels: every reply not_logo

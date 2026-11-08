"""MM11 (spec docs/17): router-only (T61), metrics (T62), IoU matching (T63),
blind vs hinted (T64), the Rule 5 report (T71), the route-two label request and
the route-three sampling-rate check (T66), and the acceptance caveats."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from test_mm5_router_stage import _build  # noqa: E402

import acceptance_evidence as ae  # noqa: E402
import blind_vs_hinted as bvh  # noqa: E402
import iou_match as im  # noqa: E402
import label_request as lr  # noqa: E402
import metrics as mt  # noqa: E402
import sampling_rate_check as src  # noqa: E402
import watermark_report as wr  # noqa: E402

from logo_pipeline.brands import BrandEntry, build_brand_list  # noqa: E402
from logo_pipeline.router_only import decision_lines, replay, write  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BRANDS = build_brand_list("2026-01-01", [BrandEntry("verisure", ("very sure",)), BrandEntry("rai", (), True),
                                         BrandEntry("hp")])


# -- T61 router-only -------------------------------------------------------------------

def test_router_only_reproduces_the_pipeline_decisions(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    outputs, firing, resolved = replay(ctx.run_dir, str(repo_root / "configs" / "pipeline.yaml"), repo_root=repo_root)
    out = tmp_path / "replay"
    write(ctx.run_dir, out, outputs, firing, resolved)
    assert decision_lines(out / "router_output.jsonl") == decision_lines(ctx.path("router_output.jsonl"))
    assert decision_lines(out / "firing_log.jsonl") == decision_lines(ctx.path("firing_log.jsonl"))


def test_router_only_under_a_changed_yaml_and_rows_csv_copy(tmp_path, repo_root):
    import yaml

    from logo_pipeline.stages.output import OutputStage
    ctx = _build(tmp_path, repo_root)
    OutputStage().run(ctx)
    cfg = yaml.safe_load((repo_root / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["router"]["thresholds"]["rule6_ocr_fuzzy_threshold"] = 99
    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump(cfg))
    outputs, firing, resolved = replay(ctx.run_dir, str(alt), repo_root=repo_root)
    out = tmp_path / "alt_out"
    info = write(ctx.run_dir, out, outputs, firing, resolved)
    df = pd.read_csv(out / "rows.csv")
    assert info["rows_csv"] and df.loc[df["det_id"] == 0, "router_rule"].iloc[0] != 6
    assert (out / df.loc[0, "crop_path"]).exists()          # paths re-expressed for the new CSV location


# -- T62 metrics -------------------------------------------------------------------------

def _rows_and_labels():
    rows, labels = [], {}
    spec = [("REVIEW", "code_a")] * 10 + [("ACCEPT", "clear_logo_with_strong_ocr_match")] * 6 + \
           [("REJECT", "high_confidence_not_logo")] * 4
    labs = ["TP"] * 5 + ["FP"] * 3 + ["wrong_brand"] * 2 + ["TP"] * 4 + ["FP"] + ["wrong_brand"] + ["FN"] * 4
    for i, ((dec, code), lab) in enumerate(zip(spec, labs)):
        rows.append({"det_id": i, "video_id": "v", "run_id": "r", "router_decision": dec, "reason_code": code,
                     "brand_evidence": "none", "canonical_match_mode": "exact", "refinement_status": "no_candidate"})
        labels[("v", "r", i)] = {"label": lab}
    return rows, labels


def test_metrics_hand_computed_precision():
    rows, labels = _rows_and_labels()
    firing = [{"deciding_rule": "6", "rules_matched_preempted": [{"rule": "16"}]}] * 6
    m = mt.compute(rows, firing, labels, "fp")
    assert mt.precision(m["by_code"]["code_a"], "fp") == (0.5, 10)            # 5 / (5 + 3 + 2)
    assert mt.precision(m["by_code"]["code_a"], "separate") == (0.625, 8)     # 5 / (5 + 3)
    assert mt.precision(m["accept"], "fp") == (pytest.approx(4 / 6), 6)
    assert mt.precision(m["accept"], "separate") == (0.8, 5)
    assert m["review_load"] == 10 and m["labelled"] == 20
    assert m["preempted"]["16"] == 6
    text = mt.render(Path("r"), m)
    assert "**ACCEPT precision:** 0.667" in text and "Review load" in text


def test_metrics_cli_without_labels(tmp_path, repo_root):
    ctx = _build(tmp_path, repo_root)
    assert mt.main(["--run", str(ctx.run_dir), "--no-labels"]) == 0
    text = (ctx.run_dir / "reports" / "metrics.md").read_text(encoding="utf-8")
    assert "Decision distribution" in text and "0 labelled" in text


# -- T63 IoU matching ----------------------------------------------------------------------

def test_iou_greedy_one_to_one():
    gold = [{"box": [0, 0, 10, 10]}, {"box": [20, 20, 30, 30]}]
    dets = [{"loose_bbox": [0, 0, 10, 11]}, {"loose_bbox": [1, 1, 10, 10]}, {"loose_bbox": [50, 50, 60, 60]}]
    pairs = im.match_frame(gold, dets, 0.5, "loose_bbox")
    assert [(g, d) for g, d, _ in pairs] == [(0, 0)]


def test_iou_counts_on_two_frames_with_channel_exclusion():
    gold = {0: [{"class_name": "verisure", "box": [0, 0, 10, 10]}, {"class_name": "rai", "box": [20, 20, 30, 30]}],
            1: [{"class_name": "hp", "box": [5, 5, 15, 15]}]}
    rows = [{"frame_idx": 0, "loose_bbox": [0, 0, 10, 10], "selected_bbox": [0, 0, 10, 10], "router_decision": "ACCEPT",
             "canonical_brand": "verisure", "detection_sources": ["gdino", "yolo"], "det_id": 0,
             "reason_code": "clear_logo_ocr_confirmed", "vlm_decision": "logo"},
            {"frame_idx": 0, "loose_bbox": [20, 20, 30, 30], "selected_bbox": [40, 40, 45, 45],
             "router_decision": "UNKNOWN_LOGO", "canonical_brand": "UNKNOWN", "detection_sources": ["gdino"],
             "det_id": 1, "reason_code": "unknown_logo", "vlm_decision": "logo"},
            {"frame_idx": 1, "loose_bbox": [90, 90, 99, 99], "selected_bbox": [90, 90, 99, 99],
             "router_decision": "REJECT", "canonical_brand": None, "detection_sources": ["yolo"]}]
    loose = im.evaluate(rows, gold, 0.5, "loose_bbox", BRANDS, {"rai"})
    assert (loose["all"]["matched"], loose["all"]["gold_not_detected"], loose["all"]["detection_without_gold"]) == (2, 1, 1)
    assert (loose["no_channel"]["matched"], loose["no_channel"]["gold_not_detected"]) == (1, 1)
    assert loose["brand_correct"] == 1 and loose["accept"] == Counter(correct=1)
    assert loose["known_routed_unknown"] == 1                 # rai is in the list but routed UNKNOWN_LOGO
    assert loose["recall_by_source"] == Counter(gdino=2, yolo=1)
    assert loose["reasons"] == Counter({("ACCEPT", "clear_logo_ocr_confirmed"): 1, ("UNKNOWN_LOGO", "unknown_logo"): 1})
    assert loose["vlm_decisions"] == Counter(logo=2) and sorted(r[1] for r in loose["matched_rows"]) == [0, 1]
    sel = im.evaluate(rows, gold, 0.5, "selected_bbox", BRANDS, {"rai"})
    assert sel["all"]["matched"] == 1                         # refinement moved the rai box off its gold
    text = im.render(Path("r"), 0.5, loose, sel, None)
    assert "selected_bbox" in text and "| UNKNOWN_LOGO | unknown_logo | 1 |" in text


def test_iou_gold_from_csv(tmp_path):
    p = tmp_path / "g.csv"
    pd.DataFrame([{"frame_idx": 3, "class_name": "hp", "x1": 1, "y1": 2, "x2": 3, "y2": 4},
                  {"frame_idx": 9, "class_name": "hp", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]).to_csv(p, index=False)
    assert list(im.gold_from_csv(p, {3: (100, 100)})) == [3]   # unsampled frame 9 ignored


# -- T64 blind vs hinted -----------------------------------------------------------------

def test_blind_vs_hinted_matrix_and_attribution():
    rows = []
    for i in range(10):
        hinted = "logo" if i < 6 else "not_logo"
        blind = "logo" if i < 4 else ("not_logo" if i >= 6 else "partial_logo")
        rows.append({"det_id": i, "vlm_decision": hinted, "vlm_decision_blind": blind,
                     "canonical_brand": "verisure" if i < 6 else None,
                     "vlm_brand_blind_canonical": "verisure" if i < 3 else ("UNKNOWN" if i < 6 else None),
                     "hinted_brand_in_faiss_hints": i in (0, 3, 4), "brand_from_audio": i == 5,
                     "ocr_brand_match": "verisure" if i == 1 else None})
    a = bvh.analyse(rows)
    assert a["matrix"][("logo", "logo")] == 4 and a["matrix"][("logo", "partial_logo")] == 2
    assert a["matrix"][("not_logo", "not_logo")] == 4
    assert (a["n_pos"], a["brand_agree"]) == (6, 3)
    assert a["faiss"] == (3, 2) and a["audio"] == (1, 1) and a["ocr"] == (1, 0)
    assert a["blind_unknown"] == 3
    assert "Blind says UNKNOWN where hinted names a brand: 3 of 6" in bvh.render(Path("r"), a, None)


# -- T71 ------------------------------------------------------------------------------------

def test_watermark_report_counts():
    firing = [{"det_id": i, "rule5_branch": b} for i, b in enumerate(
        ["inside", "outside_ocr_supported", "outside_weak_ocr", "outside_weak_ocr", "no_regions_needs_check", None])]
    rows = [{"det_id": i, "ocr_fuzzy_score": s, "reason_code": "x"} for i, s in enumerate([0, 95, 10, 92, 0, 0])]
    a = wr.analyse(firing, rows)
    assert a["n_channel"] == 5 and a["branches"]["no_regions"] == 1 and a["branches"]["outside_weak_ocr"] == 2
    assert (a["n_outside"], a["n_outside_strong"]) == (3, 2)


# -- T66 route two ----------------------------------------------------------------------------

def _routed(n_a=100, n_b=30):
    rows = [{"det_id": i, "router_decision": "REVIEW", "reason_code": "big_code"} for i in range(n_a)]
    rows += [{"det_id": n_a + i, "router_decision": "REJECT", "reason_code": "small_code"} for i in range(n_b)]
    rows += [{"det_id": 1000 + i, "router_decision": d, "reason_code": c} for i, (d, c) in
             enumerate([("ACCEPT", "clear_logo_with_strong_ocr_match")] * 3 + [("UNKNOWN_LOGO", "u")] * 2)]
    return rows


def test_label_request_is_deterministic_and_follows_the_rules():
    rows = _routed()
    sel, counts, small = lr.sample(rows, 70, seed=42)
    assert sel == lr.sample(rows, 70, seed=42)[0]
    assert counts["big_code"] == 70 and counts["small_code"] == 30 and small == ["small_code"]
    assert {r["det_id"] for r in sel} >= {1000, 1001, 1002, 1003, 1004}     # every ACCEPT and UNKNOWN_LOGO
    assert sel != lr.sample(rows, 70, seed=7)[0]
    with pytest.raises(ValueError):
        lr.sample(rows, 50, seed=42)                                        # outside the 60-80 band


# -- T66 route three ------------------------------------------------------------------------

def test_phash_matches_itself_and_rejects_a_different_frame():
    rng = np.random.default_rng(0)
    a = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    b = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    assert src.distance(src.phash(a), src.phash(a)) == 0
    assert src.distance(src.phash(a), src.phash(b)) > 10
    frames = {1: a, 2: b}
    share, _ = src.check_rate(lambda t: a if t < 0.5 else b, frames, 1.0, 10)
    assert share == 1.0
    share, _ = src.check_rate(lambda t: a, frames, 1.0, 10)                  # a wrong rate maps to the wrong frame
    assert share == 0.5


# -- acceptance evidence ------------------------------------------------------------------------

def test_acceptance_caveats_follow_the_run():
    manifest = {"mode": "frames", "config": {"router": {"enhanced_vlm_fields_available": False,
                                                        "not_logo_confidence_source": "generated",
                                                        "rules": {"rule_7": False, "rule_8": False}}}}
    text = "\n".join(ae.caveats(manifest, {}, has_blind=False, has_scores=False))
    for phrase in ("Rule 6", "YOLO", "Rules 2 and 3 are unreachable", "`generated`", "frames mode",
                   "No blind pass", "No scored decision field", "No channel declared"):
        assert phrase in text


def test_acceptance_caveat_states_a_held_scored_switch(tmp_path):
    rep = tmp_path / "reports"
    (rep / "router_scored").mkdir(parents=True)
    (rep / "score_distribution.md").write_text("# T32\n\n**Result: PASS.**\n", encoding="utf-8")

    def jsonl(path, header, decisions):
        path.write_text("\n".join(json.dumps(r) for r in [header] + [{"det_id": i, "router_decision": d}
                                                                     for i, d in enumerate(decisions)]) + "\n",
                        encoding="utf-8")
    jsonl(tmp_path / "router_output.jsonl", {"_header": True}, ["REJECT", "REJECT", "REVIEW"])
    jsonl(rep / "router_scored" / "router_output.jsonl", {"_header": True, "not_logo_confidence_source": "scored"},
          ["REJECT", "REVIEW", "REVIEW"])
    t32 = ae.t32_outcome(tmp_path)
    assert t32["passed"] and t32["replay"] == "router_scored"
    manifest = {"config": {"router": {"not_logo_confidence_source": "generated"}}}
    text = "\n".join(ae.caveats(manifest, {}, has_blind=True, has_scores=True, t32=t32))
    assert "T32 passed" in text and "review load goes from 1 to 2" in text and "A11" in text
    assert "until T32 passes" not in text


def test_labels_on_deduped_rows_are_flagged():
    rows = [{"det_id": 0, "video_id": "v", "run_id": "r", "router_decision": "DEDUPED",
             "reason_code": "duplicate_same_brand_box"}]
    m = mt.compute(rows, [], {("v", "r", 0): {"label": "TP"}})
    assert m["deduped_labelled"] == 1
    assert "1 labels are on DEDUPED rows" in mt.render(Path("r"), m)


def test_rejected_true_logos_splits_sizes_reasons_and_writes_a_sheet(tmp_path):
    import rejected_true_logos as rtl
    from PIL import Image
    (tmp_path / "reports").mkdir()
    (tmp_path / "crops").mkdir()
    Image.new("RGB", (40, 20), "red").save(tmp_path / "crops" / "c0.jpg")
    pd.DataFrame([{"det_id": 0, "loose_bbox": "[0, 0, 40, 20]", "box_area_ratio": 0.001, "confidence": 90,
                   "reasoning": "Plain text on a banner.", "crop_path": "crops/c0.jpg"},
                  {"det_id": 1, "loose_bbox": "[0, 0, 200, 100]", "box_area_ratio": 0.05, "confidence": 85,
                   "reasoning": "Clear wordmark.", "crop_path": "crops/missing.jpg"},
                  {"det_id": 2, "loose_bbox": "[0, 0, 50, 50]", "box_area_ratio": 0.01, "confidence": 80,
                   "reasoning": "", "crop_path": ""}]).to_csv(tmp_path / "rows.csv", index=False)
    pd.DataFrame([{"det_id": 0, "gold_class": "eni", "router_decision": "REJECT"},
                  {"det_id": 1, "gold_class": "uber", "router_decision": "ACCEPT"},
                  {"det_id": 2, "gold_class": "eni", "router_decision": "DEDUPED"}]
                 ).to_csv(tmp_path / "reports" / "iou_match_matched.csv", index=False)
    rejected, kept = rtl.load(tmp_path)
    assert [r["det_id"] for r in rejected] == [0] and [r["det_id"] for r in kept] == [1]   # DEDUPED in neither
    assert rejected[0]["short_side"] == 20
    assert rtl.main(["--run", str(tmp_path)]) == 0
    text = (tmp_path / "reports" / "rejected_true_logos.md").read_text(encoding="utf-8")
    assert "| text | 1 |" in text and "| banner | 1 |" in text and "1 crops" in text
    assert (tmp_path / "reports" / "rejected_true_logos.png").exists()
    assert (tmp_path / "reports" / "rejected_true_logos.csv").exists()


def test_vlm_ab_review_lists_negative_flips_and_brand_misses(tmp_path):
    import vlm_ab_review as vr
    from PIL import Image
    (tmp_path / "reports").mkdir()
    (tmp_path / "crops").mkdir()
    Image.new("RGB", (40, 20), "blue").save(tmp_path / "crops" / "c3.jpg")
    pd.DataFrame([{"det_id": d, "crop_path": f"crops/c{d}.jpg"} for d in (1, 2, 3, 4)]
                 ).to_csv(tmp_path / "rows.csv", index=False)
    cols = ["det_id", "group", "gold_class", "run_is_logo", "variant_is_logo", "run_brand", "variant_brand",
            "variant_brand_canonical", "variant_brand_correct", "variant_confidence", "elapsed_s", "variant_reasoning"]
    pd.DataFrame([[1, "rescue", "eni", "not_logo", "logo", "", "enilive", "enilive", False, 90, 1, "a"],
                  [2, "rescue", "eni", "not_logo", "logo", "", "eni", "eni", True, 90, 1, "b"],
                  [3, "negative", "", "not_logo", "logo", "", "UNKNOWN", "UNKNOWN", False, 80, 1, "c | d"],
                  [4, "negative", "", "not_logo", "not_logo", "", "", "", False, 100, 1, "e"]],
                 columns=cols).to_csv(tmp_path / "reports" / "vlm_ab_t.csv", index=False)
    flips, misses = vr.load(tmp_path, "t")
    assert [r["det_id"] for r in flips] == ["3"] and [r["det_id"] for r in misses] == ["1"]
    assert vr.main(["--run", str(tmp_path), "--tag", "t"]) == 0
    text = (tmp_path / "reports" / "vlm_ab_t_review.md").read_text(encoding="utf-8")
    assert "| eni | enilive | enilive | 1 |" in text and "c / d" in text and "(1 crops" in text
    assert (tmp_path / "reports" / "vlm_ab_t_negatives.png").exists()


def test_vlm_ab_compare_pairs_two_tags_and_lists_disagreements(tmp_path):
    import vlm_ab_compare as vc
    from PIL import Image
    (tmp_path / "reports").mkdir()
    (tmp_path / "crops").mkdir()
    Image.new("RGB", (40, 20), "green").save(tmp_path / "crops" / "c1.jpg")
    pd.DataFrame([{"det_id": d, "crop_path": f"crops/c{d}.jpg"} for d in range(1, 7)]
                 ).to_csv(tmp_path / "rows.csv", index=False)
    cols = ["det_id", "group", "gold_class", "run_is_logo", "variant_is_logo", "run_brand", "variant_brand",
            "variant_brand_canonical", "variant_brand_correct", "variant_confidence", "elapsed_s", "variant_reasoning"]

    def tag(name, calls, brands):
        groups = ["rescue", "rescue", "held", "negative", "negative", "negative"]
        gold = ["eni", "uber", "eni", "", "", ""]
        pd.DataFrame([[d, g, gc, "x", c, "", b, b, b == gc and b != "", 90, 1, f"{name} {d}"]
                      for d, g, gc, c, b in zip(range(1, 7), groups, gold, calls, brands)],
                     columns=cols).to_csv(tmp_path / "reports" / f"vlm_ab_{name}.csv", index=False)

    tag("a", ["logo", "logo", "logo", "logo", "logo", "not_logo"], ["eni", "uber", "eni", "rolex", "", ""])
    tag("b", ["not_logo", "logo", "logo", "not_logo", "not_logo", "not_logo"], ["", "uber", "eni", "", "", ""])
    paired, only_a, only_b = vc.load(tmp_path, "a", "b")
    assert len(paired) == 6 and (only_a, only_b) == (0, 0)
    stats = vc.summarise(paired)
    assert (stats["rescue"]["a"], stats["rescue"]["b"], stats["rescue"]["only_a"]) == (2, 1, 1)
    assert (stats["negative"]["a"], stats["negative"]["b"], stats["negative"]["only_b"]) == (1, 3, 2)
    assert stats["held"]["only_a"] == stats["held"]["only_b"] == 0
    assert [r["det_id"] for r in vc.disagreements(paired)] == [1, 4, 5]
    assert vc.sign_test(0, 0) == 1.0 and vc.sign_test(5, 0) == pytest.approx(0.0625)
    assert vc.sign_test(9, 0) == pytest.approx(2 / 512) and vc.sign_test(3, 3) == 1.0
    assert vc.main(["--run", str(tmp_path), "--a", "a", "--b", "b"]) == 0
    text = (tmp_path / "reports" / "vlm_ab_compare_a_vs_b.md").read_text(encoding="utf-8")
    assert "| rescue | 2 | logo or partial | 2 (100.0%) | 1 (50.0%) | 1 | 0 | 1.000 |" in text
    assert "| 4 | negative | — | logo | rolex | not_logo | — |" in text and "(1)" in text
    assert (tmp_path / "reports" / "vlm_ab_compare_a_vs_b.png").exists()


def test_vlm_speed_probe_summaries_and_report():
    import vlm_speed_probe as sp
    assert sp.parse_smi("23, 41.46, 2520, 3105, P2") == {"utilization.gpu": 23.0, "power.draw": 41.46,
                                                           "clocks.sm": 2520.0, "clocks.max.sm": 3105.0,
                                                           "pstate": "P2"}
    assert sp.parse_smi("[N/A], 1, 2, 3, P0") is None and sp.parse_smi("1, 2") is None
    calls = [{"det_id": 3, "preprocess_s": 0.5, "prefill_s": 1.0, "full_s": 11.0, "total_s": 11.5,
              "input_tokens": 1200, "image_tokens": 600, "new_tokens": 101},
             {"det_id": 9, "preprocess_s": 0.7, "prefill_s": 2.0, "full_s": 7.0, "total_s": 7.7,
              "input_tokens": 1300, "image_tokens": 700, "new_tokens": 51}]
    s = sp.summarise_calls(calls)
    assert s["decode_tok_s"] == pytest.approx(10.0) and s["prefill_s"] == pytest.approx(1.5) and s["n"] == 2
    load = sp.summarise_samples([{"cpu_system": 50, "utilization.gpu": 20, "pstate": "P2"},
                                 {"cpu_system": 70, "utilization.gpu": 40, "pstate": "P2"}])
    assert load["cpu_system"] == (60, 70) and load["power.draw"] is None and load["pstate"] == [("P2", 2)]
    assert sp.pick_det_ids({i: None for i in range(10)}, 5) == [0, 2, 4, 6, 8]
    text = sp.render(Path("r"), "x", [], {"gpu": "g"}, 12.0, {"python_loop_s": 0.4}, calls, s, load)
    assert "| 1.50 | 1.0 |" not in text and "| 0.60 | 1.50 | 10.0 | 76 | 1250 | 650 | 9.6 |" in text
    assert "| system CPU % | 60 / 70 |" in text and "| power W | — |" in text


EVAL_SCRIPTS = ("acceptance_evidence", "blind_vs_hinted", "faiss_rebenchmark", "faiss_selfmatch", "iou_match",
                "label_request", "merge_ablation", "metrics", "ordering_checks", "refinement_report",
                "rejected_true_logos", "sampling_rate_check", "score_distribution", "vlm_ab_compare",
                "vlm_ab_review", "vlm_cost", "vlm_speed_probe", "watermark_report")


@pytest.mark.parametrize("name", EVAL_SCRIPTS)
def test_every_eval_script_has_help(name):
    mod = __import__(name)
    with pytest.raises(SystemExit) as e:
        mod.main(["--help"])
    assert e.value.code == 0


def test_run_scripts_write_markdown_and_csv(tmp_path, repo_root):
    """17 DoD: every script writes markdown + CSV under reports/ and runs on a run directory."""
    import merge_ablation
    import ordering_checks
    import refinement_report
    import vlm_cost

    from logo_pipeline.stages.output import OutputStage
    from test_mm1_output_stage import _run_upto_router
    from helpers import make_ctx
    ctx = make_ctx(tmp_path, repo_root, n_frames=2)
    _run_upto_router(ctx, [{"raw_det_id": i, "frame_idx": 0, "source": "gdino", "box_xyxy": b, "score": 0.9,
                            "label": "a", "class_conf": None, "frame_width": 200, "frame_height": 100}
                           for i, b in enumerate([[10, 10, 60, 60], [12, 12, 61, 61], [100, 20, 150, 70]])])
    OutputStage().run(ctx)
    run = str(ctx.run_dir)
    expected = {"metrics": ["metrics.md", "metrics_decisions.csv"],
                "ordering_checks": ["ordering_checks.md", "ordering_checks.csv"],
                "refinement_report": ["refinement.md", "refinement.csv"],
                "merge_ablation": ["merge_distributions.md", "merge_distributions.csv"],
                "watermark_report": ["watermark_report.md", "watermark_report.csv"],
                "blind_vs_hinted": ["blind_vs_hinted.md", "blind_vs_hinted_matrix.csv"],
                "vlm_cost": ["vlm_cost.md", "vlm_cost.csv"],
                "acceptance_evidence": ["acceptance_evidence.md", "acceptance_evidence.csv"]}
    for name, files in expected.items():
        mod = __import__(name)
        args = ["--run", run] + (["--no-labels"] if name == "metrics" else []) + \
            (["--config", str(repo_root / "configs" / "pipeline.yaml")] if name == "blind_vs_hinted" else [])
        assert mod.main(args) in (0, None), name
        for f in files:
            assert (ctx.run_dir / "reports" / f).exists(), (name, f)

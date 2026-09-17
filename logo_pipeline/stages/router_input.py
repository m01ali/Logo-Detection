"""``stages/router_input.py`` (07 MM1 Step 4.3): joins every per-stage
JSONL into ``router_input.jsonl``, the frozen row (05) minus the router's
own decision fields (section K's five ``router_input``-stage fields are
computed here with a best-effort formula; their full T43 definition is MM5,
``spec docs/11`` — out of scope for this document). Missing stages (not
yet built, or disabled by the YAML) leave their fields null, which is what
lets MM1 ship a complete, valid row shape before MM2–MM9 exist.

Crop paths (05 section C, Stage=output) are intentionally left empty here:
the crop JPEGs are cut and saved by ``stages/output.py``, which runs after
the router, so this file cannot reference files that do not exist yet.
"""

from __future__ import annotations

from dataclasses import asdict

from ..schema import Row
from .base import (AUDIO_HINTS_FILE, CANONICAL_FILE, FAISS_HINTS_FILE, MERGED_DETECTIONS_FILE, OCR_FILE,
                    REFINEMENT_FILE, ROUTER_INPUT_FILE, VLM_BLIND_FILE, VLM_HINTED_FILE,
                    VLM_HINT_COMPOSITION_FILE, WATERMARK_FILE, RunContext, Stage, StageResult)


def _by_det(ctx: RunContext, relpath: str) -> dict[int, dict]:
    if not ctx.path(relpath).exists():
        return {}
    return {r["det_id"]: r for r in ctx.read_jsonl(relpath)}


def _clamp(val):
    if val is None:
        return None
    try:
        return max(0, min(100, int(float(val))))
    except (TypeError, ValueError):
        return None


class RouterInputStage(Stage):
    name = "router_input"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [ROUTER_INPUT_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.cfg
        manifest = ctx.manifest
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)
        refined = _by_det(ctx, REFINEMENT_FILE)
        ocr = _by_det(ctx, OCR_FILE)
        faiss = _by_det(ctx, FAISS_HINTS_FILE)
        vlm_hinted = _by_det(ctx, VLM_HINTED_FILE)
        vlm_blind = _by_det(ctx, VLM_BLIND_FILE)
        hint_comp = _by_det(ctx, VLM_HINT_COMPOSITION_FILE)
        canonical = _by_det(ctx, CANONICAL_FILE)
        watermark = _by_det(ctx, WATERMARK_FILE)
        sampled_by_frame = {f["frame_idx"]: f for f in ctx.sampled_frames()}
        audio_hints = ctx.read_json(AUDIO_HINTS_FILE) if ctx.path(AUDIO_HINTS_FILE).exists() else {}
        audio_brands = audio_hints.get("brands", [])
        audio_status = audio_hints.get("status", "disabled")

        rows: list[dict] = []
        for rep in reps:
            det_id = rep["det_id"]
            sframe = sampled_by_frame.get(rep["frame_idx"], {})
            ref, oc, fq = refined.get(det_id, {}), ocr.get(det_id, {}), faiss.get(det_id, {})
            vh, vb = vlm_hinted.get(det_id, {}), vlm_blind.get(det_id, {})
            hc, can, wm = hint_comp.get(det_id, {}), canonical.get(det_id, {}), watermark.get(det_id, {})
            vh_parsed = vh.get("parsed") or {}
            vb_parsed = vb.get("parsed") or {}

            canonical_brand = can.get("canonical_brand")
            ocr_brand_match = oc.get("ocr_brand_match")
            faiss_top1_score = fq.get("faiss_top1_score")
            faiss_confirmed = canonical_brand is not None and can.get("faiss_top1_brand_canonical") == canonical_brand
            ocr_confirmed = canonical_brand is not None and ocr_brand_match == canonical_brand
            if ocr_confirmed and faiss_confirmed:
                brand_evidence = "both"
            elif ocr_confirmed:
                brand_evidence = "ocr_confirmed"
            elif faiss_confirmed:
                brand_evidence = "faiss_confirmed"
            else:
                brand_evidence = "none"

            row = Row(
                video_id=manifest["video_id"], run_id=manifest["run_id"], det_id=det_id,
                frame_idx=rep["frame_idx"], timecode_s=sframe.get("timecode_s", 0.0),
                shot_id=sframe.get("shot_id"), frame_width=rep["frame_width"], frame_height=rep["frame_height"],
                det_score=rep["det_score"], box_xyxy=rep["loose_bbox"], loose_bbox=rep["loose_bbox"],
                box_area_ratio=rep["box_area_ratio"], box_touches_edge=rep["box_touches_edge"],
                detection_sources=rep["detection_sources"], merge_count=rep["merge_count"],
                merged_raw_det_ids=rep["merged_raw_det_ids"], gdino_label=rep.get("gdino_label"),
                yolo_class=rep.get("yolo_class"), yolo_class_conf=rep.get("yolo_class_conf"),

                refined_bbox=ref.get("refined_bbox"), refinement_method=ref.get("refinement_method", "none"),
                refinement_status=ref.get("refinement_status", "no_candidate"),
                selected_bbox=ref.get("selected_bbox") or rep["loose_bbox"],
                has_valid_tight_bbox=ref.get("has_valid_tight_bbox", False),
                refinement_area_ratio=ref.get("refinement_area_ratio"),

                ocr_text=oc.get("ocr_text", ""), ocr_text_raw=oc.get("ocr_text_raw", ""),
                ocr_source=oc.get("ocr_source"), ocr_max_confidence=oc.get("ocr_max_confidence"),
                ocr_brand_match=ocr_brand_match, ocr_fuzzy_score=oc.get("ocr_fuzzy_score", 0),
                ocr_match_list=oc.get("ocr_match_list"), ocr_match_mode=oc.get("ocr_match_mode"),

                faiss_top1=fq.get("faiss_top1"), faiss_top1_score=faiss_top1_score,
                faiss_top5_brands=fq.get("faiss_top5_brands", []), faiss_top5_scores=fq.get("faiss_top5_scores", []),
                faiss_top1_brand_canonical=can.get("faiss_top1_brand_canonical"),
                faiss_score_gap=fq.get("faiss_score_gap"), faiss_agrees_with_vlm=can.get("faiss_agrees_with_vlm"),
                brand_in_database=can.get("brand_in_database"),

                audio_brands=audio_brands, audio_brands_canonical=can.get("audio_brands_canonical", []),
                brand_from_audio=can.get("brand_from_audio", False), audio_signal_status=audio_status,

                is_logo=vh_parsed.get("is_logo"), vlm_decision=vh_parsed.get("is_logo"),
                brand=vh_parsed.get("brand"), vlm_brand=vh_parsed.get("brand"),
                is_truncated=vh_parsed.get("is_truncated"),
                logo_probability=_clamp(vh_parsed.get("logo_probability")), confidence=_clamp(vh_parsed.get("confidence")),
                signals_used=vh_parsed.get("signals_used", []), signals_used_raw=vh_parsed.get("signals_used_raw", []),
                reasoning=vh_parsed.get("reasoning"), vlm_parse_ok=vh.get("vlm_parse_ok", False),
                logo_probability_scored=None, not_logo_confidence=None, decision_scores=None,

                vlm_decision_blind=vb_parsed.get("is_logo"), vlm_brand_blind=vb_parsed.get("brand"),
                vlm_brand_blind_canonical=can.get("vlm_brand_blind_canonical"),
                blind_hinted_brand_agree=can.get("blind_hinted_brand_agree"),
                hint_composition=hc.get("hint_composition", []),
                hinted_brand_in_faiss_hints=can.get("hinted_brand_in_faiss_hints"),

                canonical_brand=canonical_brand, brand_canonical=canonical_brand,
                canonical_match_mode=can.get("canonical_match_mode"), canonical_match_score=can.get("canonical_match_score"),
                brand_list_version=can.get("brand_list_version", ctx.brands.version),
                brand_list_hash=can.get("brand_list_hash", ctx.brands.content_hash),

                channel=manifest.get("channel"), channel_brand_canonical=manifest.get("channel_brand_canonical"),
                watermark_regions_available=wm.get("watermark_regions_available", False),
                watermark_overlap=wm.get("watermark_overlap"), watermark_overlap_ratio=wm.get("watermark_overlap_ratio"),

                brand_evidence=brand_evidence,
                has_conflicting_ocr=bool(canonical_brand and ocr_brand_match and ocr_brand_match != canonical_brand),
                strong_visual_match=bool(faiss_top1_score is not None
                                          and faiss_top1_score >= cfg.faiss.rule14_top1_threshold),
                audio_only_prediction=bool(can.get("brand_from_audio") and not faiss_confirmed and not ocr_confirmed),
                faiss_only_prediction=bool(faiss_confirmed and not can.get("brand_from_audio") and not ocr_confirmed),
                router_decision=None, router_reason=None, reason_code=None, router_rule=None,
                router_confidence=None, risk_score=None, trust_score=None, review_priority=None,
                dedup_cluster_id=None, dedup_representative=None, rules_matched_preempted=[],

                config_hash=manifest["config_hash"], experiment=manifest.get("experiment") or manifest["run_id"],
                brand_mode=cfg.vlm.brand_mode, enabled_signals=vh.get("enabled_signals", []),
                label=None, enrolled_to_faiss=False,
            )
            rows.append(asdict(row))

        ctx.write_jsonl(ROUTER_INPUT_FILE, rows, header=ctx.header("router_input"))
        return StageResult(counts={"rows": len(rows)})

"""``stages/refine.py`` (T20-T24, ``spec docs/09``): heuristic 1
(provenance, T20), heuristic 2 (OCR-guided tightening, T21) and heuristic 5
(oversized-box guard, T22). Heuristics 3/4 are out of scope
(``heuristics_3_4_enabled`` has no other keys — 05 lock). Heuristic 6 lives
in the router dedup (T45).

Crop *files* are cut and persisted by ``stages/output.py`` (05 section C:
every path field there is ``Stage: output``, and it runs after the router so
crops can't exist yet here); this stage only computes the section-B geometry
fields and cuts an in-memory crop of the loose box to feed the cheap
heuristic-2 OCR call, sharing ``stages/ocr.py``'s reader singleton so the
verification pass (MM4) doesn't load a second copy of the model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..brands import BrandList
from ..brands import normalise as brand_normalise
from ..crops import cut_crop
from ..geometry import box_area, box_area_ratio
from .base import AUDIO_HINTS_FILE, MERGED_DETECTIONS_FILE, REFINEMENT_FILE, RunContext, Stage, StageResult
from .ocr import get_reader, resolve_gpu


def _axis_aligned(quad) -> list[float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _pad_box(box: list[float], ratio: float) -> list[float]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]


def _clamp_to(box: list[float], bounds: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = bounds
    return [max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)]


def _match_audio(norm: str, audio_brands: list[str], threshold: int, short_cutoff: int) -> tuple[str | None, int]:
    """T21's ``match_audio``: the same length-dependent matching
    ``BrandList.map_fuzzy`` uses, but against the free-text brand strings
    Whisper+Qwen pulled from the transcript rather than the canonical list —
    audio brands are candidates too (canonicalisation's own
    ``audio_brands_as_targets``, T37)."""
    if not norm or not audio_brands:
        return None, 0
    if len(norm) <= short_cutoff:
        for b in audio_brands:
            if brand_normalise(b) == norm:
                return b, 100
        return None, 0
    from rapidfuzz import fuzz, process
    keys = [brand_normalise(b) for b in audio_brands]
    best = process.extractOne(norm, keys, scorer=fuzz.WRatio, processor=None)
    if best is None:
        return None, 0
    _, score, idx = best
    score_i = int(round(float(score)))
    return (audio_brands[idx], score_i) if score_i >= threshold else (None, 0)


def ocr_guided_candidate(reader, loose_crop: Image.Image | None, loose_box: list[float], h2cfg, ocfg,
                          brands: BrandList, audio_brands: list[str]) -> tuple[list[float] | None, str, list[dict]]:
    """T21: the cheap in-module OCR call on the **loose** crop (not the
    tight/selected crop MM4's verification pass reads), upscaled when its
    min side is below ``ocr.upscale_min_side``, keeping quadrilaterals
    (unlike ``stages/ocr.py``'s ``ocr_image``, which only keeps
    text+confidence). Returns ``(candidate_box_in_frame_coords | None,
    reason, ocr_boxes_log)``.
    """
    if loose_crop is None:
        return None, "no_crop", []

    img = loose_crop.convert("RGB")
    scale = 1
    if min(img.size) < ocfg.upscale_min_side:
        scale = ocfg.upscale_factor
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    results = reader.readtext(np.array(img))

    ocr_boxes_log: list[dict] = []
    best: tuple[object, str, int] | None = None
    for quad, text, conf in results:
        text = text.strip()
        conf = round(float(conf), 3)
        norm = brand_normalise(text)
        matched, score, mode = None, 0, "unmatched"
        if conf >= h2cfg.ocr_min_confidence and norm:
            matched, score, mode = brands.map_fuzzy(norm, h2cfg.brand_match_min_score, ocfg.matching.short_name_cutoff)
            if matched is None:
                matched, score = _match_audio(norm, audio_brands, h2cfg.brand_match_min_score,
                                               ocfg.matching.short_name_cutoff)
                mode = "audio" if matched is not None else mode
        ocr_boxes_log.append({"quad": [[float(x), float(y)] for x, y in quad], "text": text, "confidence": conf,
                               "matched_brand": matched, "match_score": score, "match_mode": mode})
        if matched is not None and (best is None or score > best[2]):
            best = (quad, matched, score)

    if best is None:
        return None, "no_matching_text", ocr_boxes_log

    quad, matched, score = best
    box = _axis_aligned(quad)
    box = [c / scale for c in box]
    box = [box[0] + loose_box[0], box[1] + loose_box[1], box[2] + loose_box[0], box[3] + loose_box[1]]
    box = _pad_box(box, h2cfg.padding_ratio)
    box = _clamp_to(box, loose_box)
    return box, f"matched:{matched}:{score}", ocr_boxes_log


def passes_safety(candidate: list[float], loose: list[float], h2cfg) -> tuple[bool, dict]:
    """T21's safety checks, thresholds from
    ``refinement.heuristic_2_ocr_tightening`` (04)."""
    cx1, cy1, cx2, cy2 = candidate
    lx1, ly1, lx2, ly2 = loose
    inside_loose = cx1 >= lx1 - 1e-6 and cy1 >= ly1 - 1e-6 and cx2 <= lx2 + 1e-6 and cy2 <= ly2 + 1e-6
    w, h = cx2 - cx1, cy2 - cy1
    min_side_px = min(w, h) >= h2cfg.min_side_px
    area_loose = box_area(loose)
    ratio = (box_area(candidate) / area_loose) if area_loose > 0 else 0.0
    area_ratio_vs_loose = h2cfg.min_area_ratio_vs_loose <= ratio <= h2cfg.max_area_ratio_vs_loose
    aspect = (max(w, h) / min(w, h)) if min(w, h) > 0 else float("inf")
    max_aspect = aspect <= h2cfg.max_aspect
    checks = {"inside_loose": inside_loose, "min_side_px": min_side_px,
              "area_ratio_vs_loose": area_ratio_vs_loose, "max_aspect": max_aspect}
    return all(checks.values()), checks


class RefineStage(Stage):
    name = "refine"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [REFINEMENT_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.cfg.refinement
        ocfg = ctx.cfg.ocr
        h2cfg = cfg.heuristic_2_ocr_tightening
        h5cfg = cfg.heuristic_5_oversized_guard
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)

        audio_brands: list[str] = []
        if ctx.path(AUDIO_HINTS_FILE).exists():
            audio_brands = ctx.read_json(AUDIO_HINTS_FILE).get("brands", [])

        reader = get_reader(ocfg.languages, resolve_gpu(ocfg.gpu)) if cfg.enabled and h2cfg.enabled else None
        frame_cache: dict[int, Image.Image] = {}

        rows: list[dict] = []
        for rep in sorted(reps, key=lambda r: r["det_id"]):
            loose = rep["loose_bbox"]
            w, h = rep["frame_width"], rep["frame_height"]
            ratio = box_area_ratio(loose, w, h)
            assert abs(ratio - rep["box_area_ratio"]) < 1e-6, \
                f"det_id {rep['det_id']}: refine's box_area_ratio disagrees with merge's (T22)"
            oversized = cfg.enabled and h5cfg.enabled and ratio >= h5cfg.area_ratio

            candidate, reason, ocr_boxes_log, safety = None, "disabled", [], None
            if cfg.enabled and h2cfg.enabled:
                if rep["frame_idx"] not in frame_cache:
                    frame_cache[rep["frame_idx"]] = ctx.open_frame_image(rep["frame_idx"])
                loose_crop = cut_crop(frame_cache[rep["frame_idx"]], loose)
                candidate, reason, ocr_boxes_log = ocr_guided_candidate(
                    reader, loose_crop, loose, h2cfg, ocfg, ctx.brands, audio_brands)

            if candidate is not None:
                safe, safety = passes_safety(candidate, loose, h2cfg)
                if safe:
                    status, refined, method = "applied", candidate, "ocr_guided"
                else:
                    status, refined, method = "rejected_safety_check", None, "ocr_guided"
            elif oversized:
                status, refined, method = "oversized_region", None, "oversized_guard"
            else:
                status, refined, method = "no_candidate", None, "none"
            if oversized and status != "applied":                # the guard wins over rejected/no_candidate
                status = "oversized_region"

            selected = refined if status == "applied" else loose
            refinement_area_ratio = (round(box_area(refined) / box_area(loose), 6)
                                      if status == "applied" and box_area(loose) > 0 else None)

            rows.append({
                "det_id": rep["det_id"],
                "refined_bbox": refined,
                "refinement_method": method,
                "refinement_status": status,
                "selected_bbox": selected,
                "has_valid_tight_bbox": status == "applied",
                "refinement_area_ratio": refinement_area_ratio,
                "box_area_ratio": round(ratio, 6),
                "oversized": oversized,
                "candidate_bbox": candidate,
                "candidate_reason": reason,
                "safety_results": safety,
                "ocr_boxes": ocr_boxes_log,
            })

        header = ctx.header("refine", heuristic_1_provenance=cfg.heuristic_1_provenance,
                             heuristic_2_enabled=cfg.enabled and h2cfg.enabled,
                             heuristic_5_enabled=cfg.enabled and h5cfg.enabled,
                             heuristic_5_area_ratio=h5cfg.area_ratio)
        ctx.write_jsonl(REFINEMENT_FILE, rows, header=header)

        applied = sum(1 for r in rows if r["refinement_status"] == "applied")
        return StageResult(counts={
            "rows": len(rows), "applied": applied,
            "rejected_safety_check": sum(1 for r in rows if r["refinement_status"] == "rejected_safety_check"),
            "oversized_region": sum(1 for r in rows if r["refinement_status"] == "oversized_region"),
            "no_candidate": sum(1 for r in rows if r["refinement_status"] == "no_candidate"),
            "fallback_rate": round(1 - applied / len(rows), 4) if rows else 0.0,
        })

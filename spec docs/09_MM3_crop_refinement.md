# 09 — MM3 Crop Refinement (T20–T24)

**Effort:** 2 days. **Device:** CPU (EasyOCR runs on CPU acceptably for the cheap call; GPU if present). **Gates:** MM4 (OCR verification runs on `selected_bbox`), Rule 4 tests (T22).

Scope locked: heuristics 1 (provenance), 2 (OCR-guided tightening) and 5 (oversized-box guard). Heuristics 3 and 4 are not built and have no YAML keys beyond `heuristics_3_4_enabled: false`. Heuristic 6 lives in the router dedup (T45).

## Stage contract

`stages/refine.py` reads `merged_detections.jsonl`, `frames/`, `audio/audio_hints.json` (for the brand candidates) and the brand list; writes `refinement.jsonl` with the section-B fields of `05` and cuts `crops/det_XXXXXX/{crop.jpg, crop_enlarged.jpg, loose.jpg}` from `selected_bbox` (tight), 1.5× `selected_bbox` (enlarged, M2 `enlarge_factor`) and `loose_bbox`.

Order per detection:

```
loose = rep.loose_bbox
oversized = area(loose)/frame_area >= cfg.heuristic_5.area_ratio          # T22
candidate, method, reason = ocr_guided_candidate(loose, frame, cfg)         # T21; may be None
if candidate is not None and passes_safety(candidate, loose, cfg):
    status = "applied"; refined = candidate; refinement_method = "ocr_guided"
elif candidate is not None:
    status = "rejected_safety_check"; refined = None; refinement_method = "ocr_guided"
elif oversized:
    status = "oversized_region"; refined = None; refinement_method = "oversized_guard"
else:
    status = "no_candidate"; refined = None; refinement_method = "none"
if oversized and status != "applied": status = "oversized_region"           # the guard wins over rejected/no_candidate
selected = refined if status == "applied" else loose
has_valid_tight_bbox = (status == "applied")                                # T23
```

`fallback_to_loose` (the 18 July status) is not a separate status; it is `status != "applied"`, and the T24 fallback rate is its share.

## T20 — Heuristic 1, provenance

Every row carries `loose_bbox`, `refined_bbox` (nullable), `refinement_method`, `refinement_status`, `selected_bbox`, `refinement_area_ratio`. `loose_bbox` is always the original representative box; it never becomes a union (29 July Open 2, resolved here as the safer reading — **[ASSUMPTION]** listed in `19`).

## T21 — Heuristic 2, OCR-guided tightening

The cheap in-module OCR call (A9): `reader.readtext(np.array(crop))` on the **loose crop**, upscaled ×3 when its min side is below 80 px (M2 rule), returning `(quad, text, conf)` triples. The reader is the shared singleton from `stages/ocr.py`; this is the one extra inference per detection the correspondence budgets.

```
def ocr_guided_candidate(loose, frame, cfg):
    crop = cut(frame, loose); scale = upscale_factor if small else 1
    boxes = reader.readtext(crop)                                    # keeps quadrilaterals (unlike M2's ocr_image)
    best = None
    for quad, text, conf in boxes:
        if conf < cfg.ocr_min_confidence: continue
        norm = brands.normalise(text)
        canonical, score, mode = brands.map_fuzzy(norm, cfg.brand_match_min_score, cfg.short_name_cutoff)
        if canonical is None: canonical, score = match_audio(norm, audio_brands)   # audio brands are candidates too
        if canonical is None: continue
        if best is None or score > best.score: best = (quad, canonical, score)
    if best is None: return None, "none", "no_matching_text"
    box = axis_aligned(best.quad) / scale + loose[:2]                # back to frame coordinates
    box = pad(box, cfg.padding_ratio); box = clamp(box, loose)       # must stay inside the loose box
    return box, "ocr_guided", f"matched:{best.canonical}:{best.score}"
```

Safety checks (`passes_safety`, all from the 18 July candidate-validation list, thresholds in `04`):

- inside the loose box (after clamp, trivially true; recorded anyway);
- `min_side_px` ≥ 12;
- `min_area_ratio_vs_loose` ≤ area(candidate)/area(loose) ≤ `max_area_ratio_vs_loose` (0.05–0.95: must shrink by at least 5 % and keep at least 5 %);
- aspect ratio ≤ `max_aspect` (8).

Everything the call saw is logged in `refinement.jsonl`: `ocr_boxes` (quad, text, conf), `candidate_bbox`, `safety_results` per check, so the fallback rate can be decomposed by cause in T24.

## T22 — Heuristic 5, oversized-box guard

`area(loose_bbox) / frame_area >= refinement.heuristic_5.area_ratio` (0.45). Produces `box_area_ratio` (also written by the merge for convenience; the refine stage recomputes and asserts equality) and, when no valid tight box exists, `refinement_status = oversized_region`. Rule 4 reads `box_area_ratio` and `has_valid_tight_bbox`; the router's own `rule4_area_ratio` key defaults to the same 0.45 so the two can be tuned apart later without a code change.

## T23 — `has_valid_tight_bbox`

Boolean, `refinement_status == "applied"`. Written here, read by Rule 4 and Rule 18.

## T24 — Fallback rate

`eval/refinement_report.py --run <dir>` → `reports/refinement.md`: total rows; share by `refinement_status`; fallback rate = 1 − share(applied); breakdown of `rejected_safety_check` by failing check; `oversized_region` count; mean `refinement_area_ratio` for applied rows; the number of applied rows where the OCR verification pass (MM4) found a brand on the tight crop but not on the loose crop, and vice versa (the "OCR-brand fuzzy match before vs after refinement" metric from 18 July, computed once MM4 exists). Reported on the first full run; if the fallback rate is very high (say > 90 %) the report says the module is inert, as the client asked.

## Definition of done

- [ ] `refinement.jsonl` written for every representative; statuses drawn from the enum; `selected_bbox` never null.
- [ ] Unit tests with synthetic frames: (a) text box inside a loose box → applied; (b) text box that would shrink to 2 % → rejected_safety_check; (c) 60 % frame box with no text → oversized_region; (d) no text → no_candidate.
- [ ] Crops cut from `selected_bbox`; `loose.jpg` present when it differs.
- [ ] Refinement report generated for the MM1 acceptance run (video_3-trimmed).

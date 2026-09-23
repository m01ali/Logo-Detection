# 12 — MM6 Watermark (T47–T50)

**Effort:** 1 day. **Device:** CPU. **Gates:** none downstream; Rule 5 (MM5) reads `watermark_overlap`. T47 data trails the fresh-video delivery; T48–T50 are built and tested against an empty region set and against hand-measured regions for `video_1` (Rai Sport bug, top-left) so the checkpoint exercises the rule.

## T47 — Per-video region in the run config

`configs/videos/<video_id>.yaml` carries `channel`, `channel_brand` and `watermark_regions_ref`. The regions themselves live in the keyed store (T50) and are normalised `[x1, y1, x2, y2]` fractions of frame width/height so they survive the three `fixed-images` size families. Several regions per key are allowed (bugs move during replays and split screens; 27 July). Determining the numbers is one look at one frame; for `video_1` the Rai Sport bug is measured once and committed as `rai_sport_default` so the checkpoint can show Rule 5 working.

## T48 — `watermark_overlap` as a pre-router step

`stages/watermark.py` reads `refinement.jsonl` (uses `loose_bbox`, the box the VLM saw in context, **[ASSUMPTION]**: `selected_bbox` may have been tightened onto text inside a larger region and would under-report overlap) and the regions; writes `watermark.jsonl`:

```
def overlap(box, regions, W, H, margin, t):
    if not regions: return None, None, False
    best = 0.0
    for r in regions:
        R = expand(denormalise(r, W, H), margin*W, margin*H)
        inter = intersection_area(box, R)
        ratio = inter / min(area(box), area(R))            # containment, not IoU (27 July)
        best = max(best, ratio)
    return best >= t, best, True                            # (watermark_overlap, watermark_overlap_ratio, regions_available)
```

`margin` v1 0.02, `containment_threshold` v1 0.50. A video with no regions yields `watermark_overlap = None`, `watermark_regions_available = False` — the null is deliberate so Rule 5 cannot treat "no data" as "outside the region" (the 29 July defaults hazard).

## T49 — Rule 5, three-branch v1 form (client 12 August B1), REJECT variant disabled

Evaluated in block 3 (see `11`). Let `chan = canonical_brand in known_channel_brands`.

| Branch | Condition | Outcome |
|--------|-----------|---------|
| No regions | `chan and watermark_overlap is None and ocr_fuzzy_score < rule5_ocr_fuzzy(90)` | REVIEW `channel_brand_needs_watermark_check` (the 29 July form) |
| Inside | `chan and watermark_overlap is True` | no decision from Rule 5; normal handling continues |
| Outside, OCR supports | `chan and watermark_overlap is False and ocr_fuzzy_score >= 90 and ocr_brand_c == canonical_brand` | no decision; log `rule5_branch = outside_ocr_supported` for T71 |
| Outside, weak OCR | `chan and watermark_overlap is False and not (previous)` | REVIEW `channel_brand_outside_watermark_region`; the reason payload carries `ocr_text`, `ocr_fuzzy_score`, `faiss_top1`, `faiss_top1_score`, `brand_from_audio` (all already on the row; T73 shows them) |
| REJECT variant | `chan and watermark_overlap is False and brand_from_audio and ocr_fuzzy_score < 70` | REJECT `likely_channel_audio_contamination` — **config-disabled for all of M3** |

`known_channel_brands` = the `channel_brands` list of the referenced region entry if present, else the `channel: true` brands of the snapshot. `rule5_branch` is written on every row where `chan` holds so `eval/watermark_report.py` (T71) can count how often a channel-brand prediction falls outside the region with strong OCR support.

## T50 — Keyed region store (shape only)

`configs/watermark_regions.yaml` as in `04`: `{key: {channel_brands: [...], regions: [[x1,y1,x2,y2], ...]}}`. A video references a key; no lookup by channel, no reuse logic, no UI. The loader validates that every referenced key exists and that coordinates are within [0, 1] with `x1 < x2`, `y1 < y2`.

## Definition of done

- [ ] `watermark.jsonl` written; null when no regions.
- [ ] Unit tests: box fully inside region → True; box 1 % outside but within margin → True; box elsewhere → False; no regions → None.
- [ ] Rule 5 branch tests for all five rows of the table, including that the REJECT branch never fires when disabled even with `watermark_overlap False`.
- [ ] `rai_sport_default` regions measured on `video_1` frame 0 and committed; the MM1 acceptance run re-executed `--from-stage watermark --to-stage router` shows `channel_brand_outside_watermark_region` rows if any exist.

# 08 — MM2 Detection and Merge (T12–T19)

**Effort:** 2 days. **Device:** GPU for T12 (YOLO); CPU for T13–T19. **Gates:** MM3 (refinement reads representatives), MM7 (counters), MM11 (ablation).

## T12 — YOLO alongside Grounding DINO in a common record

1. `stages/detect_yolo.py`: `ultralytics.YOLO(cfg.detectors.yolo.weights)`, `predict(source=frame, imgsz, conf, iou=iou_nms, agnostic_nms=True, verbose=False)` per sampled frame. `agnostic_nms=True` is what "class-agnostic" means operationally: boxes of different predicted classes suppress each other, so one physical logo yields one box.
2. For each box write a `DetectionRecord` with `source: yolo`, `score = conf`, `label = class name`, `class_conf = conf` (ultralytics returns one score per box; the class is `int(cls)` → `names[cls]`). The class drives nothing downstream (`yolo_class` on the row).
3. Same `min_crop_px` drop as GDINO, applied identically to both sources so the counters compare like with like.
4. Frame coordinates: ultralytics letterboxes internally and returns boxes in original pixels (`boxes.xyxy`); no rescaling needed (11 August A1).

**Check:** on `video_3-trimmed` (340 frames, all processed by the M2 predict run under `runs/detect/yolo results/…/labels/`) the per-frame box counts from the new stage match the M2 label files for the four sampled frames after converting `cx cy w h` to `xyxy`.

## T13 — Merge stage: pooled greedy over the union

Pure function `merge_frame(raw: list[DetectionRecord], frame_w, frame_h, cfg) -> (reps, pair_log)` in `stages/merge.py`, called per frame over `raw_detections.jsonl`; nothing else is read.

```
def merge_frame(raw, W, H, cfg):
    boxes = sorted(raw, key=lambda d: d.score, reverse=True)      # deterministic tie-break: raw_det_id
    remaining = list(boxes); reps = []; log = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        for other in list(remaining):
            m = evaluate_pair(seed, other, W, H, cfg)             # see below
            log.append(m)
            if m.outcome == "merged":
                cluster.append(other); remaining.remove(other)
        rep = min(cluster, key=area)                              # locked: keep the tighter box
        reps.append(Representative(
            loose_bbox=rep.box_xyxy, det_score=rep.score,
            detection_sources={d.source for d in cluster},
            merge_count=len(cluster) - 1,
            merged_raw_det_ids=[d.raw_det_id for d in cluster if d is not rep],
            seed_raw_det_id=seed.raw_det_id, gdino_label=…, yolo_class=…))
    return reps, log
```

`evaluate_pair(a, b)`:

```
inter      = intersection_area(a, b)
iou        = inter / (area_a + area_b - inter)
contain    = inter / min(area_a, area_b)
area_ratio = min(area_a, area_b) / max(area_a, area_b)
smaller_fraction = min(area_a, area_b) / (W*H)

if smaller_fraction < cfg.min_merge_area:          outcome = "skipped_area_floor"     # T14
elif not (iou >= t_iou or contain >= t_contain):   outcome = "no_overlap"
elif area_ratio < t_area_ratio:                    outcome = "failed_size_precondition" # T19 population
else:                                              outcome = "merged"
```

Pair type: `intra` when `a.source == b.source`, else `cross`. Every evaluated pair is logged with `iou, contain, area_ratio, smaller_fraction, pair_type, outcome` (T19, and the 27 July "log every merge evaluation").

Notes fixed by the correspondence:

- Comparisons are against the seed (greedy, NMS-style, no transitive clustering — 27 July C0).
- "Overlap" is IoU ≥ 0.70 **or** containment ≥ 0.90; "near-equal size" is area ratio ≥ 0.80; both required (Revision 3 T13).
- The representative keeps its **own** detector score as `det_score`; the seed's identity is logged separately so a reviewer can see which box won.
- `detection_sources` is a set; `both` in the 27 July sense is `{gdino, yolo}` (T15).
- `det_id` is assigned to representatives in `(frame_idx, seed order)` after the merge; raw ids remain in `merged_raw_det_ids`.
- **[ASSUMPTION]** A size-aware small-box branch (centre-distance below a pixel-area cutoff), requested on 27 July and proposed on 29 July, is not in Revision 3. It is implemented as an optional branch inside `evaluate_pair` behind `merge.small_box.enabled: false` so the pure function accommodates it later; it is not tuned in M3. Listed in `19`.

## T14 — `min_merge_area`

Fraction of frame area, tested on the smaller box of the pair, v1 placeholder 0.0004. `02` §2.2 shows 0.8 % of gold boxes below it and 3.8 % below 0.001, so the first-run T19 distributions (which report `smaller_fraction` for the `skipped_area_floor` population) are compared against those percentiles; the client's steer is to move it up, not down, if ambiguous.

## T15 — `detection_sources` and `merge_count`

Written by T13. JSON list in the CSV (sorted for determinism), integer `merge_count`. `merged = merge_count > 0` is not a stored column; scripts derive it.

## T16 — The `merge_enabled: false` path

Implemented in MM1 Step 4 and kept: with the flag false the stage emits one representative per raw box, `merge_count 0`, writes an empty `merge_log.jsonl` and counters with `pairs_evaluated 0`. The T18 counters and T60 harness must run identically on either path; the ablation (MM11) is two runs of `--from-stage merge` over the same `raw_detections.jsonl`.

## T17 — Legacy filters stay out

No 500 px² minimum, no 0.5 maximum area ratio anywhere in `logo_pipeline/`. The YAML has no key for them. The only size filter is `detectors.*.min_crop_px: 16`, which is the M2 extraction-time filter, applied at detection, not in the merge. A unit test asserts that a 30×30 box on a 1920×1080 frame and a box covering 60 % of the frame both survive the merge stage.

## T18 — Merge counters

`merge_counters.json`:

```json
{
  "video": {
    "boxes_gdino": 0, "boxes_yolo": 0, "boxes_total_raw": 0,
    "pairs_evaluated": 0, "pairs_merged_intra": 0, "pairs_merged_cross": 0,
    "pairs_skipped_area_floor": 0, "pairs_failed_size_precondition": 0, "pairs_no_overlap": 0,
    "boxes_remaining": 0,
    "remaining_by_sources": {"gdino": 0, "yolo": 0, "gdino+yolo": 0},
    "vlm_calls_saved": 0                  // boxes_total_raw − boxes_remaining
  },
  "frames": { "<frame_idx>": { …same keys…, "sampled": true } }
}
```

Frames with zero boxes appear with zeros (they are sampled frames and T58 draws them).

## T19 — Distributions for pairs that failed the precondition

`eval/merge_ablation.py --run <dir>` reads `merge_log.jsonl` and writes `reports/merge_distributions.md` + `.csv`: histograms (20 bins) of `iou`, `contain`, `area_ratio` for the `failed_size_precondition` population and of `smaller_fraction` for the `skipped_area_floor` population, plus the share of overlapping pairs skipped by the floor and the extra VLM calls that implies (= `pairs_skipped_area_floor` that would otherwise have merged). This is the evidence for re-setting the thresholds and `min_merge_area` after the first full run.

## Definition of done

- [ ] YOLO stage produces boxes in frame pixels; class logged; agnostic NMS on.
- [ ] Merge is a pure function with unit tests: identical boxes merge; tight-vs-padded at area ratio 0.4 does not; contained tiny box at ratio 0.05 does not; floor skips a 15×15 box pair on 1920×1080 (fraction 0.00006 < 0.0004); greedy order is deterministic.
- [ ] `merge.enabled: false` produces the same file set.
- [ ] Counters and distributions generated for one run and read by the T60 harness.
- [ ] No legacy filter constants anywhere (grep for `500` and `0.5` in the stage).

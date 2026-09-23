# 17 — MM11 Evaluation (T61–T66, T71)

**Effort:** 2 days of build. **Device:** CPU for all scripts; GPU only for route-three pipeline runs. **Inputs:** the `video_1` full run (MM9), the client label file (route two), `fixed-images` + Roboflow labels (route three).

## T61 — Router-only runner

`python -m logo_pipeline router-only --config <yaml> --run <run_dir> [--input router_input.jsonl] [--out reports/router_<tag>/]`: loads the rows, rebuilds `RouterInput`, runs the MM5 engine and dedup, writes `router_output.jsonl`, `firing_log.jsonl`, and a `rows.csv` copy with the new decisions. Deterministic (pure functions, sorted iteration). The corpus is the `video_1` M3 run; the 1,739 rows are not used (Revision 3). A YAML diff and the two output files are enough to show what a rule change did.

## T62 — Metric scripts

`eval/metrics.py --run <dir> [--labels labels.csv] [--wrong-brand-as fp|separate]`:

Labels file schema (the review UI's export satisfies it): `det_id, video_id, run_id, label ∈ {TP, FP, FN, wrong_brand}, true_brand (optional)`. Rows join on `(video_id, run_id, det_id)`.

Outputs (`reports/metrics.md` + CSVs):

- decision distribution (ACCEPT / REVIEW / REJECT / UNKNOWN_LOGO / DEDUPED), counts and shares; review load = REVIEW + UNKNOWN_LOGO share;
- per-rule firing counts (deciding rule) and per-reason-code counts; pre-empted counts from the firing log;
- per-reason-code precision on labelled rows: `TP / (TP + FP [+ wrong_brand])`, with n;
- ACCEPT precision (the same on the ACCEPT subset) stated on its own line with n;
- `brand_evidence` distribution overall and by decision;
- `canonical_match_mode` distribution; `refinement_status` distribution; Rule 5 branch counts.

Without labels the script prints everything that does not need them (this is the "decision distribution" published after the `video_1` run).

## T63 — IoU matching for the end-to-end evaluation

`eval/iou_match.py --run <dir> --gold <roboflow labels dir or fresh-frame labels csv> --iou 0.5`:

- gold boxes denormalised against the row's `frame_width/height`; detections use `loose_bbox` (the proposal), with a second pass on `selected_bbox` reported for the refinement question;
- greedy one-to-one matching by descending IoU per frame; three counts: matched, gold boxes never re-detected, detections with no gold label; per class and with/without channel classes;
- router precision **on the matched subset only**: decision vs gold (a matched box is a true logo; `canonical_brand` vs gold class through the alias table gives brand correctness); ACCEPT precision on matched ACCEPTs; UNKNOWN_LOGO rows whose gold class is in the list are counted as "known brand routed unknown";
- recall on the labelled fresh frames (A6, 30–50 frames): matched / gold, per detector source and with the merge on/off (the merge ablation's recall column). Detector recall from the Roboflow gold set is **not** reported (out of scope, no FN by construction).

## T64 — Blind-versus-hinted agreement and FAISS-origin attribution

`eval/blind_vs_hinted.py --run <video_1 run>` → `reports/blind_vs_hinted.md`: decision agreement matrix (hinted × blind `is_logo`); brand agreement rate among rows where both say logo/partial (on `canonical_brand` vs `vlm_brand_blind_canonical`); share of hinted rows whose brand appears in the FAISS hints shown; share of those where the blind brand differs (the hint plausibly moved the answer); the same for audio and OCR hints for symmetry; how often blind says UNKNOWN where hinted names a brand. No labels required. The accuracy comparison (which of the two is right) runs on the route-two labels when they arrive and is a second section of the same report.

## T65 — Ordering verification checks

`eval/ordering_checks.py --run <dir>`: the two checks of `11` Part G with the offending `det_id`s if any. Also run automatically at the end of every `run` and at T77.

## T71 — Channel-brand predictions outside the watermark region with strong OCR support

`eval/watermark_report.py --run <dir>`: counts of `rule5_branch` values (`no_regions`, `inside`, `outside_ocr_supported`, `outside_weak_ocr`) and the share of channel-brand predictions outside the region with `ocr_fuzzy_score ≥ 90`. On the `video_1` run (Rai Sport regions measured in MM6) and on any fresh video with regions.

## T66 — Acceptance-gate corpus

### Route two (router-only gate)

`eval/label_request.py --run <video_1 run> --per-code 70 --seed 42` samples deterministically: every ACCEPT, every UNKNOWN_LOGO, and `per-code` rows (60–80) for each remaining reason code that fired at least that often (all rows for codes that fired fewer times, stated separately). Writes `label_request/labels_to_fill.csv` (`det_id, frame_idx, timecode_s, crop_path, enlarged_crop_path, annotated_frame_path, canonical_brand, router_decision, reason_code, label, true_brand`) and a folder of the crops and annotated frames, plus a note stating the count against the published distribution. The client labels in the review UI or the CSV; `eval/metrics.py` consumes it.

### Route three (end-to-end, on `fixed-images`)

Because the gold frames are held (`02` §3), route three is:

1. `run --frames-dir fixed-images --video-id <v>` with the per-video file `configs/videos/fixed_<v>.yaml` (`process_all_frames: true` so every labelled ordinal is processed; `assumed_fps 1.0`; audio unavailable; regions if known).
2. `eval/iou_match.py` against the Roboflow labels of that video (T63).
3. The sampling-rate check (`eval/sampling_rate_check.py`) runs only when a source **video** arrives: extract at the believed rate, perceptual-hash (pHash) frame N against `fixed-images/{v}_frame_{N}.png`, sweep candidate rates if it fails; the confirmed rate then rewrites `timecode_s` on the frames-mode run (`--from-stage output`). Route-three matching does not wait on it.

GPU budget is the real constraint. Tiers, hinted pass only (T64 does not need the blind pass on route three):

| Tier | Videos (labelled frames) | Approx. detections | GPU hours at 13.8 s |
|------|--------------------------|--------------------|---------------------|
| 1 | 14 (28), 1 (37) | ~1,000 (dense sport) | ~4 |
| 2 | + 27 (150) | +~2,000 | ~12 |
| 3 | + 98 (185), 100 (128) | +~4,000 | ~27 |

Tier 1 is committed; tiers 2–3 run as GPU time allows and are reported per video. This replaces the Revision 3 wording "on arrival of the first of these [videos]"; the client should be told (`19`).

## Deliverables produced by MM11 (mapping to Revision 3 §7)

| Deliverable | Script / file |
|-------------|---------------|
| Per-rule firing counts, precision by reason code, ACCEPT precision, decision distribution, review load, brand_evidence distribution | `eval/metrics.py` → `reports/metrics.md` |
| Crop refinement fallback rate | `eval/refinement_report.py` |
| Share of pairs skipped by `min_merge_area`, extra VLM calls; merge ablation | `eval/merge_ablation.py` (two runs of `--from-stage merge`) |
| Recall on labelled fresh frames | `eval/iou_match.py` |
| FAISS self-match report; re-benchmark | `eval/faiss_selfmatch.py`, `eval/faiss_rebenchmark.py` |
| Blind-vs-hinted with FAISS-origin attribution | `eval/blind_vs_hinted.py` |
| Rule 5 outside-region-with-OCR figure | `eval/watermark_report.py` |
| Two ordering checks | `eval/ordering_checks.py` |
| Reason-code enum | `configs/reason_codes.md` |
| Annotated frames and video; review-UI loading with queues; enrolment with provenance | MM7 |
| Received-artefacts record | `docs/received_artefacts.md` |
| Acceptance-evidence caveats | `reports/acceptance_evidence.md`: Rule 6 single-signal caveat; YOLO 321-class model run class-agnostic, trained on a random split of the same labelled set; the M2 prompt variant for video_1 unknown; plus, from these specs: Rules 2/3 unreachable without the enhanced fields; `not_logo_confidence` source; audio unavailable in frames mode |

## Definition of done

- [ ] `router-only` reproduces the pipeline's `router_output.jsonl` byte-for-byte on the same YAML.
- [ ] Every script has a `--help`, writes markdown + CSV under `reports/`, and runs on the `video_1` run.
- [ ] Route-two request produced and sent with the distribution.
- [ ] Tier-1 route three run and matched; per-video T63 counts reported.
- [ ] Ordering checks pass on every run in `runs/`.

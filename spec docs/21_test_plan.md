# 21 — Test Plan

Tests live in `tests/` (pytest). Fixtures are small and synthetic wherever possible so the suite runs on the Mac without torch. GPU stages are covered by the recorded acceptance runs, not by unit tests.

## Fixtures

| Fixture | Content | Used by |
|---------|---------|---------|
| `tests/fixtures/brands_small.yaml` | 12 brands incl. `rai`, `rai sport`, `verisure` (aliases `very sure`, `verishore`), `hp`, `x`, `wolkswagen` (alias `volkswagen`) | MM0, MM4, MM5 |
| `tests/fixtures/frames/` | three synthetic 1280×720 PNGs: plain background; one with rendered text "NUROFEN IBUPROFENE"; one with a tiny corner mark | MM3, MM6 |
| `tests/fixtures/raw_detections_video1.jsonl` | converted from `extraction_output/video_1/manifest.json` (1,739 GDINO boxes, 102 frames) — **local only, git-ignored**; test skips when absent | MM2 integration, MM5 volume tests |
| `tests/fixtures/router_rows.jsonl` | ~40 hand-written `RouterInput` rows, one per rule branch | MM5 |
| `tests/fixtures/video_3_trimmed_rows.csv` | the M2 CSV (212 rows) | MM7 review-UI compatibility |
| `tests/fixtures/roboflow_labels_video14/` | the 28 label files for video 14 + a list of expected denormalised boxes for two frames | MM10, MM11 |

## MM0

- `test_brands_load_hash_stable`, `test_alias_collision_rejected`, `test_normalise_accents` (`L'Oréal` → `loreal`), `test_short_exact_rule` (`hb` vs `hp` → 0; `hp` → 100), `test_map_fuzzy_long` (`nurofenn` → `nurofen` ≥ 90), `test_channel_set`.
- `test_snapshot_has_321_entries` against the generated file.

## MM1

- Config: unknown key fails; hash changes with any value; per-video override merges; two-switch validation (`include_in_vlm_prompt: true` with `compute_and_log: false` is rejected).
- Frame sources: folder source orders ordinals numerically (`9_frame_0010` after `9_frame_0009`), reports `fps_is_assumed`; video source on `video_3-trimmed` returns 340 frames at 30 fps.
- Sampling: `interval 2.0` on a 30 fps, 6,067-frame source yields the 102 M2 indices; `per_shot` returns one frame per scene; `segment` matches M2.
- Detection record round-trip JSONL; header line written once.
- Schema: `CSV_COLUMNS` covers every `Row` field and nothing else; export script renders `05`.
- Idempotency: running a stage twice with the same hash is a no-op; `--force` re-runs.
- Acceptance run (recorded): `video_3-trimmed` `rows.csv` has 212 rows; M2 columns match names; review UI loads.

## MM2

- Geometry: `iou`, `containment`, `area_ratio` on hand-computed pairs.
- `evaluate_pair`: identical boxes → merged; tight vs padded (area ratio 0.4) → `failed_size_precondition`; 5 % contained box → `failed_size_precondition`; two 15×15 boxes on 1920×1080 → `skipped_area_floor`; non-overlapping → `no_overlap`.
- `merge_frame`: greedy order deterministic; representative is the smaller box; `detection_sources` union; `merge_count`; `merged_raw_det_ids`.
- `merge.enabled: false` produces one representative per raw box and empty log.
- Counters sum: `boxes_total_raw − boxes_remaining = Σ merge_count`.
- No constants `500` or `0.5` in `stages/merge.py` (a grep test).
- Integration (local fixture): merging the 1,739 GDINO boxes intra-detector runs in < 5 s and reports the counters.

## MM3

- The four synthetic cases in `09` (applied / rejected_safety_check / oversized_region / no_candidate).
- Candidate stays inside the loose box after padding and clamping.
- `selected_bbox` equals `refined_bbox` only when applied.
- Refinement report renders on the fixture run.

## MM4

- Normalisation table (`10`): `"RAI SPORT HD"` → `rai sport` via alias; `"very sure"` (audio) → `verisure`, list `canonical`; concatenation of lines beats single lines when the brand spans lines.
- Below-threshold match keeps `ocr_fuzzy_score` and nulls `ocr_brand_match`.
- Enlarged-crop fallback sets `ocr_source`.

## MM5

- One test per rule branch in `11` (≈ 30 tests), each asserting decision, reason code, and the five-value tuple shape.
- Engine: block order; first-match-wins; disabled rules skipped; `MissingRouterInput` raised when an enabled rule's field is absent (e.g. `rule_1_fallback_logprob` without scores); enhanced-field switch omits clauses.
- Ordering: `partial_logo` + `UNKNOWN` → Rule 9 (not 15); strong OCR logo → Rule 6 with 16 in `rules_matched_preempted` only if its condition held; T65 checks pass on the fixture.
- Dedup (`11` block 5) cases; shared function called by the merge with different keys gives different clustering on the same boxes.
- Canonicalisation: alias-first; fuzzy; audio target; `unmatched` retains normalised string; determinism (same input twice → identical output); LLM merge off by default.
- `configs/reason_codes.md` regenerates without diff from the enum.

## MM6

- Overlap cases (`12`), including margin and null-when-no-regions.
- Rule 5 five-row table; REJECT branch inert when disabled.
- Regions store validation (bad coordinates rejected; missing key rejected).

## MM7

- `rows.csv` column order equals `CSV_COLUMNS`; lists JSON-encoded; relative paths with forward slashes.
- Annotated frames: count equals sampled frames; an empty frame still gets an image with the footer.
- Video: duration ≈ frames × interval (± 1 s) — checked with `ffprobe` when available.
- Review UI: loads the M2 fixture CSV and the M3 CSV (smoke test via `streamlit.testing.v1.AppTest`); queue presets filter; Rule 5 panel appears for channel-brand rows; Accept-into-FAISS calls `enrol` (mocked DB) with a record carrying `video_id/run_id/det_id`.
- Harness report renders on the fixture run.

## T77

The checklist in `14`, executed by hand and recorded in `reports/checkpoint_checklist.md`.

## MM9 (GPU, recorded)

- Blind prompt contains no hint sections (unit test on the rendered text, no model needed).
- Spike report exists with the pass/fail statement; `scores.jsonl` line count equals representatives when enabled.
- Distribution report; router key flipped only on pass (assert in a config test that `scored` requires `scores.jsonl` present).

## MM10

- `EnrolmentRecord` round-trip; `enrol` writes N + augmentations records with `augmented_of`; vector indices consistent with `index.ntotal`.
- All eight FAISS-manager sites call `enrol` (a static test that `add_logos` is referenced only inside `faiss_store/enrol.py`).
- Denormalised Roboflow boxes for two video-14 frames match the expected list.
- Self-match and re-benchmark scripts run on a 20-crop subset with a tiny in-memory index (random vectors) and produce reports with both populations.

## MM11

- `router-only` output equals the pipeline's `router_output.jsonl` on the fixture run.
- `metrics.py` with a 20-row labels file gives hand-computed precision values; `--wrong-brand-as fp` changes them as expected.
- `iou_match.py` on two synthetic frames: matched / never-detected / no-gold counts correct; channel-class exclusion.
- `blind_vs_hinted.py` agreement matrix on a 10-row fixture.
- `label_request.py` deterministic with seed; respects the 60–80 band and the "every ACCEPT / every UNKNOWN_LOGO" rule.
- `sampling_rate_check.py`: pHash of a frame against itself passes; against a neighbour fails.

## Regression and provenance

- Two full CPU replays of the same run directory produce identical `rows.csv` (byte compare).
- Every run writes `config_hash`, `brand_list_hash`, `git_sha`; a test asserts their presence on every row.
- `pip freeze` snapshot recorded in the manifest.

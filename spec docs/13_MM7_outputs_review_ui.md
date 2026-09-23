# 13 — MM7 Outputs and Review UI (T56–T60, T73, T75)

**Effort:** 3 days. **Device:** CPU (video writer; Streamlit). **Gates:** T77 checkpoint.

## T56 — Per-detection logging of the frozen schema

`stages/output.py` writes `rows.csv` with the exact `CSV_COLUMNS` of `05`, one row per representative after dedup (DEDUPED rows are kept as rows with their decision, so the review screen can show them and T62 can count them). Lists serialise as JSON strings (M2 convention: `signals_used`, `faiss_top5_brands`, `audio_brands`); booleans as `True/False`; nulls as empty. `router_input.jsonl` and `router_output.jsonl` are the JSONL twins.

## T57 — `shot_id` on every row

Copied from `sampled_frames.json` (produced in MM1 Step 2) onto each row. Frames mode: shot ids from `ContentDetector` over the ordinal sequence; recorded as such in the manifest. No backfill onto the 1,739 rows (Revision 3).

## T58 — Annotated frame images for all sampled frames

For every entry in `sampled_frames.json` write `annotated_frames/frame_XXXXXX.jpg` (quality 90), even with zero detections. Drawing per box (after dedup, DEDUPED boxes drawn thin and grey):

- outline colour by `router_decision`: ACCEPT green, REVIEW orange, REJECT red, UNKNOWN_LOGO blue, DEDUPED grey, null (unrouted) white;
- label `"{canonical_brand or brand or '—'} · {router_decision} · {reason_code}"` plus `det_id`, wrapped to the frame width; `[cut off]` when `is_truncated`; `(partial)` when `partial_logo`;
- `selected_bbox` drawn solid; `loose_bbox` dashed when it differs (so the reviewer sees the refinement);
- watermark regions drawn as a dotted rectangle when present;
- footer strip: `video_id · frame {idx} · t={timecode:.1f}s · shot {shot_id} · {n} detections`.

Reuses `generate_hints.annotate_frames` drawing helpers (font loading, label background) with the label text replaced.

## T59 — Annotated video (MP4)

`annotated.mp4` from the annotated frames only (the sampled frames are what was processed; drawing on unprocessed frames would misrepresent recall). Each annotated frame is held for `interval_s` seconds (video mode) or `1/assumed_fps` (frames mode) at 25 fps output, so the video plays at real-time duration with the burnt-in timecode advancing. Written with OpenCV `mp4v` then re-encoded to H.264 by `ffmpeg` if available (the Phase 1 `test.py` did this). `per_shot` sampling holds each frame for its shot's duration. The writer streams frame by frame (Phase 1's `annotate_video` loaded the whole video into memory and wrote only annotated frames; it is not reused).

## T60 — Fresh-video run harness

`python -m logo_pipeline harness --config … --video-config …` = `run --to-stage merge` followed by `eval/merge_ablation.py`, producing `reports/harness.md` with the T18 counters per video and per frame, the T19 distributions, frame count, run time per stage, and (if the GPU stages were included) the decision distribution. No labels required. On the fresh videos the client supplies, the harness runs the full pipeline on the GPU machine; on the Mac it runs from copied raw detections. The recall figure from the 30–50 labelled fresh frames is computed by `eval/iou_match.py` (MM11) when the labels arrive, not by the harness.

## T73 — Review UI compatibility and queue filters

Changes to `app/review_ui.py` (about half a day):

1. **No conversion.** `rows.csv` loads as is; paths are CSV-relative and resolve through the existing `_resolve_path`. All M2 columns retained. New columns are ignored by existing code paths.
2. **Queue filters** in the sidebar: multiselect on `router_decision` (default `REVIEW, UNKNOWN_LOGO`) and multiselect on `reason_code` (options = values present), plus two preset buttons "REVIEW queue" and "UNKNOWN_LOGO queue". The existing label / `is_logo` / confidence filters stay.
3. **Router panel** under the VLM prediction: decision (coloured), reason code, rule number, `rules_matched_preempted`, `brand_evidence`, `canonical_brand`, `refinement_status`.
4. **Rule 5 evidence panel** (shown for rows whose `canonical_brand` is a channel brand or whose reason code starts with `channel_brand_`): `ocr_text`, `ocr_fuzzy_score`, `faiss_top1`, `faiss_top1_score`, `brand_from_audio`, `watermark_overlap`, `watermark_overlap_ratio`.
5. **Crop columns**: tight crop (selected), loose crop when different, enlarged crop, annotated frame (new fourth image using `annotated_frame_path`, which M2 already exported but the UI did not display).
6. **Metrics panel** additionally shows counts per decision and per reason code for the loaded CSV.
7. Labelling works as today (TP/FP/FN/wrong_brand written back to the CSV). Labels on DEDUPED rows are allowed but flagged.

The old CSVs (`extraction_output/video_3-trimmed/vlm_predictions.csv`) must still load: every new column read is guarded with `if col in df.columns`.

## T75 — "Accept into FAISS" on the review screen (one day, Revision 3 §4)

Visible for rows with `router_decision in {REVIEW, UNKNOWN_LOGO}` (and for ACCEPT rows behind an "also enrol" toggle, **[ASSUMPTION]**, since ACCEPT is "the decision that writes into the index" but the pipeline itself does not write in M3 — see `19`).

UI: a text input pre-filled with `canonical_brand` (empty for UNKNOWN), a checkbox "add to canonical list too" that only reminds the reviewer of the `06` procedure (no code path writes the YAML), and a button **Accept into FAISS**. On click:

```
img  = open(resolve(crop_path))                           # the selected_bbox crop
rec  = EnrolmentRecord(brand=typed_or_canonical, source_id=f"{video_id}/{run_id}/det_{det_id}",
                       video_id, run_id, frame_idx, det_id, crop_path, enrolled_at=utcnow(),
                       enrolled_by="review_ui", origin="review_ui.accept", review_label=row.label)
faiss_store.enrol.enrol([img], [rec], augmentations=None)   # the one instrumented method (T55)
row.enrolled_to_faiss = True; save_csv()
```

Pending enrolments are batched: **Accept** adds to a session queue shown in the sidebar; **Commit to index** calls `enrol` for the batch and `db.save()` once (the index file is 487 MB; saving per click is slow). The FAISS database object is loaded lazily on first use (CLIP on CPU is acceptable for single crops). The FAISS manager's own tabs are untouched; both applications now write through `faiss_store.enrol`.

Because `enrol` is the T55 method, MM7 depends on the enrolment API existing before MM10 fully instruments it: MM7 ships `faiss_store/enrol.py` with the record type and the legacy-list adapter; MM10 adds the eight call-site reroutes in the FAISS manager and the self-match/re-benchmark scripts. This is the reason `03` places `faiss_store/` outside `eval/`.

## Definition of done

- [ ] `rows.csv` has every `05` column in order; loads in the review UI; old M2 CSV still loads.
- [ ] Annotated frames exist for every sampled frame including empties; the video plays for the source duration.
- [ ] Harness report generated for `video_3-trimmed` and for one `fixed-images` video in frames mode.
- [ ] Queue presets show REVIEW and UNKNOWN_LOGO rows; Rule 5 panel visible on channel-brand rows.
- [ ] Accept into FAISS enrols one crop with a provenance record and the index reloads with `ntotal + 1`.

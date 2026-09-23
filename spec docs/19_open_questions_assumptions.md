# 19 — Open Questions and Assumptions

Every **[ASSUMPTION]** marker in the other documents resolves here. Section A needs the client; Section B needs the user; Section C lists assumptions taken silently because they are low-risk and reversible.

## A. To raise with the client

1. **Enhanced VLM fields (`brand_type`, `is_brand_identifier`).** The 24 July table lists them as "Available"; they have never been in the M2 prompt and the prompt is not redesigned. Under the v1 default, Rules 2 and 3 cannot fire and the corresponding clauses in Rules 6, 7, 8, 15 and 16 are omitted (`11` Part C). Options: (a) accept as an acceptance-evidence caveat; (b) add the two fields to the output JSON as a one-line addition of the same kind as T67 (`vlm.enhanced_fields: true`), which changes the prompt text slightly. Recommendation: (b), because Rule 2 is the only rule that rejects ingredients and slogans, a failure mode the client raised repeatedly.
2. **`fixed-images` provenance.** Confirm that the folder is the frame pool from which the Roboflow frames were uploaded (the mapping is verified; the provenance is inferred), the extraction rate (1 fps believed), and that the frames may be used for route three directly. Also whether the 60 unlabelled videos may serve as a fallback source of the 30–50 recall frames.
3. **Route three tiering.** GPU cost makes tier 1 (videos 14 and 1) the committed scope; tiers 2–3 as time allows (`17` T66). The Revision 3 wording ("on arrival of the first of these videos") is superseded.
4. **Route-two labelling window.** Under the re-baselined schedule it is about two days. Ask which the client prefers: label-dependent figures as a supplement after close, or a later close.
5. **Channel-brand set.** The `channel: true` marking in the snapshot (`07` MM0): confirm the list, especially `tim`, `sky wifi`, `x factor`, `infinity plus`.
6. **`not_logo_confidence` before the scored field exists.** v1 derives it from the M2 generated `confidence` on `not_logo` rows (24 July accepted the primary variant "OK now"). Confirm that this is acceptable until T32 passes.
7. **T75 for ACCEPT rows.** The button is specified for REVIEW and UNKNOWN_LOGO queues; should ACCEPT rows be enrolable from the same screen (the pipeline does not write to the index by itself in M3)?
8. **Audio in frames mode.** For `fixed-images` runs the audio signal is unavailable; Rule 11 cannot fire and T26 matches against the canonical list alone. Confirm this is acceptable for route three.
9. **Fresh-video notice.** Revision 3 promised notice on 16 September for delivery by 23 September. Under the re-baseline the notice date is 25 September for delivery by 2 October. The client must be told which.
10. **Watermark line in the blind call.** The blind pass suppresses the "channel watermark is X" line as a hint (`15` T67). Confirm.

## B. For the user to decide

1. **Schedule re-baseline.** `18` §3.2 proposes 17 September → 15 October. Confirm or adjust before anything is sent.
2. **Git history and the exposed token.** Revoke now (non-negotiable); decide whether to rewrite history (`git filter-repo`) or only fix the tip.
3. **Which GPU machine** the build targets (the RTX 3090 referenced in `task2-whisper-qwen.py`, Kaggle, or Colab) — it determines `requirements-gpu.txt` pins, transfer of 12 GB of frames, and whether the checkpoint run can be overnight.
4. **Use of the 1,739 rows as a CPU development fixture.** The rows are historical for acceptance, but `extraction_output/video_1/manifest.json` plus its crops are the only local data with realistic box geometry. Proposal: convert the manifest to `raw_detections.jsonl` (GDINO only) for unit and integration tests of MM2–MM5 on the Mac while the GPU machine is unavailable. This produces no deliverable and no acceptance figure.
5. **Committing per-video configs with client paths** (`configs/videos/*.yaml`): paths are machine-specific; keep them as templates with a `.local.yaml` override that is git-ignored.
6. **T55 storage shape** (`16`): flat string list kept + `enrolments.jsonl`, versus a mixed-type `metadata.json`. The spec chooses the former for backward compatibility.
7. **Rule 5 placement** in block 3 after Rule 14 (`11`). The 18 July pseudocode evaluates it after the audio/FAISS-only rules; Revision 3 says "numeric order within a block", which would place it first in block 1 as a REVIEW rule out of place. Confirm the block-3 placement.

## C. Assumptions taken in the specs (reversible, low risk)

| # | Assumption | Where |
|---|------------|-------|
| C1 | `loose_bbox` is always the original detector box; merged partners are referenced by `merged_raw_det_ids`; no `union_bbox` | `05` A, `08` |
| C2 | Small-box centre-distance merge branch shipped disabled behind `merge.small_box.enabled` | `08` T13 |
| C3 | `refinement_status` enum is `applied / no_candidate / rejected_safety_check / oversized_region`; `fallback_to_loose` derived | `09` |
| C4 | `watermark_overlap` uses `loose_bbox` | `12` T48 |
| C5 | Rows with unparsed VLM JSON get REVIEW `vlm_output_unparsed` | `11` Part D |
| C6 | `router_confidence = clip((trust − risk + 10)/20, 0, 1)`; logged only | `11` Part F |
| C7 | `logo_probability_scored = 100 × (P(logo) + P(partial_logo))` | `15` T30 |
| C8 | Annotated video shows sampled frames only, each held for the sampling interval at 25 fps | `13` T59 |
| C9 | `evaluation.channel_classes` list used for with/without reporting | `04` |
| C10 | Frames-mode shot detection uses `ContentDetector`; `shot_id` is still written | `03` §5.2 |
| C11 | Route two sampler seed 42, `per-code 70` (inside the 60–80 band) | `17` T66 |
| C12 | The `UNKNOWN` wording change (T67) is applied in MM1, not MM9, to avoid touching the prompt twice | `07` Step 5 |
| C13 | Deduped rows remain in `rows.csv` with decision DEDUPED | `13` T56 |
| C14 | Rule 11 REJECT variant is evaluated before its REVIEW variant inside block 3 | `11` |
| C15 | `x` (one-character brand) and the other 37 short names only match by exact OCR read | `06` §6 |

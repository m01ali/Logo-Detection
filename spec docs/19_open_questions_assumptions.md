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
11. **Rule 1 threshold on the scored `not_logo_confidence`.** T32 passed on `video_1` (1,596 distinct values; the score agrees with the hinted decision on 584 of 584 logo/partial rows and 1,111 of 1,243 `not_logo` rows). But `rule1_not_logo_confidence: 0.80` was set for the generated field, which clusters at 0.9–0.95 on `not_logo` rows; the scored field puts most `not_logo` rows between 0.45 and 0.90. Switching as `15` T32 prescribes moves 905 rows from REJECT to REVIEW (review load 263 → 1,168 of 1,829; `router-only` replay). The `video_1` acceptance run is therefore held on the generated field, with the scored fields logged. Options: (a) keep the generated field for M3; (b) switch with a threshold set for the scored field (no hinted-logo row scores above 0.44; 0.5 reproduces the hinted `not_logo` decision on 89 % of rows), chosen against route-two labels rather than against the model's own decision; (c) switch at 0.80 and accept the review load. Recommendation: (a) for M3 acceptance, (b) once labels exist.
12. **Rule 1 rejects most true logos on route three.** On tier 1, 583 of 584 (video 14) and 489 of 499 (video 1) gold boxes are re-detected and every ACCEPT is right (43/43, 69/69), but Rule 1 (`high_confidence_not_logo`) REJECTs 76 % and 64 % of the matched true logos: the VLM answers `not_logo` at generated confidence 100. The rejected crops are legible sponsor wordmarks and crests (`runs/14/route3/reports/rejected_true_logos.png`); the reasoning cites "text" on 96 % / 89 % of them and treats garbled OCR ("cowo", "Rci", "Unecainas") as evidence that no brand is there. Two causes: (i) image size — at the processor default a 66 × 27 px crop reaches the model as 8 image tokens, and the rejected boxes are smaller (median short side 42 px against 71 px kept on video 14); (ii) the prompt's "plain scene text" exclusion together with the hints, read by the model as "unreadable, so not a logo". Options, each measured by `vlm-ab` on sampled rescue / held / negative groups (`reports/vlm_ab_<tag>.md`): (a) `vlm.crop_min_pixels: 50176` — an input change, the prompt text is unchanged; (b) (a) plus `vlm.wordmark_rule: true` — one paragraph after the not_logo definition, a prompt change of the T67 kind; (c) no change, the figure reported as a caveat. Either change means re-running the VLM stage on the three runs. At the 37–50 s per call measured in these experiments (18–29 s for the same boxes in the original runs; see the note on the machine below), that is about 11–15 h per route-three video and about 40–50 h for `video_1`'s 1,829 hinted and 1,830 blind calls. `vlm-ab` figures (40 rejected true logos, 20 kept true logos and 40 boxes no labeller drew, per video; the same boxes for every variant):

| variant | rejected true logos recovered (v14 / v1) | kept true logos lost | unlabelled boxes newly called logos (v14 / v1) | rescued boxes with the right brand (v14 / v1) |
|---|---|---|---|---|
| (a) upscale only | 25 % / 47.5 % | 0 / 0 | 7.5 % / 2.5 % | — |
| (b) upscale + wordmark paragraph | 87.5 % / 87.5 % | 0 / 0 | 17.5 % / 22.5 % | 30 / 35, 29 / 35 |
| (b′) upscale + tightened paragraph | 85 % / 77.5 % | 0 / 0 | 12.5 % / 5 % | 30 / 34, 27 / 31 |

Over both videos, (b) recovers 70 of 80 sampled rejected true logos and newly calls 16 of 80 unlabelled boxes logos; (b′) recovers 65 and calls 7. Projected over the runs, that is about 661 against 619 of 756 rejected true logos escaping Rule 1, and 132 against 55 of 653 unlabelled boxes.

Video 14's rescued boxes had the right brand on only 12 of 35 under (b) before the alias fix. That was mostly alias-table gaps rather than wrong brands; the `06` §4 procedure added them (brand list version 2026-11-07: `como`, `como 1907` → `como calcio`; `enilive`, `eni live`, `enelive` → `eni`; `san bernardo`, `s bernardo`, `sbernardo` → `san bernardo acqua`; `leovegas`, `leovegas play` → `lesvegas play`). With them it is 30 of 35, and the run's own brand-correct share on matched boxes rises from 15.1 % to 19.4 %. Misreadings that name another real brand (`amoco`, `revox`, `enel`) were not added. Of the 16 unlabelled boxes (b) turned into logos (`eval/vlm_ab_review.py`), about 6 are logos the labeller did not draw (a small Rolex, the Spotlight 29 mark, an Uber fragment, the Como crest on a kit) and about 9 are clocks, scoreboards and captions, several given a brand seen elsewhere in the frame ("12:52" → Rolex); none can reach ACCEPT without an OCR read of the same brand (Rule 6), so their cost is review load. (b′) (`vlm.wordmark_exclusions: true`) tightens the paragraph against both: text graphics stay `not_logo`, and the brand must be visible in the region itself. Samples of 40 give wide intervals (35/40 is roughly 73–96 %), so (b) and (b′) were compared box by box on the same boxes (`eval/vlm_ab_compare.py`). (b′) is right on no true logo that (b) misses. It loses 5 rescued true logos (exact sign test p = 0.06 over both videos) and leaves 9 more unlabelled boxes `not_logo` (p = 0.004). Judged by eye from the contact sheets:

- **The 5 true logos (b′) loses:** 4 are legible marks (ATP Tour, FAGE, Rai twice) and one is an unreadable crop. (b′)'s reason is that it cannot name the brand, a case the paragraph says should be `UNKNOWN`.
- **The 9 unlabelled boxes (b′) leaves `not_logo`:** 5 are clocks and scoreboards; 4 are marks the labeller did not draw (an Uber fragment, two "24" channel marks, the "29" mark).
- **What (b′) still calls logos:** news captions and headlines ("ULTIM'ORA KUWAIT", "MIGRANTI.", "CALCIO SERIE A.").

Over the 80 unlabelled boxes, (b) calls 9 text graphics and 7 undrawn marks logos; (b′) calls 4 and 3. (b′) therefore gives up about 9 real marks to avoid 5 text graphics.

**Recommendation: (b).** A wrong `logo` call costs a review: none can reach ACCEPT without an OCR read of the same brand (Rule 6). A missed logo is a REJECT that nobody sees. (b)'s cost is about 132 more REVIEW rows over the two route-three runs, close to half of them marks the labeller did not draw. Its brand is right on 59 of 70 rescued boxes, and the brand string is sometimes taken from elsewhere in the frame (Rolex on a Rai mark); the reviewer corrects it. (b′) stays in the code, off. `vlm.wordmark_rule` changes the prompt and needs the client's sign-off; until then both settings stay at their defaults (off) and no run changes. The per-call times above were measured while about 17.5 GB of the 24 GB card was held by other processes (an Ollama server, VMware sandboxes), as checked straight afterwards. The re-run cost is to be re-measured with those stopped.

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

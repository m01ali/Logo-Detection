# 14 — MM8 Checkpoint (T77)

**Effort:** 0 days of build (Revision 3 lists no day figure); about half a day to run, package and present. **Device:** GPU for the run, CPU to package. **Placement:** after Groups 1, 2, 6 and 9 (MM1–MM7), before Group 10 is written. Not a milestone gate; Group 5 (MM9) proceeds in parallel unless a defect is found.

## What the client sees

One video processed from the YAML through to annotated frames, with the per-run CSV loading in the review screen and the REVIEW and UNKNOWN_LOGO queues populated (Revision 3 T77).

## Which video

Two options, both prepared:

| Option | Pros | Cons |
|--------|------|------|
| **A. `video_1` hinted pass only (recommended)** | It is the corpus video; the hinted `vlm/hinted.jsonl` is reused by MM9, which adds `vlm/blind.jsonl` to the same run directory — no GPU time is wasted. Rai Sport bug, Frosta/Orogel/Verisure/Barbie adverts, NeoBorocillina (not in the list → exercises UNKNOWN_LOGO) | ≈ 7 h of GPU for ~1,800–2,500 detections at 13.8 s each (YOLO adds boxes; the merge removes some) |
| B. `video_3-trimmed` | 212 detections, under an hour | Only 4 sampled frames; not the corpus |

Run B first as the smoke test on the day, then A. Present A if it finishes in time; otherwise present B and A's partial output.

## Checkpoint procedure

1. Freeze the code at a tag `m3-checkpoint`; record the git SHA in `run_manifest.json`.
2. `run` with the shipped YAML (`interval_s 2.0`, merge on, refinement on, all signals computed and in the prompt, blind pass **off** at this point, scoring off, Rules 7/8/5-REJECT off).
3. Copy the run directory to the Mac; run `--from-stage merge --to-stage output` to prove the CPU replay; diff `rows.csv` (must be identical).
4. Generate `reports/`: decision distribution by reason code, merge counters and distributions (T18/T19), refinement fallback rate (T24), canonicalisation mode distribution, Rule 5 branch counts, per-stage timings.
5. Package for the client: the annotated frames folder, `annotated.mp4`, `rows.csv`, `configs/pipeline.yaml`, `configs/videos/video_1.yaml`, `configs/reason_codes.md`, the reports, and a two-page note listing what is and is not yet in the run (blind fields null, `not_logo_confidence` from the generated field, Rules 2/3 unreachable, no fresh-video watermark regions beyond `video_1`).
6. Walk the client through the review screen: load the CSV, show the REVIEW queue, the UNKNOWN_LOGO queue, a Rule 5 row with its evidence panel, one Accept-into-FAISS enrolment on a copy of the index.

## Acceptance for the checkpoint (self-check before showing)

- [ ] Every sampled frame has an annotated image; count equals `sampled_frames.json`.
- [ ] `rows.csv` loads without warnings; no null `router_decision` except `vlm_parse_ok == False` rows (those show `vlm_output_unparsed`).
- [ ] Both queues non-empty; at least one `channel_brand_*` row (Rai Sport bug present in `video_1`).
- [ ] The two T65 ordering checks pass on the run.
- [ ] Config hash, brand list version and hash, git SHA visible in the manifest and on every row.

## What feedback can change before Group 10

Thresholds (YAML only), reason-code wording, the annotated-frame legend, queue presets. A rule-logic defect is fixed in MM5 and replayed with the router-only runner — which is why MM11's T61 is the first thing built after the checkpoint.

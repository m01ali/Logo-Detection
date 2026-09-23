# M3 Specification Set — Index

**Milestone:** 3 (Rule-based router, quarantine workflow, crop refinement, configured pipeline)
**Source of truth:** `M3_Final_Implementation_Task_List_06_09_26.docx` (Revision 3, "the list that gets built")
**Prepared:** 16 September 2026, from a full audit of the repository, the `fixed-images` folder, the Roboflow export, the FAISS index, the M2 extraction outputs, and the complete M3 correspondence chain (18 July → 6 September).

These documents turn the 74 tasks of Revision 3 into buildable specifications, grouped into eleven mini-milestones (MM0–MM11) that follow the build order and day estimates in Section 8 of the task list. Every task ID (T5–T77), rule number (Rule 1–18) and reason code used here is the one used in the correspondence, so the specs can be quoted back to the client without translation.

## Reading order

| # | File | What it is | Read when |
|---|------|-----------|-----------|
| 00 | `00_README.md` | This index, conventions, headline findings | First |
| 01 | `01_research_findings.md` | Codebase audit: what exists, what is reusable, what is missing, discrepancies between the correspondence and the code | Before any code is written |
| 02 | `02_data_inventory.md` | Data audit: `fixed-images`, Roboflow export, FAISS index, M2 extraction outputs, videos. Includes the verified mapping between the Roboflow labels and `fixed-images` | Before Group 0 and before Group 10 |
| 03 | `03_architecture.md` | Target package layout, stage contracts, run-directory layout, CLI, resumability, dual ingest (video / frame folder) | Before MM1 |
| 04 | `04_config_yaml_spec.md` | The single YAML (T8): every key, type, v1 value, and which task owns it | Before MM1, referenced by every MM |
| 05 | `05_row_schema.md` | The frozen row schema (T9) with every field, type and producing stage; the document that goes to the client for confirmation | MM1, then frozen |
| 06 | `06_canonical_brand_list.md` | The versioned brand-list snapshot (T5): format, hash, aliases, add-a-brand procedure, initial alias seeds | MM0 |
| 07 | `07_MM0_MM1_foundation.md` | MM0 (T5, T76) and MM1 (T7–T11): the refactor that gates everything | Build start |
| 08 | `08_MM2_detection_merge.md` | MM2 (T12–T19): YOLO wiring, pooled greedy merge, `min_merge_area`, counters, distributions | After MM1 |
| 09 | `09_MM3_crop_refinement.md` | MM3 (T20–T24): heuristics 1, 2, 5; `selected_bbox`; fallback rate | After MM2 |
| 10 | `10_MM4_ocr_brand_matching.md` | MM4 (T25–T29): OCR verification pass, length-dependent matching against the canonical list + audio brands | After MM3 |
| 11 | `11_MM5_router.md` | MM5 (T37–T46, T72): canonicalisation, all 18 rules in six blocks with exact conditions, the complete reason-code enum, dedup, firing log | After MM4 |
| 12 | `12_MM6_watermark.md` | MM6 (T47–T50): per-video regions, `watermark_overlap`, Rule 5 three-branch form | After MM5 |
| 13 | `13_MM7_outputs_review_ui.md` | MM7 (T56–T60, T73, T75): per-run CSV, annotated frames and video, fresh-video harness, review-UI queue filter, Accept-into-FAISS | After MM6 |
| 14 | `14_MM8_checkpoint_T77.md` | MM8 (T77): the one-video end-to-end checkpoint the client asked to see before Group 10 | After MM7 |
| 15 | `15_MM9_vlm_blind_pass.md` | MM9 (T67, T30–T36): UNKNOWN in the prompt, log-prob spike, closed-set decision scoring, blind pass, hint-composition log, cost scoping | After MM8 (GPU) |
| 16 | `16_MM10_faiss.md` | MM10 (T51–T55): self-match diagnostic, two-population re-benchmark, instrumented enrolment | After MM9 |
| 17 | `17_MM11_evaluation.md` | MM11 (T61–T66, T71): router-only runner, metric scripts, IoU matching, blind-vs-hinted, ordering checks, corpus routes two and three | Last |
| 18 | `18_schedule_dependencies_risks.md` | Dependency graph, critical path, day budget per MM, Revision 3 dates versus the re-baselined forecast, risks | Planning |
| 19 | `19_open_questions_assumptions.md` | Every assumption taken in these specs and every question that still needs the client or the user | Before sending anything to the client |
| 20 | `20_received_artefacts_record.md` | T76: the received-artefacts record, seeded with what is on this machine today | MM0, then maintained |
| 21 | `21_test_plan.md` | Unit, fixture and acceptance tests per mini-milestone; what "done" is checked against | Every MM |

## Conventions

- **Task IDs** `T5`–`T77` are Revision 3 IDs. Closed tasks (T1–T4, T6, T68–T70) are not re-specified; their results are summarised in `01`/`02` where they affect the build. T35 and T74 are removed and appear only in the out-of-scope list.
- **Rules** are numbered 1–18 as in the 18 July document; the six-block order is Rejects 1–5 → Accepts 6–8 → Review/conflict 9–14 → UNKNOWN_LOGO 15–16 → Dedup 17–18 → default REVIEW.
- **Decisions locked** in Section 3 of Revision 3 are treated as fixed. Where these specs had to choose something the correspondence left open, the choice is marked **[ASSUMPTION]** inline and collected in `19`.
- **v1 values** are the shipped YAML defaults. Every numeric threshold lives in the YAML (T8). No spec here introduces a constant that is not also listed in `04`.
- **Mini-milestones** (MM) are the build units. Each MM doc has: goal, tasks covered, inputs, step list, definition of done, evidence produced, and estimated effort. The MM order is the Revision 3 build order: Groups 0, 1, 2, 3, 4, 6, 7, 9, T77, 5, 8, 10.
- **Paths** are relative to the repository root `/Users/alii/Documents/Code/Logo-Detection` unless stated.

## Headline findings that shape these specs

1. **`fixed-images` is the frame pool the Roboflow gold labels were drawn from.** All 5,830 labelled images map by filename to `fixed-images/{video}_frame_{n}.png`, with identical dimensions and only JPEG-level pixel differences. The folder holds 20,222 frames from 100 videos; 40 of those videos carry labels (5,830 frames), 60 do not (14,392 frames). This removes the sampling-rate and resolution-mapping blockers on route three of T66, provided the pipeline gains a **frame-folder ingest mode** (specified in `03`). It does not supply audio, so the audio signal is absent in that mode. See `02` and `17`.
2. **The router, merge, refinement, watermark, canonical list and evaluation scripts are all greenfield**, exactly as Revision 3 states. The M2 code that is wrapped rather than rewritten is: hint computation and the VLM call (`generate_hints.py`), audio brand extraction (`generate_hints.py` / `task2-whisper-qwen.py`), annotation and CSV export (`generate_hints.py`), the review screen (`app/review_ui.py`), the FAISS manager and database class (`app/streamlit_faiss_app.py`, `models/faiss_db.py`), and the Phase 1 shot detector and fuzzy matcher (`utils/`). See `01`.
3. **The build machine matters.** The Mac venv has no torch, transformers or ultralytics and no CUDA. Detection, OCR at scale and every VLM step run on the GPU machine. Merge, refinement, router, evaluation and both Streamlit apps run anywhere. The specs mark each stage CPU or GPU.
4. **Schedule.** Revision 3 dates assumed a start on Monday 8 September with T7 done on 11 September. As of 16 September no pipeline package exists in the repository, so the plan is re-baselined in `18` on the same 20-day effort. The re-baselined dates are a forecast for the user to confirm, not a commitment made here.
5. **A live secret is committed.** `task2-whisper-qwen.py` line 40 sets `HF_TOKEN` to a literal Hugging Face token. It must be revoked and removed in MM0 housekeeping (see `01` §6 and `07`).

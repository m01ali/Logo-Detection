# 15 — MM9 VLM, Confidence and Blind Pass (T67, T30–T36)

**Effort:** 2 days of build. **Device:** GPU throughout. **Placement:** between T77 and Group 8 so the checkpoint is not delayed by GPU work. **Produces:** the M3 full run over `video_1` (both passes), which is the corpus for T61, T64 and route two.

## T67 — UNKNOWN as an explicit output option

Done in MM1 Step 5 (one line in the open-vocabulary instruction). MM9 verifies it against the blind prompt too: both calls share `build_final_instruction`, and the blind call's instruction must not mention hints at all ("Base your answer ONLY on the images shown above"). Unit test: the rendered blind prompt contains no `## Signal: audio_brands|faiss_matches|ocr_text` header and no "Known context: the channel watermark" line **[ASSUMPTION]**: the watermark line is a hint and is suppressed in the blind call; listed in `19`.

## T30 — Timeboxed log-prob spike (≤ 1 day, hard stop)

Goal: establish that, on the deployed checkpoint (`Qwen2.5-VL-7B-Instruct`, 4-bit, transformers), the four decision labels can be scored under the hinted context and yield a spread distribution.

Procedure (`stages/vlm_scoring.py`, `--spike` mode over a 50-detection sample spanning the content types):

1. Build the hinted prompt exactly as the production call, then an assistant prefix `{"is_logo": "` (the JSON opening the model would emit).
2. For each label `L ∈ {logo, partial_logo, not_logo, uncertain}`: teacher-force `L + '"'` after the prefix; sum the log-probs of `L`'s tokens (method `sequence_logprob`; labels are short fixed strings so length bias is negligible and renormalisation is sound, 29 July).
3. `decision_scores = softmax(seq_logprob over the four labels)`; `logo_probability_scored = 100 × (P(logo) + P(partial_logo))` **[ASSUMPTION]** (partial counts toward "any logo", consistent with the M2 definition of the generated field); `not_logo_confidence = P(not_logo)`.
4. Record whether constrained/JSON decoding was active (it is not; plain greedy generation is used for the text output), the checkpoint id, quantisation and the prefix used.
5. Cost per detection of the scoring pass (vision tower reused via KV cache where transformers allows; otherwise a second forward pass) → feeds T36.

Exit criteria: scores computed for all 50; the `logo`/`not_logo` mass tracks `is_logo` in the majority of rows; at least 20 distinct `logo_probability_scored` values. If the spike fails (e.g. `uncertain` absorbs mass because it is not in the prompt's allowed values, or the prefix trick misaligns tokens), the categorical fallback of 29 July is scoped and the client informed; Rules 1-fallback and 8 stay off.

## T31 — Closed-set decision scoring in the pipeline

`vlm.scoring.enabled: true` runs the T30 procedure for every detection after the hinted call, writing `vlm/scores.jsonl` (`decision_scores`, `logo_probability_scored`, `not_logo_confidence`). `router.not_logo_confidence_source` stays `generated` until T32 passes.

## T32 — Distribution check before any rule consumes the score

`eval/score_distribution.py --run <video_1 run>` → `reports/score_distribution.md`: histograms of `logo_probability_scored` and `not_logo_confidence` (100 bins), stratified by `is_logo`; number of distinct values; share of `logo`/`partial_logo` rows above 95; comparison with the generated field (which took only 0/95/100 in M2). Pass criterion (stated in the report): no single 1-point bin holds more than 50 % of the positive rows and there are ≥ 30 distinct values. On pass: switch `router.not_logo_confidence_source: scored`, enable `rule_1_fallback_logprob`; Rule 8 remains off regardless (locked). On fail: report it, leave the router on the generated field, and say so in the acceptance evidence.

## T33 — Blind pass paired per candidate in the same run

`stages/vlm.py` makes two calls per representative when `vlm.blind_pass: true`:

- **Hinted**: image signals + every text signal whose `include_in_vlm_prompt` is true, open-vocabulary instruction, watermark line, UNKNOWN option → `vlm/hinted.jsonl`.
- **Blind**: `vlm.blind_pass_signals` (the four image signals) only, same instruction minus hint references and the watermark line → `vlm/blind.jsonl` (`vlm_decision_blind`, `vlm_brand_blind`, plus the raw JSON).

Every signal is computed and logged for both (they are computed once per detection by their own stages; the blind pass simply does not render them). Both files are resumable per `det_id`, and the blind pass can be added to an existing run directory (`--stages vlm --blind-only`), which is how the checkpoint's hinted run becomes the full corpus run without repeating the hinted pass.

## T34 — Hint composition log

`vlm/hint_composition.jsonl`, one line per detection: `[{"item": "rai sport", "source": "watermark"}, {"item": "verisure", "source": "audio"}, {"item": "Admiral Oversea Corporation", "source": "faiss", "rank": 1, "score": 0.973}, {"item": "SPORT HD", "source": "ocr", "confidence": 0.81}]`, plus `signals_rendered` (the section names actually in the prompt) and `mode: open`. Canonicalisation derives `hinted_brand_in_faiss_hints` from it, which is T64's FAISS-origin attribution. In open mode this measures "the hinted brand matched a FAISS hint that was shown", the weaker statement Revision 3 §10 records.

## T36 — Cost scoping

`reports/vlm_cost.md` from the spike and from the full run: seconds per hinted call, per blind call, per scoring pass; detections per sampled frame before and after the merge; projected hours for `video_1` and for the route-three tiers (`17` §T66). The Revision 3 planning figure is ~13 h of GPU for two passes over ~1,739 detections; the report replaces it with measured numbers.

## The M3 full run over `video_1` (the corpus)

Run order on the GPU machine: if the checkpoint used option A, `run --stages vlm --blind-only`, then `vlm_scoring` (if T30 passed), then `--from-stage canonicalise --to-stage output` (CPU-capable). Result: `runs/video_1/<run_id>/` with both passes, scores, decisions, `rows.csv`. Publish the decision distribution by reason code (`eval/metrics.py --no-labels`) and issue the route-two label request (`17`).

## Definition of done

- [ ] Blind prompt renders no hint sections; unit test.
- [ ] Spike report written within the timebox with a pass/fail statement.
- [ ] `video_1` run has `hinted.jsonl`, `blind.jsonl`, `hint_composition.jsonl`, (optionally) `scores.jsonl` with one line per representative.
- [ ] Distribution report written; router key switched only on pass.
- [ ] Cost report written.

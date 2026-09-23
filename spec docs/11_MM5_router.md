# 11 — MM5 Canonicalisation and Router (T37–T46, T72)

**Effort:** 2 days. **Device:** CPU. **Gates:** MM6 (Rule 5 lives here, reads the watermark stage), MM7 (decisions on frames and CSV), MM11 (router-only runner replays this engine).

## Part A — Canonicalisation (T37)

`stages/canonicalise.py` reads `vlm/hinted.jsonl`, `vlm/blind.jsonl` (when present), `faiss_hints.jsonl`, `audio/audio_hints.json`, the brand list; writes `canonical.jsonl` (sections E-derived, F-derived, H-derived, I of `05`).

Deterministic map `canonicalise(s) -> (canonical, mode, score)`:

1. `s is None` → `(None, None, None)`; `s == "UNKNOWN"` → `("UNKNOWN", "unknown", None)`.
2. `norm = brands.normalise(s)`.
3. **Alias table first**: `brands.map_exact(norm)` (aliases, then canonical names) → `mode = alias | exact`.
4. **Fuzzy onto the canonical list** with the length rule: `brands.map_fuzzy(norm, canonicalisation.fuzzy_threshold=90, short_name_cutoff=4)` → `mode = fuzzy`.
5. **Audio brands as targets**: exact-then-fuzzy against the normalised `audio_brands` of the video (audio brands are themselves canonicalised first through steps 3–4, so `very sure` → `verisure` before it can be a target). A hit returns the canonicalised audio brand (`mode = audio`).
6. Otherwise `(norm, "unmatched", best_score)` — the normalised VLM string is kept as `canonical_brand` so dedup can still group repeated spellings exactly, and the `unmatched` mode is the feed for the add-a-brand procedure.

Applied to: `vlm_brand` → `canonical_brand` (and the M2 column `brand_canonical`), `vlm_brand_blind` → `vlm_brand_blind_canonical`, `faiss_top1` → `faiss_top1_brand_canonical`, `audio_brands` → `audio_brands_canonical`. Derived: `faiss_agrees_with_vlm`, `brand_from_audio`, `brand_in_database` (`canonical_brand` or any of its aliases equals a normalised index brand string), `blind_hinted_brand_agree`, `hinted_brand_in_faiss_hints` (canonical_brand ∈ canonicalised `faiss_top5_brands`).

**LLM merge (optional, off).** `canonicalisation.llm_merge.enabled: true` runs M2 `canonicalize_brands` over the set of `unmatched` strings only, after step 6, and records `mode = llm_merge` with the raw mapping saved to `canonical_llm_merge.json`. Off by default; when on, the output is logged, never silently substituted (Revision 3 T37).

**Open mode load (Revision 3 §2).** The report `reports/canonicalisation.md` lists the distribution of `canonical_match_mode` and the top unmatched strings so the alias table can be grown after the first full run.

## Part B — Router inputs (`stages/router_input.py`)

`RouterInput` is a dataclass with every field the rules read. Construction from the joined row raises `MissingRouterInput(field)` if a field a **enabled** rule needs is `None` for a reason other than "legitimately null" (e.g. `watermark_overlap` may be null when no regions exist; `ocr_brand_match` may be null). The rule functions declare their required fields; the engine checks them once per run against the config and refuses to start if an enabled rule needs a field no stage produces (27 July: no `.get()` defaults).

Derived booleans (18 July §9, with the thresholds from `04`):

```
vlm_brand_c   = canonical_brand           (None / "UNKNOWN" / str)
ocr_brand_c   = canonicalise(ocr_brand_match)
faiss_brand_c = faiss_top1_brand_canonical

has_conflicting_ocr   = ocr_brand_c not in (None,"UNKNOWN") and vlm_brand_c not in (None,"UNKNOWN")
                        and ocr_brand_c != vlm_brand_c and ocr_fuzzy_score >= rule13_ocr_fuzzy(85)
strong_visual_match   = faiss_agrees_with_vlm and faiss_top1_score >= rule7_top1(0.97) and faiss_score_gap >= rule7_gap(0.08)
audio_only_prediction = brand_from_audio and ocr_fuzzy_score < rule11_ocr_fuzzy(80) and not strong_visual_match
faiss_only_prediction = brand_from_faiss and ocr_fuzzy_score < rule12_ocr_fuzzy(80) and faiss_top1_score < rule12_top1(0.97)
brand_from_faiss      = hinted_brand_in_faiss_hints                      # the hinted brand appears in the FAISS hints shown
brand_evidence        = both | ocr_confirmed | faiss_confirmed | none    # T43:
                        ocr_confirmed  := ocr_brand_c == vlm_brand_c and ocr_fuzzy_score >= general(90)
                        faiss_confirmed:= faiss_agrees_with_vlm and faiss_top1_score >= rule14_top1(0.90)
not_logo_confidence   = decision_scores.not_logo (T31) if router.not_logo_confidence_source == "scored"
                        else confidence/100 when is_logo == "not_logo" else None          # [ASSUMPTION] see §E
```

## Part C — The 18 rules, six blocks (T38–T42, T72)

Evaluation: blocks in order; within a block rules in numeric order; the first rule whose condition holds returns `(decision, reason, router_confidence, risk_score, trust_score)` and evaluation stops (T44). Disabled rules are skipped. The firing log (T46) additionally records every enabled rule whose condition held but was pre-empted, which requires condition functions to be side-effect-free and evaluable on their own.

**Enhanced VLM fields are not available in M3.** `brand_type`, `is_brand_identifier` and `evidence_text` were listed as "Available" on 24 July but the M2 prompt never produced them and the prompt is not redesigned. Under `router.enhanced_vlm_fields_available: false` (v1): Rules 2 and 3 are config-disabled (they cannot fire), and the `brand_type == "brand"` / `is_brand_identifier is True` clauses in Rules 6, 7, 8, 15 and 16 are **omitted**, with the omission printed in the reason-code document and the acceptance evidence. Listed in `19` as the first question to raise: adding the two fields to the prompt is a one-line JSON addition of the same size as T67.

### Block 1 — Rejects

| Rule | Condition (v1) | Decision / reason code | Notes |
|------|----------------|------------------------|-------|
| 1 | `vlm_decision == not_logo and not_logo_confidence >= 0.80` | REJECT `high_confidence_not_logo` | Primary variant. `not_logo_confidence` source per §E. |
| 1-fallback | `vlm_decision == not_logo and logo_probability_scored <= 20` | REJECT `low_logo_probability_not_logo` | `rule_1_fallback_logprob: false` until T31/T32 pass. |
| 2 | `brand_type in {generic_word, ingredient, slogan, ui_icon, sign}` | REJECT `non_brand_content_<brand_type>` | Disabled (no input). |
| 3 | `is_brand_identifier is False and vlm_decision in {not_logo, partial_logo, uncertain}` | REJECT `not_a_brand_identifier` | Disabled (no input). |
| 4 | `box_area_ratio >= rule4_area_ratio(0.45) and not has_valid_tight_bbox` | REJECT `oversized_scene_region_without_tight_logo` | With a valid tight box the row continues (crops already use it). |
| 5-REJECT | `canonical_brand in known_channel_brands and watermark_overlap is False and brand_from_audio and ocr_fuzzy_score < 70` | REJECT `likely_channel_audio_contamination` | **Disabled for all of M3** (`rule_5_reject: false`). |

### Block 2 — Accepts

| Rule | Condition (v1) | Decision / reason code | Notes |
|------|----------------|------------------------|-------|
| 6 | `vlm_decision == logo and is_truncated is False and ocr_fuzzy_score >= rule6_ocr_fuzzy_threshold(95) and ocr_brand_c == vlm_brand_c` (+ enhanced clauses when available) | ACCEPT `clear_logo_with_strong_ocr_match` | The only auto-accept in v1. Reads its own key (T72). |
| 7 | `vlm_decision == logo and is_truncated is False and strong_visual_match and not has_conflicting_ocr` | ACCEPT `clear_logo_with_strong_visual_database_match` | Implemented, `rule_7: false` pending T54. |
| 8 | `vlm_decision == logo and logo_probability_scored >= 95 and is_truncated is False and not has_conflicting_ocr and not audio_only_prediction and not faiss_only_prediction and box_area_ratio < 0.45` | ACCEPT `high_confidence_visual_logo_no_conflicts` | Implemented, `rule_8: false` for all of M3. |

### Block 3 — Review and conflict

| Rule | Condition (v1) | Decision / reason code |
|------|----------------|------------------------|
| 9a | `vlm_decision == partial_logo and ocr_fuzzy_score >= rule9_strong_ocr(95) and ocr_brand_c == vlm_brand_c and is_truncated is False` | REVIEW `partial_logo_with_strong_ocr_support`, `review_priority = low` |
| 9b | `vlm_decision == partial_logo` | REVIEW `partial_logo_default_review` |
| 10 | `is_truncated is True` | REVIEW `truncated_logo_requires_review` |
| 11-REJECT | `brand_from_audio and vlm_decision in {partial_logo, uncertain} and ocr_text == "" and not strong_visual_match` | REJECT `weak_visual_crop_rescued_by_audio_hint` (both variants adopted 24 July; the REJECT variant is evaluated before the REVIEW variant, and it is in this block because that is where 24 July placed both) |
| 11-REVIEW | `audio_only_prediction` | REVIEW `audio_only_brand_prediction` |
| 12a | `brand_from_faiss and ocr_fuzzy_score < 80 and not faiss_agrees_with_vlm` | REVIEW `faiss_only_or_faiss_conflicting_prediction` |
| 12b | `brand_from_faiss and faiss_top1_score < rule12_top1(0.97) and ocr_fuzzy_score < rule12_ocr_fuzzy_second(90)` | REVIEW `faiss_score_not_discriminative_enough` |
| 13 | `has_conflicting_ocr` | REVIEW `ocr_vlm_brand_conflict` |
| 14 | `faiss_brand_c not in (None, UNKNOWN) and vlm_brand_c not in (None, UNKNOWN) and faiss_brand_c != vlm_brand_c and faiss_top1_score >= rule14_top1(0.90)` | REVIEW `faiss_vlm_brand_conflict` |
| 5-REVIEW | see `12_MM6_watermark.md` §T49 (three-branch form) | REVIEW `channel_brand_outside_watermark_region` or `channel_brand_needs_watermark_check`; otherwise no decision |

Rule 5's REVIEW variant is placed in this block, after 14, because it routes to REVIEW and 24 July's table treats it as a review rule; the 18 July pseudocode also evaluates it inside the review block. **[ASSUMPTION]** listed in `19`.

The 18 July ordering evaluates truncation (10) before partial (9) inside the review block; Revision 3 fixes numeric order within a block. The engine uses numeric order: 9 then 10. The firing log will show how often both matched; T65 does not depend on this choice.

### Block 4 — UNKNOWN_LOGO (original 18 July conditions, T41; partial_logo removed from 16, T39)

| Rule | Condition (v1) | Decision / reason code |
|------|----------------|------------------------|
| 15 | `vlm_decision == logo and vlm_brand_c in (None, "", UNKNOWN)` (+ enhanced clauses when available) | UNKNOWN_LOGO `logo_like_but_brand_unknown` |
| 16 | `vlm_decision == logo and ocr_text != "" and ocr_fuzzy_score < rule16_ocr_fuzzy(80)` (+ enhanced clauses) | UNKNOWN_LOGO `readable_brand_like_text_not_in_database` |

### Block 5 — Dedup (per frame, post-VLM site of the shared function, T45)

Runs after every row in a frame has a block-1..4 decision or the default.

| Rule | Condition | Effect |
|------|-----------|--------|
| 17 | group rows by `canonical_brand` (non-null, non-UNKNOWN); within a group cluster boxes with `iou >= 0.50 or containment >= 0.80` (**and** `area_ratio >= 0.80`, the shared function's precondition; `min_merge_area` disabled at this site) | keep one representative per cluster; others → DEDUPED `duplicate_same_brand_box`, `dedup_cluster_id`, `dedup_representative` |
| 18 | box A mostly contained in box B (`containment >= 0.80`), same `canonical_brand`, B not oversized (`box_area_ratio < 0.45`) | keep B, A → DEDUPED `subset_box_same_brand`; if B is oversized and A `has_valid_tight_bbox` → keep A, B → DEDUPED `oversized_container_replaced_by_tight_box` |

Representative selection order (18 July): decision priority ACCEPT > REVIEW > UNKNOWN_LOGO > REJECT; `logo` over `partial_logo`; non-truncated; higher `ocr_fuzzy_score`; higher `confidence`; `has_valid_tight_bbox`; larger box only if not oversized. The same brand in genuinely separate placements survives as separate rows because the cluster test is geometric.

**Shared dedup function** `router/dedup.py: cluster_boxes(boxes, iou_t, contain_t, area_ratio_t, min_area_fraction=None)` is also what `stages/merge.py` calls with the merge keys — one implementation, two call sites, different YAML keys (T45). The merge site is brand-agnostic; the router site groups on `canonical_brand`.

### Block 6 — Default

`REVIEW default_uncertain_or_insufficient_evidence`, `review_priority = normal`. A `not_logo` with mid confidence lands here by design (24 July); the decision distribution reports it by reason code.

## Part D — Reason-code enum (T44)

`router/reason_codes.py` defines the enum; `configs/reason_codes.md` is generated from it and is a deliverable. v1 members (rule → codes):

```
R1  high_confidence_not_logo | low_logo_probability_not_logo
R2  non_brand_content_generic_word | non_brand_content_ingredient | non_brand_content_slogan | non_brand_content_ui_icon | non_brand_content_sign   (unreachable in v1)
R3  not_a_brand_identifier   (unreachable in v1)
R4  oversized_scene_region_without_tight_logo
R5  channel_brand_outside_watermark_region | channel_brand_needs_watermark_check | likely_channel_audio_contamination (disabled)
R6  clear_logo_with_strong_ocr_match
R7  clear_logo_with_strong_visual_database_match (disabled)
R8  high_confidence_visual_logo_no_conflicts (disabled)
R9  partial_logo_with_strong_ocr_support | partial_logo_default_review
R10 truncated_logo_requires_review
R11 weak_visual_crop_rescued_by_audio_hint | audio_only_brand_prediction
R12 faiss_only_or_faiss_conflicting_prediction | faiss_score_not_discriminative_enough
R13 ocr_vlm_brand_conflict
R14 faiss_vlm_brand_conflict
R15 logo_like_but_brand_unknown
R16 readable_brand_like_text_not_in_database
R17 duplicate_same_brand_box
R18 subset_box_same_brand | oversized_container_replaced_by_tight_box
D   default_uncertain_or_insufficient_evidence
X   vlm_output_unparsed          # [ASSUMPTION] rows with vlm_parse_ok false → REVIEW with this code (block 6 special case)
```

Each entry in the generated document carries: rule number, block, decision, the condition text, the fields read, and "what a reviewer should check".

## Part E — `not_logo_confidence` before T31 lands

Rule 1 primary needs `not_logo_confidence`. Until the scored field exists (MM9), `router.not_logo_confidence_source: generated` derives it from the M2 `confidence` field on `not_logo` rows (`confidence / 100`), which is the field the M2 prompt defined as "certainty that the region is truly not a logo". The 24 July disposition accepted the primary variant "OK now". After T32's distribution check, the key switches to `scored`. The source is printed in the acceptance evidence. **[ASSUMPTION]** listed in `19`.

## Part F — Trust / risk scoring layer (18 July §8)

Computed exactly as the 18 July pseudocode with the same constants (in the YAML under `router.scoring_layer`), written to `trust_score`, `risk_score`, `router_confidence = clip((trust − risk + 10) / 20, 0, 1)` **[ASSUMPTION]** for the fifth tuple member; never used for a decision.

## Part G — Firing log (T46)

`firing_log.jsonl`, one line per row: `det_id, frame_idx, deciding_rule, reason_code, decision, rules_matched_preempted: [ {rule, reason_code} ], dedup: {cluster_id, representative}`. `eval/ordering_checks.py` (T65) reads this file:

1. No row with `vlm_decision == partial_logo` carries an UNKNOWN_LOGO reason code.
2. No row that satisfied an enabled accept-rule condition (6, 7, 8 as enabled) received a Rule 16 decision.

Both are computed from `rules_matched_preempted` plus the decision, so they cost nothing.

## Definition of done

- [ ] Every rule is a function `rule_n(inp: RouterInput, cfg) -> Verdict | None` with a unit test per branch (`21`).
- [ ] The engine refuses to start when an enabled rule's required field is not produced (test: enable `rule_1_fallback_logprob` without scores).
- [ ] Ordering tests: a `partial_logo` row with `UNKNOWN` brand → Rule 9, not 15; a `logo` row with strong OCR and `ocr_fuzzy_score 96` → Rule 6, and Rule 16 appears in `rules_matched_preempted` only if its condition held.
- [ ] Dedup tests: two same-brand boxes at IoU 0.6 → one DEDUPED; same-brand boxes far apart → both kept; tight box inside an oversized same-brand box → oversized one DEDUPED.
- [ ] `configs/reason_codes.md` generated; canonicalisation report generated on the acceptance run.

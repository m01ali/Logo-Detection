# 10 — MM4 OCR Verification and Brand Matching (T25–T29)

**Effort:** 1 day. **Device:** CPU (GPU optional). **Gates:** MM5 (six rules read the OCR fields: 6, 9, 11, 12, 13, 16; Rule 5 too).

## Stage contract

`stages/ocr.py` (the verification pass) reads `refinement.jsonl` (for `selected_bbox`), `frames/`, `audio/audio_hints.json`, the brand list; writes `ocr.jsonl` with the section-D fields of `05`. It runs **after** refinement, on `selected_bbox` (A9). The refinement-side cheap call in MM3 and this pass share one `easyocr.Reader(languages, gpu)` instance created lazily once per process.

## T25 — OCR on `selected_bbox`, normalised

1. Cut `selected_bbox` from the frame; upscale ×3 if min side < 80 (M2). Read lines.
2. If no line is returned and `fallback_to_enlarged_crop` is true, read the 1.5× enlarged crop and set `ocr_source = enlarged_crop` (M2 behaviour, kept so the VLM hint text is unchanged).
3. `ocr_text_raw` = lines joined with " | ". `ocr_text` = normalised: lowercase → unicode NFKD → drop combining marks → keep `[a-z0-9 ]` → collapse whitespace. `ocr_max_confidence` = max line confidence.

## T26 — Match against the canonical list plus the video's audio brands

Candidate set = canonical names ∪ aliases (from the snapshot) ∪ normalised `audio_brands` for this video. Never the FAISS top-k (27 July C7). Each candidate carries its list tag (`canonical` or `audio`). The snapshot version and hash are written on every row (`brand_list_version`, `brand_list_hash`).

Queries: each normalised OCR line separately, and the concatenation of all lines (wordmarks split across lines, e.g. "rai" + "sport"). Result = the maximum-score (candidate, query) pair; ties prefer `canonical` over `audio`, then the longer candidate.

## T27 — Length-dependent matching

```
def match_one(query, candidate, cfg):
    short = len(query) <= cfg.short_name_cutoff or len(candidate) <= cfg.short_name_cutoff
    if short:
        return (100, "exact_short") if query == candidate else (0, "exact_short")
    return (WRatio(query, candidate), "fuzzy")
```

`short_name_cutoff` v1 = 4; `general_fuzzy_threshold` v1 = 90. Both in the YAML with the Rule 6 key (95) separate (T72). The rule "fires when either string is short" is exactly the `or` above. Consequence recorded for the client (29 July): HP, LG, BP, GE, 3M-style names only ever match on an exact read.

The scorer is rapidfuzz `WRatio` on normalised strings (the M2/Phase 1 choice); `smart_score` from `utils/fuzzy_matching.py` is available as an alternative behind `ocr.matching.scorer` but is not the default because its Jaro-Winkler branch inflates short-string scores, the very problem T27 addresses.

## T28 — No auto-quarantine

A short string that fails exact matching yields `ocr_brand_match = None`, `ocr_fuzzy_score = 0` if nothing else matched, and the row proceeds; no rule reads "OCR was short". Similarly a fuzzy best score below the general threshold yields `ocr_brand_match = None` but **`ocr_fuzzy_score` keeps the best score** (rules threshold on the score, e.g. Rule 16 `< 80`, Rule 11 `< 80`, Rule 13 `≥ 85`), and `ocr_match_candidate` records what that score was against.

## T29 — What is recorded

| Field | Value |
|-------|-------|
| `ocr_brand_match` | canonical spelling (aliases resolved) or audio brand string, only when score ≥ general threshold (or exact for short) |
| `ocr_fuzzy_score` | best score 0–100 regardless of threshold |
| `ocr_match_list` | `canonical` / `audio` / null |
| `ocr_match_mode` | `exact_short` / `fuzzy` / null |
| `ocr_match_candidate` | the candidate behind the best score even when below threshold (diagnostic) |
| `ocr_match_query` | which query string won (line or concatenation) |

## Interaction with the VLM prompt

The hinted VLM call shows `ocr_text` lines with confidences exactly as M2 did (the prompt is not redesigned). It does **not** show `ocr_brand_match`; the match is a router input, not a hint. That keeps the Rule 6 caveat as stated in Revision 3 (auto-accept rests on the OCR fuzzy threshold) rather than making it worse.

## Definition of done

- [ ] `ocr.jsonl` for every row; fields per table.
- [ ] Unit tests: `"NUROFEN"` vs canonical `nurofen`-like entry → fuzzy 100; `"hp"` vs `hp` → exact_short 100; `"hb"` vs `hp` → 0; `"RAI SPORT HD"` → `rai sport` via alias; `"very sure"` from audio → `verisure` via alias, list `canonical` (alias resolves to canonical even though the string came from audio); accent stripping `"L'Oréal"` → `loreal`.
- [ ] The MM1 acceptance run re-executed `--from-stage refine --to-stage ocr` on the Mac.

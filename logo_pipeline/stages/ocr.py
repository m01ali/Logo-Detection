"""``stages/ocr.py`` (T25-T29, ``spec docs/10``): the OCR verification
pass. Reads the crop cut from ``selected_bbox`` (after refinement — A9),
falls back to the enlarged crop when empty (M2 behaviour, ``ocr_image`` /
``compute_ocr_hints`` in ``generate_hints.py``) so the VLM prompt's
``ocr_text`` section is unchanged (T25); then matches the normalised lines
against the canonical brand list plus this video's audio brands (T26-T29).
Shares one EasyOCR reader singleton with ``stages/refine.py``'s heuristic-2
cheap call.
"""

from __future__ import annotations

from ..brands import BrandList
from ..brands import normalise as brand_normalise
from ..crops import tight_and_enlarged
from .base import (AUDIO_HINTS_FILE, MERGED_DETECTIONS_FILE, OCR_FILE, REFINEMENT_FILE, RunContext, Stage,
                    StageResult)

_READERS: dict[tuple, object] = {}


def get_reader(languages: list[str], gpu: bool):
    """Shared EasyOCR singleton, also used by ``stages/refine.py``'s
    heuristic-2 cheap call so the two stages don't each load their own copy
    of the model."""
    key = (tuple(languages), gpu)
    if key not in _READERS:
        import easyocr
        _READERS[key] = easyocr.Reader(list(languages), gpu=gpu)
    return _READERS[key]


def resolve_gpu(mode) -> bool:
    if mode in (True, False):
        return mode
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def ocr_image(reader, image, min_side: int = 80, upscale: int = 3) -> list[dict]:
    """M2 ``ocr_image``: upscale tiny crops first so text is legible."""
    import numpy as np
    from PIL import Image

    img = image.convert("RGB")
    if min(img.size) < min_side:
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
    results = reader.readtext(np.array(img))
    return [{"text": text.strip(), "confidence": round(float(conf), 3)}
            for _, text, conf in results if text.strip()]


# ---------------------------------------------------------------------------
# T26-T29: brand matching
# ---------------------------------------------------------------------------

def build_candidate_set(brands: BrandList, audio_brands: list[str], match_lists: list[str]) -> list[dict]:
    """T26: canonical names ∪ aliases (tagged ``canonical``) ∪ this video's
    normalised audio brands (tagged ``audio``) — never the FAISS top-k (27
    July C7). ``match_lists`` (04) can drop either source. ``canonical_form``
    is what ``ocr_brand_match`` reports when this candidate wins and clears
    the threshold; ``display`` is the literal string that was scored (an
    alias spelling stays itself here even though it resolves to its
    canonical name in ``canonical_form``)."""
    candidates: list[dict] = []
    if "canonical" in match_lists:
        for c in brands.canonical:
            candidates.append({"text": brand_normalise(c), "display": c, "list": "canonical", "canonical_form": c})
        for e in brands.entries:
            if e.deprecated:
                continue
            for a in e.aliases:
                k = brand_normalise(a)
                if k:
                    candidates.append({"text": k, "display": a, "list": "canonical", "canonical_form": e.canonical})
    if "audio" in match_lists:
        for b in audio_brands:
            norm = brand_normalise(b)
            if norm:
                candidates.append({"text": norm, "display": b, "list": "audio", "canonical_form": b})
    return candidates


def get_scorer(name: str):
    """``ocr.matching.scorer`` (04): WRatio is the M2/Phase-1 default;
    ``smart_score`` (``utils/fuzzy_matching.py``) is available but not
    default because its Jaro-Winkler branch inflates short-string scores —
    the exact problem T27's length-dependent matching addresses."""
    from rapidfuzz import fuzz
    if name == "token_set_ratio":
        return fuzz.token_set_ratio
    if name == "smart_score":
        from utils.fuzzy_matching import smart_score
        return smart_score
    return fuzz.WRatio


def match_one(query: str, candidate: str, short_cutoff: int, scorer_fn) -> tuple[int, str]:
    """T27: exact match when either string is short (<= ``short_name_cutoff``
    — "fires when either string is short"), else the configured fuzzy
    scorer. Consequence for the client (29 July): HP/LG/BP/GE/3M-style names
    only ever match on an exact read."""
    if len(query) <= short_cutoff or len(candidate) <= short_cutoff:
        return (100, "exact_short") if query == candidate else (0, "exact_short")
    return int(round(scorer_fn(query, candidate))), "fuzzy"


def _better(current, candidate):
    """Tie-break (T26): higher score wins; ties prefer ``canonical`` over
    ``audio``, then the longer candidate string."""
    if current is None:
        return candidate
    if candidate[0] != current[0]:
        return candidate if candidate[0] > current[0] else current
    c_canon = current[2]["list"] == "canonical"
    n_canon = candidate[2]["list"] == "canonical"
    if n_canon != c_canon:
        return candidate if n_canon else current
    if len(candidate[2]["text"]) != len(current[2]["text"]):
        return candidate if len(candidate[2]["text"]) > len(current[2]["text"]) else current
    return current


def _resolve_match(best, mcfg) -> dict:
    if best is None:
        return {"ocr_brand_match": None, "ocr_fuzzy_score": 0, "ocr_match_list": None,
                "ocr_match_mode": None, "ocr_match_candidate": None, "ocr_match_query": None}
    score, mode, cand, query = best
    passes = (score == 100) if mode == "exact_short" else (score >= mcfg.general_fuzzy_threshold)
    return {
        "ocr_brand_match": cand["canonical_form"] if passes else None,   # T28: no auto-quarantine either way
        "ocr_fuzzy_score": score,                                        # best score, regardless of threshold
        "ocr_match_list": cand["list"], "ocr_match_mode": mode,
        "ocr_match_candidate": cand["display"], "ocr_match_query": query,
    }


def match_ocr_text(norm_lines: list[str], candidates: list[dict], scorer_fn, mcfg) -> dict:
    """T26: queries are each normalised OCR line separately, plus their
    concatenation (wordmarks split across lines, e.g. "rai" + "sport").
    Result = the max-score (query, candidate) pair over every combination."""
    queries = list(dict.fromkeys(q for q in norm_lines if q))
    if norm_lines:
        concat = " ".join(q for q in norm_lines if q)
        if concat and concat not in queries:
            queries.append(concat)

    best = None
    for query in queries:
        for c in candidates:
            score, mode = match_one(query, c["text"], mcfg.short_name_cutoff, scorer_fn)
            best = _better(best, (score, mode, c, query))
    return _resolve_match(best, mcfg)


class OcrStage(Stage):
    name = "ocr"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE, REFINEMENT_FILE]
    outputs = [OCR_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        ocfg = ctx.cfg.ocr
        mcfg = ocfg.matching
        reps = {r["det_id"]: r for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        refined = {r["det_id"]: r for r in ctx.read_jsonl(REFINEMENT_FILE)}
        done = {r["det_id"] for r in ctx.read_jsonl(OCR_FILE)} if not ctx.force else set()

        todo = [det_id for det_id in reps if det_id not in done]
        if not todo:
            ctx.log.stage_skipped(self.name, "all detections already OCR'd")
            return StageResult(counts={"skipped": len(reps)})

        reader = get_reader(ocfg.languages, resolve_gpu(ocfg.gpu))

        audio_brands: list[str] = []
        if ctx.path(AUDIO_HINTS_FILE).exists():
            audio_brands = ctx.read_json(AUDIO_HINTS_FILE).get("brands", [])
        candidates = build_candidate_set(ctx.brands, audio_brands, mcfg.match_lists)
        scorer_fn = get_scorer(mcfg.scorer)

        frame_cache: dict[int, object] = {}
        rows = ctx.read_jsonl(OCR_FILE) if not ctx.force else []
        rows = [r for r in rows if r["det_id"] not in todo]

        for i, det_id in enumerate(sorted(todo), 1):
            rep = reps[det_id]
            frame_idx = rep["frame_idx"]
            box = refined.get(det_id, {}).get("selected_bbox") or rep["loose_bbox"]
            if frame_idx not in frame_cache:
                frame_cache[frame_idx] = ctx.open_frame_image(frame_idx)
            tight, enlarged, _ = tight_and_enlarged(frame_cache[frame_idx], box, ctx.cfg.crops.enlarge_factor,
                                                     rep["frame_width"], rep["frame_height"])

            lines: list[dict] = []
            source: str | None = None
            if tight is not None:
                lines = ocr_image(reader, tight, ocfg.upscale_min_side, ocfg.upscale_factor)
                source = "tight_crop" if lines else None
            if not lines and ocfg.fallback_to_enlarged_crop and enlarged is not None:
                lines = ocr_image(reader, enlarged, ocfg.upscale_min_side, ocfg.upscale_factor)
                source = "enlarged_crop" if lines else None

            raw_texts = [l["text"] for l in lines]
            norm_texts = [brand_normalise(t) for t in raw_texts if brand_normalise(t)]
            match = match_ocr_text(norm_texts, candidates, scorer_fn, mcfg)

            rows.append({
                "det_id": det_id,
                "ocr_text": " | ".join(norm_texts),
                "ocr_text_raw": " | ".join(raw_texts),
                "ocr_source": source,
                "ocr_max_confidence": max((l["confidence"] for l in lines), default=None),
                "ocr_lines": lines,
                **match,
            })
            ctx.log.progress(i, len(todo), unit="det")

        header = ctx.header("ocr", languages=ocfg.languages, scorer=mcfg.scorer,
                             general_fuzzy_threshold=mcfg.general_fuzzy_threshold,
                             short_name_cutoff=mcfg.short_name_cutoff, match_lists=mcfg.match_lists)
        ctx.write_jsonl(OCR_FILE, sorted(rows, key=lambda r: r["det_id"]), header=header)
        matched = sum(1 for r in rows if r["ocr_brand_match"] is not None)
        return StageResult(counts={"detections_ocrd": len(todo), "brand_matches": matched})

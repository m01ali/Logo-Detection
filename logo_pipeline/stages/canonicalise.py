"""``stages/canonicalise.py`` (T37, ``spec docs/11`` Part A): the deterministic
map of every free-text brand string onto the canonical list.

``canonicalise`` is a pure function of ``(string, brand list, config,
audio targets)`` so the router's input builder (which must canonicalise
``ocr_brand_match`` too) and this stage share one implementation. The
optional LLM merge (off by default) only ever runs over ``unmatched``
strings, records ``mode = llm_merge`` on the rows it touches, and saves the
raw mapping next to the run so it is never a silent substitution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..brands import BrandList, normalise
from .base import (AUDIO_HINTS_FILE, CANONICAL_FILE, FAISS_HINTS_FILE, MERGED_DETECTIONS_FILE, VLM_BLIND_FILE,
                    VLM_HINTED_FILE, RunContext, Stage, StageResult)

UNKNOWN = "UNKNOWN"
LLM_MERGE_FILE = "canonical_llm_merge.json"


@dataclass(frozen=True)
class Canonical:
    canonical: str | None
    mode: str | None
    score: int | None


def _fuzzy_target(norm: str, targets: list[str], threshold: int, short_cutoff: int) -> tuple[str | None, int]:
    """Length-rule matching of ``norm`` against already-normalised ``targets`` (T27's rule,
    applied to the audio brands): short strings only match exactly."""
    if not targets:
        return None, 0
    if norm in targets:
        return norm, 100
    if len(norm) <= short_cutoff:
        return None, 0
    from rapidfuzz import fuzz, process
    long_targets = [t for t in targets if len(t) > short_cutoff]
    if not long_targets:
        return None, 0
    best = process.extractOne(norm, long_targets, scorer=fuzz.WRatio, processor=None)
    if best is None:
        return None, 0
    score = int(round(float(best[1])))
    return (best[0], score) if score >= threshold else (None, score)


def canonicalise_to_list(s: str | None, brands: BrandList, ccfg) -> Canonical:
    """Steps 1-4 only: nothing but the canonical list is a target. An unmatched
    string keeps its normalised form so audio brands can still be compared exactly."""
    if s is None:
        return Canonical(None, None, None)
    if s == UNKNOWN:
        return Canonical(UNKNOWN, "unknown", None)
    norm = normalise(s)
    if not norm:
        return Canonical(norm, "unmatched", 0)
    exact, mode = brands.map_exact_with_mode(norm)
    if exact is not None:
        return Canonical(exact, mode, 100)
    fuzzy, score, _ = brands.map_fuzzy(norm, ccfg.fuzzy_threshold, ccfg.short_name_cutoff)
    if fuzzy is not None:
        return Canonical(fuzzy, "fuzzy", score)
    return Canonical(norm, "unmatched", score)


def canonicalise(s: str | None, brands: BrandList, ccfg, audio_canonical: list[str] | tuple[str, ...] = ()) -> Canonical:
    """T37's ``canonicalise(s) -> (canonical, mode, score)``. ``audio_canonical`` are the
    video's audio brands *already* run through steps 1-4 (so ``very sure`` is ``verisure``)."""
    first = canonicalise_to_list(s, brands, ccfg)
    if first.mode != "unmatched":
        return first
    if ccfg.audio_brands_as_targets and audio_canonical:
        targets = [normalise(a) for a in audio_canonical]
        hit, score = _fuzzy_target(first.canonical or "", targets, ccfg.fuzzy_threshold, ccfg.short_name_cutoff)
        if hit is not None:
            return Canonical(audio_canonical[targets.index(hit)], "audio", score)
        return Canonical(first.canonical, "unmatched", max(first.score or 0, score))
    return first


def canonicalise_audio_brands(audio_brands: list[str], brands: BrandList, ccfg) -> list[str]:
    """Each audio brand through steps 1-4; unmatched ones keep their normalised string."""
    out: list[str] = []
    for a in audio_brands:
        c = canonicalise_to_list(a, brands, ccfg).canonical
        if c and c != UNKNOWN and c not in out:
            out.append(c)
    return out


def _real(brand: str | None) -> bool:
    return brand not in (None, "", UNKNOWN)


def index_brand_keys(ctx: RunContext) -> set[str] | None:
    """Normalised brand strings of the FAISS index, or ``None`` when its metadata is not available."""
    path = ctx.resolve_repo_path(ctx.cfg.faiss.metadata_path)
    if not path.exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data if isinstance(data, list) else data.get("brands", [])
    return {normalise(n) for n in names if isinstance(n, str)}


def brand_in_database(canonical: str | None, brands: BrandList, index_keys: set[str] | None) -> bool | None:
    """``canonical_brand`` or any of its aliases equals a normalised index brand string."""
    if index_keys is None or not _real(canonical):
        return None if index_keys is None else False
    keys = {normalise(canonical)}
    entry = brands.entry(canonical)
    if entry is not None:
        keys.update(normalise(a) for a in entry.aliases)
    return bool(keys & index_keys)


def hinted_canonical(hinted: dict, brands: BrandList, ccfg, audio_canonical: list[str],
                     merge_map: dict[str, str] | None = None) -> Canonical:
    cb = canonicalise((hinted.get("parsed") or {}).get("brand"), brands, ccfg, audio_canonical)
    if merge_map and cb.mode == "unmatched" and merge_map.get(cb.canonical or "", cb.canonical) != cb.canonical:
        return Canonical(merge_map[cb.canonical], "llm_merge", cb.score)
    return cb


def derive_row(rep_id: int, hinted: dict, blind: dict, faiss: dict, brands: BrandList, ccfg,
               audio_canonical: list[str], index_keys: set[str] | None,
               merge_map: dict[str, str] | None = None) -> dict:
    vb = blind.get("parsed") or {}
    cb = hinted_canonical(hinted, brands, ccfg, audio_canonical, merge_map)
    blind_c = canonicalise(vb.get("brand"), brands, ccfg, audio_canonical).canonical
    top1_c = canonicalise(faiss.get("faiss_top1"), brands, ccfg, audio_canonical).canonical
    top5_c = [canonicalise(b, brands, ccfg, audio_canonical).canonical for b in faiss.get("faiss_top5_brands", [])]

    brand = cb.canonical
    agrees = None if (brand is None or top1_c is None) else (_real(brand) and brand == top1_c)
    return {
        "det_id": rep_id,
        "canonical_brand": brand, "canonical_match_mode": cb.mode, "canonical_match_score": cb.score,
        "brand_list_version": brands.version, "brand_list_hash": brands.content_hash,
        "faiss_top1_brand_canonical": top1_c, "faiss_agrees_with_vlm": agrees,
        "brand_in_database": brand_in_database(brand, brands, index_keys),
        "audio_brands_canonical": audio_canonical,
        "brand_from_audio": _real(brand) and brand in audio_canonical,
        "vlm_brand_blind_canonical": blind_c,
        "blind_hinted_brand_agree": None if (vb.get("brand") is None or brand is None) else brand == blind_c,
        "hinted_brand_in_faiss_hints": _real(brand) and brand in top5_c,
    }


def default_llm_merge(strings: list[str]) -> tuple[dict[str, str], str]:
    """The M2 ``canonicalize_brands`` LLM pass over a list of strings; needs the GPU/text model."""
    from generate_hints import CANONICALIZE_PROMPT, CANONICALIZE_SYSTEM_PROMPT, parse_brand_mapping
    from models.description_model import Qwen2_5TextModel

    model = Qwen2_5TextModel()
    raw = model.run_example(CANONICALIZE_PROMPT(strings), CANONICALIZE_SYSTEM_PROMPT)
    return parse_brand_mapping(raw, strings), raw


def run_llm_merge(hinted: dict[int, dict], brands: BrandList, ccfg, audio_canonical: list[str],
                  llm: Callable[[list[str]], tuple[dict[str, str], str]]) -> dict:
    """Runs ``llm`` over the distinct ``unmatched`` strings only (after step 6); returns the payload
    for ``canonical_llm_merge.json``. Rows are rewritten by ``hinted_canonical`` from ``changed``."""
    unmatched = sorted({c.canonical for h in hinted.values()
                        for c in [canonicalise((h.get("parsed") or {}).get("brand"), brands, ccfg, audio_canonical)]
                        if c.mode == "unmatched" and c.canonical})
    if not unmatched:
        return {"input": [], "mapping": {}, "raw_response": "", "changed": {}}
    mapping, raw = llm(unmatched)
    return {"input": unmatched, "mapping": mapping, "raw_response": raw,
            "changed": {k: v for k, v in mapping.items() if k != v}}


def _distribution_report(rows: list[dict]) -> str:
    from collections import Counter
    modes = Counter(r["canonical_match_mode"] or "null" for r in rows)
    unmatched = Counter(r["canonical_brand"] for r in rows if r["canonical_match_mode"] == "unmatched")
    lines = ["# Canonicalisation report (T37, open mode)", "",
             f"{len(rows)} detections.", "", "## canonical_match_mode", "", "| mode | rows |", "|---|---|"]
    lines += [f"| {m} | {n} |" for m, n in sorted(modes.items(), key=lambda kv: -kv[1])]
    lines += ["", "## Top unmatched strings (candidates for the alias table / add-a-brand)", ""]
    if unmatched:
        lines += ["| normalised string | rows |", "|---|---|"]
        lines += [f"| {s} | {n} |" for s, n in unmatched.most_common(25)]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


class CanonicaliseStage(Stage):
    name = "canonicalise"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE]
    outputs = [CANONICAL_FILE]

    def __init__(self, llm_merge: Callable | None = None):
        self._llm_merge = llm_merge

    def run(self, ctx: RunContext) -> StageResult:
        ccfg = ctx.cfg.canonicalisation
        reps = ctx.read_jsonl(MERGED_DETECTIONS_FILE)
        hinted = {r["det_id"]: r for r in ctx.read_jsonl(VLM_HINTED_FILE)}
        blind = {r["det_id"]: r for r in ctx.read_jsonl(VLM_BLIND_FILE)}
        faiss = {r["det_id"]: r for r in ctx.read_jsonl(FAISS_HINTS_FILE)}
        audio_brands = (ctx.read_json(AUDIO_HINTS_FILE).get("brands", [])
                        if ctx.path(AUDIO_HINTS_FILE).exists() else [])
        audio_canonical = canonicalise_audio_brands(audio_brands, ctx.brands, ccfg)
        index_keys = index_brand_keys(ctx) if ctx.cfg.signals.faiss.compute_and_log else None

        merge_map = None
        if ccfg.llm_merge.enabled:
            payload = run_llm_merge(hinted, ctx.brands, ccfg, audio_canonical, self._llm_merge or default_llm_merge)
            ctx.write_json(LLM_MERGE_FILE, payload)
            ctx.log.info("canonicalise: LLM merge changed %d strings: %s", len(payload["changed"]), payload["changed"])
            merge_map = payload["changed"]

        rows = [derive_row(r["det_id"], hinted.get(r["det_id"], {}), blind.get(r["det_id"], {}),
                           faiss.get(r["det_id"], {}), ctx.brands, ccfg, audio_canonical, index_keys, merge_map)
                for r in reps]

        header = ctx.header("canonicalise", llm_merge=ccfg.llm_merge.enabled)
        ctx.write_jsonl(CANONICAL_FILE, rows, header=header)
        report = ctx.path("reports/canonicalisation.md")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(_distribution_report(rows), encoding="utf-8")
        return StageResult(counts={"rows": len(rows), "unmatched": sum(
            1 for r in rows if r["canonical_match_mode"] == "unmatched")})

"""``stages/faiss_query.py`` (07 MM1 Step 5.2): wraps
``generate_hints.load_faiss_db`` + the ``faiss_top_k`` dedup-by-brand logic;
computes ``faiss_score_gap``. Crops are cut from ``selected_bbox`` (equal to
``loose_bbox`` until MM3, 09) directly from the sampled frame, never from a
file on disk (03 §3).

GPU preferred (CLIP embedding of every crop) but runs on CPU (M2 default).
Governed by ``signals.faiss.compute_and_log`` (T10): when false the fields
are written null and the index is never loaded.
"""

from __future__ import annotations

from ..crops import cut_crop
from .base import FAISS_HINTS_FILE, MERGED_DETECTIONS_FILE, REFINEMENT_FILE, RunContext, Stage, StageResult

_DB_CACHE: dict[tuple, object] = {}


def _get_db(index_path: str, metadata_path: str):
    key = (index_path, metadata_path)
    if key not in _DB_CACHE:
        from generate_hints import load_faiss_db
        _DB_CACHE[key] = load_faiss_db(index_path, metadata_path)
    return _DB_CACHE[key]


def faiss_top_k_image(db, image, top_k: int = 5, overfetch_factor: int = 3, threshold: float = 0.0) -> list[dict]:
    """``generate_hints.faiss_top_k`` re-expressed over an in-memory crop
    instead of a file path (03 §3 rule: crops are never pre-baked to disk).
    Same dedup-by-brand / top-k logic, same ``db.search_logo`` call."""
    raw = db.search_logo(image, threshold=threshold, k=top_k * overfetch_factor)
    seen: dict[str, float] = {}
    for r in raw:
        brand = r["brand_name"]
        if brand not in seen or r["similarity"] > seen[brand]:
            seen[brand] = r["similarity"]
    return sorted(
        [{"brand_name": b, "similarity": round(s, 4)} for b, s in seen.items()],
        key=lambda x: x["similarity"], reverse=True,
    )[:top_k]


class FaissQueryStage(Stage):
    name = "faiss_query"
    device = "gpu"
    inputs = [MERGED_DETECTIONS_FILE, REFINEMENT_FILE]
    outputs = [FAISS_HINTS_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        fcfg = ctx.cfg.faiss
        sw = ctx.cfg.signals.faiss
        reps = {r["det_id"]: r for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        refined = {r["det_id"]: r for r in ctx.read_jsonl(REFINEMENT_FILE)}

        if not sw.compute_and_log:
            rows = [{"det_id": d, "faiss_top1": None, "faiss_top1_score": None,
                      "faiss_top5_brands": [], "faiss_top5_scores": [], "faiss_score_gap": None}
                     for d in reps]
            header = ctx.header("faiss_query", compute_and_log=False)
            ctx.write_jsonl(FAISS_HINTS_FILE, rows, header=header)
            ctx.log.stage_skipped(self.name, "signals.faiss.compute_and_log is false")
            return StageResult(counts={"detections": 0, "disabled": True})

        done = {r["det_id"] for r in ctx.read_jsonl(FAISS_HINTS_FILE)} if not ctx.force else set()
        todo = [d for d in reps if d not in done]
        rows = [r for r in ctx.read_jsonl(FAISS_HINTS_FILE) if r["det_id"] not in todo] if not ctx.force else []

        if todo:
            db = _get_db(str(ctx.resolve_repo_path(fcfg.index_path)), str(ctx.resolve_repo_path(fcfg.metadata_path)))
            ctx.log.info("faiss_query: index has %d vectors", db.index.ntotal)
            frame_cache: dict[int, object] = {}
            for i, det_id in enumerate(sorted(todo), 1):
                rep = reps[det_id]
                frame_idx = rep["frame_idx"]
                box = refined.get(det_id, {}).get("selected_bbox") or rep["loose_bbox"]
                if frame_idx not in frame_cache:
                    frame_cache[frame_idx] = ctx.open_frame_image(frame_idx)
                crop = cut_crop(frame_cache[frame_idx], box)
                hits = faiss_top_k_image(db, crop, fcfg.top_k, fcfg.overfetch_factor,
                                          fcfg.similarity_threshold) if crop is not None else []
                brands = [h["brand_name"] for h in hits]
                scores = [h["similarity"] for h in hits]
                rows.append({
                    "det_id": det_id,
                    "faiss_top1": brands[0] if brands else None,
                    "faiss_top1_score": scores[0] if scores else None,
                    "faiss_top5_brands": brands, "faiss_top5_scores": scores,
                    "faiss_score_gap": round(scores[0] - scores[1], 4) if len(scores) >= 2 else None,
                })
                ctx.log.progress(i, len(todo), unit="det")

        header = ctx.header("faiss_query", compute_and_log=True, top_k=fcfg.top_k, embedder=fcfg.embedder)
        ctx.write_jsonl(FAISS_HINTS_FILE, sorted(rows, key=lambda r: r["det_id"]), header=header)
        return StageResult(counts={"detections_queried": len(todo)})

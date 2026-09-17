"""``stages/ocr.py`` (07 MM1 Step 5.3, T25 hint form; matching is T26–T29
in MM4). Reader singleton; reads the tight crop cut from ``selected_bbox``,
falls back to the enlarged crop when empty (M2 behaviour, ``ocr_image`` /
``compute_ocr_hints`` in ``generate_hints.py``) so the VLM prompt's
``ocr_text`` section is unchanged. ``ocr_brand_match``/``ocr_fuzzy_score``/
``ocr_match_list``/``ocr_match_mode`` stay null/0 until MM4.
"""

from __future__ import annotations

from ..brands import normalise as brand_normalise
from ..crops import tight_and_enlarged
from .base import MERGED_DETECTIONS_FILE, OCR_FILE, REFINEMENT_FILE, RunContext, Stage, StageResult

_READERS: dict[tuple, object] = {}


def _get_reader(languages: list[str], gpu: bool):
    key = (tuple(languages), gpu)
    if key not in _READERS:
        import easyocr
        _READERS[key] = easyocr.Reader(list(languages), gpu=gpu)
    return _READERS[key]


def _resolve_gpu(mode) -> bool:
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


class OcrStage(Stage):
    name = "ocr"
    device = "cpu"
    inputs = [MERGED_DETECTIONS_FILE, REFINEMENT_FILE]
    outputs = [OCR_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        ocfg = ctx.cfg.ocr
        reps = {r["det_id"]: r for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        refined = {r["det_id"]: r for r in ctx.read_jsonl(REFINEMENT_FILE)}
        done = {r["det_id"] for r in ctx.read_jsonl(OCR_FILE)} if not ctx.force else set()

        todo = [det_id for det_id in reps if det_id not in done]
        if not todo:
            ctx.log.stage_skipped(self.name, "all detections already OCR'd")
            return StageResult(counts={"skipped": len(reps)})

        reader = _get_reader(ocfg.languages, _resolve_gpu(ocfg.gpu))
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
            rows.append({
                "det_id": det_id,
                "ocr_text": " | ".join(norm_texts),
                "ocr_text_raw": " | ".join(raw_texts),
                "ocr_source": source,
                "ocr_max_confidence": max((l["confidence"] for l in lines), default=None),
                "ocr_lines": lines,
                # Matching (T26-T29) is MM4; these stay null/0 until then.
                "ocr_brand_match": None, "ocr_fuzzy_score": 0,
                "ocr_match_list": None, "ocr_match_mode": None,
            })
            ctx.log.progress(i, len(todo), unit="det")

        header = ctx.header("ocr", languages=ocfg.languages, note="MM1: hint form only; matching is MM4 (T26-T29)")
        ctx.write_jsonl(OCR_FILE, sorted(rows, key=lambda r: r["det_id"]), header=header)
        return StageResult(counts={"detections_ocrd": len(todo)})

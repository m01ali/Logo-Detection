"""``stages/vlm.py`` (07 MM1 Step 5.4, MM9 T33/T34/T67): the hinted and blind
VLM calls per representative.

Wraps ``generate_hints._build_vlm_content`` and the tokenise/generate/parse
tail of ``generate_hints.run_vlm_on_detection``. Image loading is
reimplemented rather than wrapped: ``run_vlm_on_detection``'s loader assumes
M2's ``extraction_output/`` layout (pre-baked ``crop.jpg`` files), but crop
JPEGs are written only by ``stages/output.py`` (05 section C), which runs
after ``vlm``. This stage cuts its own crops in memory from ``frames/`` (03 §3).

**Hinted** (``vlm.hinted_pass``): image signals + every text signal whose
``compute_and_log and include_in_vlm_prompt`` is true, the open-vocabulary
instruction, the watermark line, the UNKNOWN option -> ``vlm/hinted.jsonl``,
plus ``vlm/hint_composition.jsonl`` (T34).

**Blind** (``vlm.blind_pass``): ``vlm.blind_pass_signals`` (image signals
only) and the same instruction minus every hint reference and the watermark
line -> ``vlm/blind.jsonl``. Every signal was computed by its own stage
either way; the blind call simply does not render the text ones.

Both files are appended one record at a time and resume per ``det_id``, so a
crash keeps its progress and the blind pass can be added to a run directory
that already has the hinted one (``run --stages vlm --blind-only``).
"""

from __future__ import annotations

import json as _json
import re
import time

from PIL import Image

from ..config import text_signal_in_prompt
from ..crops import tight_and_enlarged
from .base import (AUDIO_HINTS_FILE, FAISS_HINTS_FILE, MERGED_DETECTIONS_FILE, OCR_FILE, REFINEMENT_FILE,
                    VLM_BLIND_FILE, VLM_HINTED_FILE, VLM_HINT_COMPOSITION_FILE, JsonlAppender, RunContext,
                    Stage, StageResult)

IMAGE_SIGNALS = ("tight_crop", "enlarged_crop", "full_frame", "temporal_context")
_MODEL_CACHE: dict[tuple, tuple] = {}


def _get_model(model_id: str, quantisation: str, max_pixels: int):
    key = (model_id, quantisation)
    if key not in _MODEL_CACHE:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        kwargs: dict = {}
        if quantisation == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif quantisation == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", **kwargs).eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left", max_pixels=max_pixels)
        _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]


def _hint_composition(audio_brands, faiss_hits, ocr_row, channel_watermark) -> list[dict]:
    """T34 hint-composition entries."""
    items: list[dict] = []
    if channel_watermark:
        items.append({"item": channel_watermark, "source": "watermark"})
    for b in audio_brands:
        items.append({"item": b, "source": "audio"})
    for rank, h in enumerate(faiss_hits, 1):
        items.append({"item": h["brand_name"], "source": "faiss", "rank": rank, "score": h["similarity"]})
    if ocr_row and ocr_row.get("ocr_text"):
        items.append({"item": ocr_row["ocr_text"], "source": "ocr", "confidence": ocr_row.get("ocr_max_confidence")})
    return items


def parse_reply(raw_text: str) -> dict | None:
    """The first JSON object in the reply, with ``signals_used`` normalised (M2 behaviour)."""
    from generate_hints import normalize_signals_used
    parsed = None
    try:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            parsed = _json.loads(m.group())
    except _json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return None
    if "signals_used" in parsed:
        parsed["signals_used_raw"] = parsed.get("signals_used")
        parsed["signals_used"] = normalize_signals_used(parsed.get("signals_used"))
    return parsed


def build_messages(content: list[dict]) -> list[dict]:
    return [{"role": "user", "content": content}]


def generate(model, processor, content: list[dict], vcfg) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = build_messages(content)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                       return_tensors="pt").to(model.device, model.dtype)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=vcfg.max_new_tokens,
                                       do_sample=(vcfg.decoding != "greedy"))
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class DetectionInputs:
    """Everything one representative's VLM calls need, read once from the run directory."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.reps = {r["det_id"]: r for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        self.refined = {r["det_id"]: r for r in ctx.read_jsonl(REFINEMENT_FILE)}
        self.ocr = {r["det_id"]: r for r in ctx.read_jsonl(OCR_FILE)} if ctx.path(OCR_FILE).exists() else {}
        self.faiss = ({r["det_id"]: r for r in ctx.read_jsonl(FAISS_HINTS_FILE)}
                      if ctx.path(FAISS_HINTS_FILE).exists() else {})
        audio = ctx.read_json(AUDIO_HINTS_FILE) if ctx.path(AUDIO_HINTS_FILE).exists() else {"brands": []}
        self.audio_brands = audio.get("brands", [])
        self.sampled = {f["frame_idx"]: f for f in ctx.sampled_frames()}
        cfg = ctx.cfg
        image = {n for n in IMAGE_SIGNALS if getattr(cfg.signals, n).include_in_vlm_prompt}
        self.text_signals = {n for n in ("audio", "faiss", "ocr") if text_signal_in_prompt(cfg, n)}
        names = {"audio": "audio_brands", "faiss": "faiss_matches", "ocr": "ocr_text"}
        self.hinted_signals = image | {names[n] for n in self.text_signals}
        self.blind_signals = set(cfg.vlm.blind_pass_signals) & set(IMAGE_SIGNALS)
        self.channel_watermark = (ctx.video.channel_brand or "") if cfg.vlm.channel_watermark_line else ""

    def images(self, det_id: int, frame_img: Image.Image):
        rep = self.reps[det_id]
        box = self.refined.get(det_id, {}).get("selected_bbox") or rep["loose_bbox"]
        tight, enlarged, _ = tight_and_enlarged(frame_img, box, self.ctx.cfg.crops.enlarge_factor,
                                                rep["frame_width"], rep["frame_height"])
        sframe = self.sampled[rep["frame_idx"]]
        temporal: list[tuple[str, Image.Image]] = []
        for off in self.ctx.cfg.sampling.temporal_offsets_for_vlm:
            key = f"{off:+d}"
            rel = sframe.get("temporal_frames", {}).get(key)
            if rel and self.ctx.path(rel).exists():
                temporal.append((key, Image.open(self.ctx.path(rel)).convert("RGB")))
        return tight, enlarged, temporal

    def content(self, det_id: int, frame_img: Image.Image, blind: bool) -> tuple[list[dict], list[dict], list[str]]:
        """``(content, hint_composition, signals_rendered)`` for one call."""
        from generate_hints import _build_vlm_content

        rep = self.reps[det_id]
        sframe = self.sampled[rep["frame_idx"]]
        tight, enlarged, temporal = self.images(det_id, frame_img)
        signals = self.blind_signals if blind else self.hinted_signals
        ocr_row = self.ocr.get(det_id)
        faiss_row = self.faiss.get(det_id, {})
        hits = [{"brand_name": b, "similarity": s} for b, s in
                zip(faiss_row.get("faiss_top5_brands", []), faiss_row.get("faiss_top5_scores", []))]
        audio = self.audio_brands if not blind and "audio" in self.text_signals else []
        hits = hits if not blind and "faiss" in self.text_signals else []
        ocr_data = ({"source": ocr_row.get("ocr_source"), "lines": ocr_row.get("ocr_lines", [])}
                    if ocr_row and not blind and "ocr" in self.text_signals else None)
        watermark = "" if blind else self.channel_watermark
        content = _build_vlm_content(
            crop_img=tight if "tight_crop" in signals else None,
            enlarged_img=enlarged if "enlarged_crop" in signals else None,
            frame_img=frame_img if "full_frame" in signals else None,
            temporal_pairs=temporal if "temporal_context" in signals else [],
            detection={"score": rep["det_score"], "timecode_seconds": sframe["timecode_s"]},
            audio_brands=audio, faiss_hits=hits, ocr_data=ocr_data, signals=set(signals),
            brand_mode=self.ctx.cfg.vlm.brand_mode, channel_watermark=watermark, blind=blind,
        )
        composition = [] if blind else _hint_composition(audio, hits, ocr_row if ocr_data else None, watermark)
        return content, composition, sorted(signals)


def _complete(ctx: RunContext, rel: str, det_ids) -> bool:
    if not ctx.path(rel).exists():
        return False
    return set(det_ids) <= {r["det_id"] for r in ctx.read_jsonl(rel)}


class VlmStage(Stage):
    name = "vlm"
    device = "gpu"
    inputs = [MERGED_DETECTIONS_FILE, REFINEMENT_FILE, OCR_FILE, FAISS_HINTS_FILE, AUDIO_HINTS_FILE]
    outputs = [VLM_HINTED_FILE]

    def outputs_exist(self, ctx: RunContext) -> bool:
        vcfg = ctx.cfg.vlm
        det_ids = [r["det_id"] for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)]
        if not ctx.path(VLM_HINTED_FILE).exists():
            return False
        if vcfg.hinted_pass and not _complete(ctx, VLM_HINTED_FILE, det_ids):
            return False
        if vcfg.blind_pass and not _complete(ctx, VLM_BLIND_FILE, det_ids):
            return False
        return True

    def run(self, ctx: RunContext) -> StageResult:
        vcfg = ctx.cfg.vlm
        which = ctx.options.get("vlm_pass", "both")          # "both" | "blind" (--blind-only)
        if which == "blind" and not vcfg.blind_pass:
            raise ValueError("--blind-only needs vlm.blind_pass: true in the resolved config")
        do_hinted = vcfg.hinted_pass and which != "blind"
        do_blind = vcfg.blind_pass

        det_ids = sorted(r["det_id"] for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE))
        hinted = JsonlAppender(ctx.path(VLM_HINTED_FILE), ctx.header(
            "vlm", model_id=vcfg.model_id, quantisation=vcfg.quantisation, brand_mode=vcfg.brand_mode,
            pass_type="hinted"), truncate=ctx.force and do_hinted)
        comp = JsonlAppender(ctx.path(VLM_HINT_COMPOSITION_FILE), ctx.header("vlm", pass_type="hint_composition"),
                             truncate=ctx.force and do_hinted)
        blind = (JsonlAppender(ctx.path(VLM_BLIND_FILE), ctx.header(
            "vlm", model_id=vcfg.model_id, quantisation=vcfg.quantisation, brand_mode=vcfg.brand_mode,
            pass_type="blind", blind_pass_signals=list(vcfg.blind_pass_signals)), truncate=ctx.force)
                 if do_blind else None)
        try:
            todo_h = [d for d in det_ids if d not in hinted.done] if do_hinted else []
            todo_b = [d for d in det_ids if d not in blind.done] if blind is not None else []
            todo = sorted(set(todo_h) | set(todo_b))
            if not vcfg.hinted_pass:
                ctx.log.info("vlm: vlm.hinted_pass is false, no hinted calls")
            if not todo:
                return StageResult(counts={"calls": 0, "blind_calls": 0, "skipped": len(det_ids)})

            inp = DetectionInputs(ctx)
            model, processor = _get_model(vcfg.model_id, vcfg.quantisation, vcfg.max_pixels)
            n_h = n_b = 0
            set_h, set_b = set(todo_h), set(todo_b)
            for i, det_id in enumerate(todo, 1):
                frame_img = ctx.open_frame_image(inp.reps[det_id]["frame_idx"])
                if det_id in set_h:
                    content, composition, rendered = inp.content(det_id, frame_img, blind=False)
                    t0 = time.perf_counter()
                    raw = generate(model, processor, content, vcfg)
                    parsed = parse_reply(raw)
                    hinted.write({"det_id": det_id, "raw": raw, "parsed": parsed, "vlm_parse_ok": parsed is not None,
                                  "enabled_signals": rendered, "elapsed_s": round(time.perf_counter() - t0, 3)})
                    if vcfg.log_hint_composition:
                        comp.write({"det_id": det_id, "hint_composition": composition, "signals_rendered": rendered,
                                    "mode": vcfg.brand_mode})
                    n_h += 1
                if det_id in set_b:
                    content, _, rendered = inp.content(det_id, frame_img, blind=True)
                    t0 = time.perf_counter()
                    raw = generate(model, processor, content, vcfg)
                    parsed = parse_reply(raw)
                    blind.write({"det_id": det_id, "raw": raw, "parsed": parsed, "vlm_parse_ok": parsed is not None,
                                 "enabled_signals": rendered, "elapsed_s": round(time.perf_counter() - t0, 3)})
                    n_b += 1
                ctx.log.progress(i, len(todo), unit="det")
        finally:
            hinted.close()
            comp.close()
            if blind is not None:
                blind.close()
        return StageResult(counts={"calls": n_h, "blind_calls": n_b})

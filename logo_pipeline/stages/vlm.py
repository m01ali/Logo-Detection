"""``stages/vlm.py`` (07 MM1 Step 5.4): wraps ``generate_hints._build_vlm_content``
and the tokenise/generate/parse tail of ``generate_hints.run_vlm_on_detection``.

Image loading is reimplemented rather than wrapped: ``run_vlm_on_detection``'s
loader assumes M2's ``extraction_output/`` file layout (pre-baked
``crop.jpg`` files), but crop JPEGs are written only by ``stages/output.py``
(05 section C, "Stage: output"), which runs after ``vlm`` in the stage
order. This stage cuts its own crops in memory from ``frames/`` (03 §3).
Everything else — the signal-section text, the closed-set candidate list,
the final instruction (including the T67 UNKNOWN wording, already applied to
``generate_hints.build_final_instruction``), the output JSON schema,
tokenisation, generation and parsing — is the same code M2 uses.

Sections are included only when ``compute_and_log and include_in_vlm_prompt``
(image signals: ``include_in_vlm_prompt`` alone, 03 §9); computation of the
text signals happened upstream in their own stages regardless of the prompt
switch. Hinted pass only (07 Goal: "the blind pass are stubs that pass rows
through unchanged"); blind-pass fields stay null until MM9.
"""

from __future__ import annotations

import json as _json
import re

from PIL import Image

from ..config import text_signal_in_prompt
from ..crops import tight_and_enlarged
from .base import (AUDIO_HINTS_FILE, FAISS_HINTS_FILE, MERGED_DETECTIONS_FILE, OCR_FILE, REFINEMENT_FILE,
                    VLM_HINTED_FILE, VLM_HINT_COMPOSITION_FILE, RunContext, Stage, StageResult)

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
    """T34 hint-composition entries (07 C12: the shape MM9 will extend, not redo)."""
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


class VlmStage(Stage):
    name = "vlm"
    device = "gpu"
    inputs = [MERGED_DETECTIONS_FILE, REFINEMENT_FILE, OCR_FILE, FAISS_HINTS_FILE, AUDIO_HINTS_FILE]
    outputs = [VLM_HINTED_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        vcfg = ctx.cfg.vlm
        video = ctx.video
        reps = {r["det_id"]: r for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        refined = {r["det_id"]: r for r in ctx.read_jsonl(REFINEMENT_FILE)}
        ocr_by_det = {r["det_id"]: r for r in ctx.read_jsonl(OCR_FILE)} if ctx.path(OCR_FILE).exists() else {}
        faiss_by_det = {r["det_id"]: r for r in ctx.read_jsonl(FAISS_HINTS_FILE)} if ctx.path(FAISS_HINTS_FILE).exists() else {}
        audio_hints = ctx.read_json(AUDIO_HINTS_FILE) if ctx.path(AUDIO_HINTS_FILE).exists() else {"brands": []}
        audio_brands = audio_hints.get("brands", [])
        sampled_by_frame = {f["frame_idx"]: f for f in ctx.sampled_frames()}

        if not vcfg.hinted_pass:
            ctx.write_jsonl(VLM_HINTED_FILE, [], header=ctx.header("vlm", hinted_pass=False))
            ctx.log.stage_skipped(self.name, "vlm.hinted_pass is false")
            return StageResult(counts={"calls": 0})

        done = {r["det_id"] for r in ctx.read_jsonl(VLM_HINTED_FILE)} if not ctx.force else set()
        todo = [d for d in reps if d not in done]
        if not todo:
            ctx.log.stage_skipped(self.name, "all detections already have a hinted VLM call")
            return StageResult(counts={"calls": 0, "skipped": len(reps)})
        rows = [r for r in ctx.read_jsonl(VLM_HINTED_FILE) if r["det_id"] not in todo]
        comp_rows = ([r for r in ctx.read_jsonl(VLM_HINT_COMPOSITION_FILE) if r["det_id"] not in todo]
                     if ctx.path(VLM_HINT_COMPOSITION_FILE).exists() else [])

        from qwen_vl_utils import process_vision_info

        import torch
        from generate_hints import _build_vlm_content, normalize_signals_used

        model, processor = _get_model(vcfg.model_id, vcfg.quantisation, vcfg.max_pixels)

        image_signals = {n for n in ("tight_crop", "enlarged_crop", "full_frame", "temporal_context")
                          if getattr(ctx.cfg.signals, n).include_in_vlm_prompt}
        text_signal_names = {n for n in ("audio", "faiss", "ocr") if text_signal_in_prompt(ctx.cfg, n)}
        text_name_map = {"audio": "audio_brands", "faiss": "faiss_matches", "ocr": "ocr_text"}
        rendered = image_signals | {text_name_map[n] for n in text_signal_names}
        channel_watermark = (video.channel_brand or "") if vcfg.channel_watermark_line else ""

        for i, det_id in enumerate(sorted(todo), 1):
            rep = reps[det_id]
            frame_idx = rep["frame_idx"]
            sframe = sampled_by_frame[frame_idx]
            box = refined.get(det_id, {}).get("selected_bbox") or rep["loose_bbox"]
            frame_img = ctx.open_frame_image(frame_idx)
            tight, enlarged, _ = tight_and_enlarged(frame_img, box, ctx.cfg.crops.enlarge_factor,
                                                     rep["frame_width"], rep["frame_height"])

            temporal_pairs: list[tuple[str, Image.Image]] = []
            if "temporal_context" in image_signals:
                for off in ctx.cfg.sampling.temporal_offsets_for_vlm:
                    key = f"{off:+d}"
                    rel = sframe.get("temporal_frames", {}).get(key)
                    if rel and ctx.path(rel).exists():
                        temporal_pairs.append((key, Image.open(ctx.path(rel)).convert("RGB")))

            ocr_row = ocr_by_det.get(det_id)
            ocr_data = {"source": ocr_row.get("ocr_source"), "lines": ocr_row.get("ocr_lines", [])} if ocr_row else None
            faiss_row = faiss_by_det.get(det_id, {})
            faiss_hits = [{"brand_name": b, "similarity": s} for b, s in
                          zip(faiss_row.get("faiss_top5_brands", []), faiss_row.get("faiss_top5_scores", []))]
            row_audio_brands = audio_brands if "audio" in text_signal_names else []
            row_faiss_hits = faiss_hits if "faiss" in text_signal_names else []
            row_ocr_data = ocr_data if "ocr" in text_signal_names else None

            detection_stub = {"score": rep["det_score"], "timecode_seconds": sframe["timecode_s"]}
            content = _build_vlm_content(
                crop_img=tight if "tight_crop" in image_signals else None,
                enlarged_img=enlarged if "enlarged_crop" in image_signals else None,
                frame_img=frame_img if "full_frame" in image_signals else None,
                temporal_pairs=temporal_pairs,
                detection=detection_stub,
                audio_brands=row_audio_brands,
                faiss_hits=row_faiss_hits,
                ocr_data=row_ocr_data,
                signals=rendered,
                brand_mode=vcfg.brand_mode,
                channel_watermark=channel_watermark,
            )
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt").to(model.device, model.dtype)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=vcfg.max_new_tokens,
                                                do_sample=(vcfg.decoding != "greedy"))
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            parsed = None
            try:
                m = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if m:
                    parsed = _json.loads(m.group())
            except _json.JSONDecodeError:
                parsed = None
            if parsed is not None and "signals_used" in parsed:
                parsed["signals_used_raw"] = parsed.get("signals_used")
                parsed["signals_used"] = normalize_signals_used(parsed.get("signals_used"))

            rows.append({"det_id": det_id, "raw": raw_text, "parsed": parsed, "vlm_parse_ok": parsed is not None,
                         "enabled_signals": sorted(rendered)})
            comp_rows.append({
                "det_id": det_id,
                "hint_composition": _hint_composition(row_audio_brands, row_faiss_hits, row_ocr_data if ocr_row else None,
                                                       channel_watermark),
                "signals_rendered": sorted(rendered), "mode": vcfg.brand_mode,
            })
            ctx.log.progress(i, len(todo), unit="det")

        header = ctx.header("vlm", model_id=vcfg.model_id, quantisation=vcfg.quantisation,
                             brand_mode=vcfg.brand_mode, pass_type="hinted")
        ctx.write_jsonl(VLM_HINTED_FILE, sorted(rows, key=lambda r: r["det_id"]), header=header)
        if vcfg.log_hint_composition:
            ctx.write_jsonl(VLM_HINT_COMPOSITION_FILE, sorted(comp_rows, key=lambda r: r["det_id"]), header=header)
        return StageResult(counts={"calls": len(todo)})

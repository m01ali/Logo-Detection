"""``stages/audio.py`` (07 MM1 Step 5.1, T10): Whisper + text-LLM brand
extraction, orchestrated directly over ``models.audio_model.WhisperModel``
and ``models.description_model.Qwen2_5TextModel`` rather than calling
``generate_hints.compute_audio_hints`` (which has three M2 defects this
stage exists to fix, 01 §6): it writes into ``<run_dir>/audio/`` (not the
cwd), forces greedy decoding (``do_sample=False``, determinism), and reads
the prompt variant from ``audio.brand_prompt`` (04) instead of always using
the M2 prompt.

Video-level: one ``audio/audio_hints.json``, not one line per detection.
``audio_brands`` is broadcast onto every row by ``router_input.py`` (M2
convention, 05 F).
"""

from __future__ import annotations

from ..prompts import brand_prompt_pair
from .base import AUDIO_HINTS_FILE, AUDIO_WAV_FILE, RunContext, Stage, StageResult

_LANG_CODE = {"italian": "it", "english": "en"}


class AudioStage(Stage):
    name = "audio"
    device = "gpu"
    inputs: list[str] = []
    outputs = [AUDIO_HINTS_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        acfg = ctx.cfg.audio
        sw = ctx.cfg.signals.audio
        manifest = ctx.manifest

        if not sw.compute_and_log:
            ctx.write_json(AUDIO_HINTS_FILE, {"status": "disabled", "brands": [], "transcript": "", "language": None})
            ctx.log.stage_skipped(self.name, "signals.audio.compute_and_log is false")
            return StageResult(counts={"status": "disabled"})

        if manifest["mode"] == "frames":
            ctx.write_json(AUDIO_HINTS_FILE, {"status": "signal_unavailable", "brands": [],
                                               "transcript": "", "language": None})
            ctx.log.stage_skipped(self.name, "frames mode has no audio track (02 §3.1)")
            return StageResult(counts={"status": "signal_unavailable"})

        if ctx.path(AUDIO_HINTS_FILE).exists() and not ctx.force:
            ctx.log.stage_skipped(self.name, "audio_hints.json already exists")
            cached = ctx.read_json(AUDIO_HINTS_FILE)
            return StageResult(counts={"status": cached.get("status", "ok"), "brands": len(cached.get("brands", []))})

        if not manifest.get("has_audio") or not acfg.enabled:
            reason = "video has no audio track" if not manifest.get("has_audio") else "audio.enabled is false"
            ctx.write_json(AUDIO_HINTS_FILE, {"status": "empty", "brands": [], "transcript": "", "language": None,
                                               "note": reason})
            ctx.log.stage_skipped(self.name, reason)
            return StageResult(counts={"status": "empty"})

        wav_path = ctx.path(AUDIO_WAV_FILE)
        from models.audio_model import WhisperModel

        ctx.log.info("audio: loading Whisper (%s) …", acfg.whisper_model_id)
        whisper = WhisperModel(model_id=acfg.whisper_model_id)
        language_hint = acfg.language if acfg.task == "transcribe" else "english"
        transcript, _chunks, detected_lang = whisper.run_example(
            str(wav_path), mode=acfg.task, language=language_hint, return_language=True)
        lang_code = detected_lang or _LANG_CODE.get((acfg.language or "").lower(), "orig")
        ctx.path(f"audio/transcript.{lang_code}.txt").write_text(transcript, encoding="utf-8")
        del whisper
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        from models.description_model import Qwen2_5TextModel

        ctx.log.info("audio: loading text model (%s) for brand extraction …", acfg.text_model_id)
        text_model = Qwen2_5TextModel(model_id=acfg.text_model_id)
        system_prompt, user_prompt = brand_prompt_pair(acfg.brand_prompt, transcript, acfg.language)
        brand_string = text_model.run_example(user_prompt, system_prompt, do_sample=(acfg.text_decoding != "greedy"))
        brands = sorted({b.lower().strip() for b in brand_string.split("\n") if b.strip()})
        del text_model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        hints = {
            "status": "ok" if brands else "empty",
            "transcript": transcript, "language": lang_code, "task": acfg.task,
            "brand_prompt_variant": acfg.brand_prompt, "text_decoding": acfg.text_decoding,
            "brands": brands,
        }
        ctx.write_json(AUDIO_HINTS_FILE, hints)
        ctx.log.info("audio: extracted %d brands: %s", len(brands), brands)
        return StageResult(counts={"status": hints["status"], "brands": len(brands)})

"""21_test_plan.md — MM1 groundwork: stages/audio.py's branches (disabled,
frames mode, no audio track, cached, and the real compute path with
WhisperModel/Qwen2_5TextModel mocked out)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.stages.audio import AudioStage


def test_audio_disabled_by_signal_switch(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, overrides={"signals": {"audio": {"compute_and_log": False,
                                                                          "include_in_vlm_prompt": False}}})
    result = AudioStage().run(ctx)
    assert result.counts["status"] == "disabled"
    hints = ctx.read_json("audio/audio_hints.json")
    assert hints["brands"] == []


def test_audio_unavailable_in_frames_mode(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    manifest = ctx.manifest
    manifest["mode"] = "frames"
    ctx.write_json("run_manifest.json", manifest)
    ctx.invalidate()
    result = AudioStage().run(ctx)
    assert result.counts["status"] == "signal_unavailable"


def test_audio_empty_when_no_audio_track(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)   # helpers.make_ctx sets has_audio: False
    result = AudioStage().run(ctx)
    assert result.counts["status"] == "empty"
    hints = ctx.read_json("audio/audio_hints.json")
    assert hints["brands"] == []


def test_audio_uses_cache_without_reloading_models(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root)
    ctx.write_json("audio/audio_hints.json", {"status": "ok", "brands": ["rai sport"], "transcript": "x",
                                               "language": "it"})
    with patch("models.audio_model.WhisperModel") as w:
        result = AudioStage().run(ctx)
    w.assert_not_called()
    assert result.counts["status"] == "ok"
    assert result.counts["brands"] == 1


def test_audio_real_compute_path_greedy_and_prompt_variant(tmp_path, repo_root):
    manifest_overrides = {"has_audio": True}
    ctx = make_ctx(tmp_path, repo_root, overrides={"audio": {"brand_prompt": "task2_italian",
                                                              "text_decoding": "greedy"}})
    manifest = ctx.manifest
    manifest.update(manifest_overrides)
    ctx.write_json("run_manifest.json", manifest)
    ctx.invalidate()
    (ctx.run_dir / "audio").mkdir(exist_ok=True)
    (ctx.run_dir / "audio" / "audio.wav").write_bytes(b"fake")

    whisper_instance = MagicMock()
    whisper_instance.run_example.return_value = ("ciao rai sport", None, "it")
    text_instance = MagicMock()
    text_instance.run_example.return_value = "rai sport\nverisure\n"

    with patch("models.audio_model.WhisperModel", return_value=whisper_instance) as WM, \
         patch("models.description_model.Qwen2_5TextModel", return_value=text_instance) as QM:
        result = AudioStage().run(ctx)

    assert WM.called and QM.called
    # do_sample=False was requested for the audio stage's determinism (07 MM1 Step 5.1 / T10).
    call_kwargs = text_instance.run_example.call_args.kwargs
    assert call_kwargs.get("do_sample") is False

    assert result.counts["status"] == "ok"
    assert result.counts["brands"] == 2
    hints = ctx.read_json("audio/audio_hints.json")
    assert hints["brands"] == ["rai sport", "verisure"]
    assert hints["brand_prompt_variant"] == "task2_italian"
    assert (ctx.run_dir / "audio" / "transcript.it.txt").read_text(encoding="utf-8") == "ciao rai sport"

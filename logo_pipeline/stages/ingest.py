"""``stages/ingest.py`` (07 MM1 Step 2): opens the source, writes
``run_manifest.json`` (03 §4), extracts audio in video mode.

Audio extraction replicates ``models.audio_model.convert_video_to_audio``
but writes into ``<run_dir>/audio/audio.wav`` instead of the cwd (01 finding
#6). The function is not imported for this reason.
"""

from __future__ import annotations

import subprocess

from ..frame_source import FolderFrameSource, VideoFrameSource, open_frame_source
from .base import AUDIO_WAV_FILE, RUN_MANIFEST, RunContext, Stage, StageResult, utcnow_iso


def _git_sha(repo_root) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True,
                              text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _pip_freeze() -> list[str]:
    try:
        import importlib.metadata as md
        return sorted(f"{d.name}=={d.version}" for d in md.distributions() if d.name)
    except Exception:
        return []


def extract_audio(video_path, out_wav_path) -> bool:
    """Write ``out_wav_path`` from ``video_path``'s audio track via moviepy.

    Returns ``False`` (and writes nothing) when the source has no audio
    track, which some trimmed clips do not.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:  # moviepy 1.x
        from moviepy.editor import VideoFileClip

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            return False
        clip.audio.write_audiofile(str(out_wav_path), logger=None)
    return True


class IngestStage(Stage):
    name = "ingest"
    device = "cpu"
    inputs: list[str] = []
    outputs = [RUN_MANIFEST]

    def run(self, ctx: RunContext) -> StageResult:
        cfg, video = ctx.cfg, ctx.video
        source = ctx.open_source()
        if source is None:
            # Video/frames not reachable from this machine (e.g. GPU-produced
            # run copied to the Mac): the manifest from run creation stands.
            if ctx.path(RUN_MANIFEST).exists():
                ctx.log.stage_skipped(self.name, "source unreachable, manifest already exists")
                return StageResult(counts={"reused": True})
            raise FileNotFoundError(
                f"Cannot open source for video_id={video.video_id!r} "
                f"(mode={cfg.ingest.mode}) and no run_manifest.json exists yet.")

        has_audio = False
        if cfg.ingest.mode == "video" and isinstance(source, VideoFrameSource):
            try:
                has_audio = extract_audio(source.source_path, ctx.path(AUDIO_WAV_FILE))
            except Exception as exc:  # pragma: no cover - moviepy/ffmpeg env issues
                ctx.log.warning("ingest: audio extraction failed (%s); continuing without audio", exc)
                has_audio = False

        manifest = {
            "video_id": video.video_id,
            "run_id": ctx.run_dir.name,
            "config_hash": ctx.config_hash,
            "config": ctx.resolved.resolved,
            "effective_config": ctx.resolved.effective,
            "brand_list_version": ctx.brands.version,
            "brand_list_hash": ctx.brands.content_hash,
            "git_sha": _git_sha(ctx.repo_root),
            "pip_freeze": _pip_freeze(),
            "created_at": utcnow_iso(),
            "stage_list": list(cfg.run.stages),
            "signal_switches": __import__("logo_pipeline.config", fromlist=["signal_switches"]).signal_switches(cfg),
            **source.describe(),
            "has_audio": has_audio,
            "channel": video.channel,
            "channel_brand": video.channel_brand,
        }
        ctx.write_json(RUN_MANIFEST, manifest)
        ctx.invalidate()
        source.close()
        return StageResult(counts={
            "frame_count": manifest["frame_count"], "fps": manifest["fps"],
            "fps_is_assumed": manifest["fps_is_assumed"], "has_audio": has_audio,
        })

"""Shared test helper: a minimal real ``RunContext`` over a synthetic
sampled-frame fixture, for stage unit tests that don't need real media or
heavy ML deps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from logo_pipeline.brands import load_brand_list
from logo_pipeline.config import resolve_config
from logo_pipeline.logging_utils import StageLogger
from logo_pipeline.stages.base import RunContext


def make_ctx(tmp_path: Path, repo_root: Path, *, n_frames: int = 2, frame_size=(200, 100),
             overrides: dict | None = None) -> RunContext:
    """A RunContext with a fake video_manifest + sampled_frames.json + saved
    frame JPEGs, ready for a stage that reads from ``frames/``."""
    video = {"video_id": "synthetic", "source": "synthetic.mp4",
             "overrides": overrides or {}}
    resolved = resolve_config(str(repo_root / "configs" / "pipeline.yaml"), video)
    brands = load_brand_list(repo_root / "configs" / "brands" / "roboflow_2026-04-28.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    log = StageLogger(run_dir, log_every_n=1000)
    ctx = RunContext(run_dir, resolved, brands, log, repo_root=repo_root)

    w, h = frame_size
    sampled = []
    for i in range(n_frames):
        frame_idx = i * 10
        arr = np.full((h, w, 3), 30 * (i + 1) % 255, dtype=np.uint8)
        (run_dir / "frames").mkdir(exist_ok=True)
        Image.fromarray(arr).save(run_dir / "frames" / f"frame_{frame_idx:06d}.jpg")
        sampled.append({"frame_idx": frame_idx, "timecode_s": float(frame_idx), "shot_id": 0,
                        "path": f"frames/frame_{frame_idx:06d}.jpg", "temporal_frames": {}})
    ctx.write_json("sampled_frames.json", sampled)
    ctx.write_json("run_manifest.json", {
        "video_id": "synthetic", "run_id": run_dir.name, "config_hash": resolved.config_hash,
        "mode": "video", "fps": 10.0, "fps_is_assumed": False, "frame_count": n_frames * 10 + 1,
        "frame_width": w, "frame_height": h, "has_audio": False, "channel": None, "channel_brand": None,
        "channel_brand_canonical": None,
    })
    ctx.invalidate()
    return ctx

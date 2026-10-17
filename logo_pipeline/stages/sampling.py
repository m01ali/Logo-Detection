"""``stages/sampling.py`` (07 MM1 Step 2): ``interval | per_shot | segment``
-> ``sampled_frames.json``; saves ``frames/`` and ``temporal/``.

``interval`` reproduces M2's ``_sample_indices`` (``step = round(fps *
interval_s)``) in video mode; in frames mode the same step is computed from
``assumed_fps`` and applied to *positions* in the sorted ordinal list, not to
raw frame numbers (03 §5.2). ``segment`` is the M2 legacy mode. ``per_shot``
takes the middle or first frame of each scene in ``shots.json``.
"""

from __future__ import annotations

import bisect

import cv2
from PIL import Image

from .base import FRAMES_DIR, SAMPLED_FRAMES_FILE, SHOTS_FILE, TEMPORAL_DIR, RunContext, Stage, StageResult, frame_file_name


def _interval_video_indices(total: int, fps: float, interval_s: float) -> list[int]:
    step = max(1, round(fps * interval_s))
    return list(range(0, total, step)) if total > 0 else []


def _interval_frames_positions(n: int, assumed_fps: float, interval_s: float) -> list[int]:
    step = max(1, round(assumed_fps * interval_s))
    return list(range(0, n, step)) if n > 0 else []


def _segment_indices(total: int, fps: float, duration_s: float) -> list[int]:
    seg_len = max(1, round(fps * duration_s))
    indices = []
    for seg_start in range(0, total, seg_len):
        seg_end = min(seg_start + seg_len - 1, total - 1)
        indices.append((seg_start + seg_end) // 2)
    return sorted(set(indices))


def _shot_id_for(frame_idx: int, scenes: list[dict]) -> int | None:
    for s in scenes:
        if s["start_frame"] <= frame_idx <= s["end_frame"]:
            return s["shot_id"]
    return None


def _save_jpeg(bgr, path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil.save(path, quality=quality)


class SamplingStage(Stage):
    name = "sampling"
    device = "cpu"
    inputs = [SHOTS_FILE]
    outputs = [SAMPLED_FRAMES_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.cfg
        manifest = ctx.manifest
        scenes = ctx.read_json(SHOTS_FILE) if ctx.path(SHOTS_FILE).exists() else []
        source = ctx.open_source()
        if source is None:
            raise FileNotFoundError("sampling needs a reachable frame source")

        mode = cfg.sampling.mode
        is_frames = manifest["mode"] == "frames"
        window = cfg.sampling.temporal_window
        quality = cfg.ingest.save_frame_jpeg_quality

        if is_frames:
            ordinals = source.indices()  # sorted
            if ctx.video.frame_ordinals:          # route three: exactly the labelled ordinals (17 T66)
                wanted = set(ctx.video.frame_ordinals)
                positions = [i for i, o in enumerate(ordinals) if o in wanted]
            elif ctx.video.process_all_frames:
                positions = list(range(len(ordinals)))
            elif mode == "interval":
                positions = _interval_frames_positions(len(ordinals), source.fps, cfg.sampling.interval_s)
            elif mode == "segment":
                # position space stands in for M2's frame-index space (03 §5.2 general rule)
                positions = [p for p in _segment_indices(len(ordinals), source.fps, cfg.sampling.segment.duration_s)
                             if p < len(ordinals)]
            elif mode == "per_shot":
                positions = []
                for s in scenes:
                    lo = bisect.bisect_left(ordinals, s["start_frame"])
                    hi = bisect.bisect_right(ordinals, s["end_frame"]) - 1
                    if lo > hi:
                        continue
                    positions.append(lo if cfg.sampling.per_shot.position == "first" else (lo + hi) // 2)
            else:
                raise ValueError(f"unknown sampling.mode {mode!r}")
            frame_indices = [ordinals[p] for p in sorted(set(positions))]
        else:
            total = manifest["frame_count"]
            fps = manifest["fps"]
            if mode == "interval":
                frame_indices = _interval_video_indices(total, fps, cfg.sampling.interval_s)
            elif mode == "segment":
                frame_indices = _segment_indices(total, fps, cfg.sampling.segment.duration_s)
            elif mode == "per_shot":
                frame_indices = []
                for s in scenes:
                    lo, hi = s["start_frame"], min(s["end_frame"], total - 1)
                    frame_indices.append(lo if cfg.sampling.per_shot.position == "first" else (lo + hi) // 2)
                frame_indices = sorted(set(frame_indices))
            else:
                raise ValueError(f"unknown sampling.mode {mode!r}")

        sampled: list[dict] = []
        for frame_idx in frame_indices:
            bgr = source.read(frame_idx)
            if bgr is None:
                ctx.log.warning("sampling: could not read frame %d — skipping", frame_idx)
                continue
            rel_path = f"{FRAMES_DIR}/{frame_file_name(frame_idx)}"
            _save_jpeg(bgr, ctx.path(rel_path), quality)

            offsets = range(-window, window + 1)
            temporal_paths: dict[str, str | None] = {}
            for offset in offsets:
                if offset == 0:
                    continue
                key = f"{offset:+d}"
                if is_frames:
                    neighbour = source.neighbour(frame_idx, offset)
                    t_bgr = source.read(neighbour) if neighbour is not None else None
                    t_idx = neighbour
                else:
                    t_idx = frame_idx + offset
                    t_bgr = source.read(t_idx) if 0 <= t_idx < manifest["frame_count"] else None
                if t_bgr is None:
                    temporal_paths[key] = None
                    continue
                t_rel = f"{TEMPORAL_DIR}/frame_{frame_idx:06d}/T{offset:+d}_frame{t_idx:06d}.jpg"
                _save_jpeg(t_bgr, ctx.path(t_rel), quality)
                temporal_paths[key] = t_rel

            sampled.append({
                "frame_idx": frame_idx,
                "timecode_s": round(source.timecode(frame_idx), 3),
                "shot_id": _shot_id_for(frame_idx, scenes),
                "path": rel_path,
                "temporal_frames": temporal_paths,
            })

        source.close()
        ctx.write_json(SAMPLED_FRAMES_FILE, sampled)
        ctx.invalidate()
        return StageResult(counts={"sampled_frames": len(sampled), "mode": mode})

"""``stages/output.py`` (T56, T58, T59, ``spec docs/13``): cuts and saves the
crop / enlarged-crop / loose-crop JPEGs (the only stage allowed to, 05 section
C), draws an annotated image for every sampled frame (``annotate.draw_frame``),
writes ``rows.csv`` in the frozen column order (05) and ``annotated.mp4``.

``rows.csv`` keeps DEDUPED rows with their decision so the review screen can
show them and T62 can count them. Image paths are written relative to the
CSV's own directory with forward slashes (``output.csv_paths: relative``,
what the review UI's ``_resolve_path`` expects) or absolute.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PIL import Image

from ..annotate import draw_frame, frame_holds, write_video
from ..crops import cut_crop, tight_and_enlarged
from ..schema import CSV_COLUMNS, Row, row_to_csv_dict
from .base import (ANNOTATED_FRAMES_DIR, ANNOTATED_VIDEO_FILE, CROPS_DIR, FRAMES_DIR, ROUTER_INPUT_FILE,
                    ROUTER_OUTPUT_FILE, SHOTS_FILE, WATERMARK_FILE, RunContext, Stage, StageResult, det_dir_name,
                    frame_file_name)

_ROW_FIELDS = set(Row.__dataclass_fields__)
_PATH_FIELDS = ("crop_path", "enlarged_crop_path", "loose_crop_path", "frame_path", "annotated_frame_path")


def _save_jpeg(img, path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality)


def csv_path_for(ctx: RunContext, rel: str | None, csv_dir: Path, mode: str) -> str | None:
    """A run-relative path re-expressed for the CSV (05 C)."""
    if not rel:
        return rel
    target = ctx.path(rel).resolve()
    if mode == "absolute":
        return target.as_posix()
    return Path(os.path.relpath(target, csv_dir.resolve())).as_posix()


def _regions(ctx: RunContext) -> list:
    if not ctx.path(WATERMARK_FILE).exists():
        return []
    header, _ = ctx.read_jsonl(WATERMARK_FILE, with_header=True)
    return list((header or {}).get("regions") or [])


def video_step_s(ctx: RunContext) -> float | None:
    """The source time between two sampled frames in frames mode (``None`` in video mode)."""
    if ctx.manifest.get("mode") != "frames":
        return None
    fps = float(ctx.manifest.get("fps") or ctx.cfg.ingest.frames.assumed_fps)
    if ctx.cfg.sampling.mode == "interval":
        if ctx.video.frame_ordinals:
            return 1.0 / fps          # sparse labelled ordinals: hold each for one source frame
        k = 1 if ctx.video.process_all_frames else max(1, round(fps * ctx.cfg.sampling.interval_s))
        return k / fps
    return 1.0 / fps


class OutputStage(Stage):
    name = "output"
    device = "cpu"
    inputs = [ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE]
    outputs: list[str] = []   # csv_name is configurable; see outputs_exist

    def outputs_exist(self, ctx: RunContext) -> bool:
        ocfg = ctx.cfg.output
        if not ctx.path(ocfg.csv_name).exists():
            return False
        return not (ocfg.annotated_video and ocfg.annotated_frames) or ctx.path(ANNOTATED_VIDEO_FILE).exists()

    def run(self, ctx: RunContext) -> StageResult:
        ocfg = ctx.cfg.output
        router_input = {r["det_id"]: r for r in ctx.read_jsonl(ROUTER_INPUT_FILE)}
        router_output = {r["det_id"]: r for r in ctx.read_jsonl(ROUTER_OUTPUT_FILE)}
        sampled = ctx.sampled_frames()

        rows: list[Row] = []
        by_frame: dict[int, list[Row]] = {}
        frame_cache: dict[int, Image.Image] = {}

        for det_id in sorted(router_input):
            merged = dict(router_input[det_id])
            merged.update({k: v for k, v in router_output.get(det_id, {}).items() if k != "det_id"})
            row = Row(**{k: v for k, v in merged.items() if k in _ROW_FIELDS})

            frame_idx = row.frame_idx
            if frame_idx not in frame_cache:
                frame_cache = {frame_idx: ctx.open_frame_image(frame_idx)}   # rows are in frame order
            frame_img = frame_cache[frame_idx]

            tight, enlarged, _ = tight_and_enlarged(frame_img, row.selected_bbox, ctx.cfg.crops.enlarge_factor,
                                                     row.frame_width, row.frame_height)
            det_dir = f"{CROPS_DIR}/{det_dir_name(det_id)}"
            if tight is not None:
                _save_jpeg(tight, ctx.path(f"{det_dir}/crop.jpg"), ctx.cfg.crops.jpeg_quality)
                row.crop_path = f"{det_dir}/crop.jpg"
            if enlarged is not None:
                _save_jpeg(enlarged, ctx.path(f"{det_dir}/crop_enlarged.jpg"), ctx.cfg.crops.jpeg_quality)
                row.enlarged_crop_path = f"{det_dir}/crop_enlarged.jpg"
            if row.selected_bbox == row.loose_bbox:
                row.loose_crop_path = row.crop_path      # 05 C: "else same file"
            else:
                loose_crop = cut_crop(frame_img, row.loose_bbox)
                if loose_crop is not None:
                    _save_jpeg(loose_crop, ctx.path(f"{det_dir}/loose.jpg"), ctx.cfg.crops.jpeg_quality)
                    row.loose_crop_path = f"{det_dir}/loose.jpg"
            row.frame_path = f"{FRAMES_DIR}/{frame_file_name(frame_idx)}"

            rows.append(row)
            by_frame.setdefault(frame_idx, []).append(row)

        counts: dict = {"rows": len(rows), "csv": ocfg.csv_name, "annotated_frames": 0}
        if ocfg.annotated_frames:
            regions = _regions(ctx)
            frame_paths = []
            for f in sampled:
                dets = by_frame.get(f["frame_idx"], [])
                rel = f"{ANNOTATED_FRAMES_DIR}/{frame_file_name(f['frame_idx'])}"
                img = draw_frame(ctx.open_frame_image(f["frame_idx"]), dets, f, ctx.video.video_id, regions, ocfg.draw)
                _save_jpeg(img, ctx.path(rel), 90)
                frame_paths.append(ctx.path(rel))
                for row in dets:
                    row.annotated_frame_path = rel
            counts["annotated_frames"] = len(sampled)

            if ocfg.annotated_video and frame_paths:
                shots = ctx.read_json(SHOTS_FILE) if ctx.path(SHOTS_FILE).exists() else []
                holds = frame_holds(sampled, ctx.cfg.sampling.mode, ctx.cfg.sampling.interval_s,
                                    ctx.cfg.sampling.segment.duration_s, video_step_s(ctx), shots)
                info = write_video(frame_paths, holds, ctx.path(ANNOTATED_VIDEO_FILE), codec=ocfg.video_codec)
                counts.update({"video_s": info["duration_s"], "video_h264": info["h264"]})

        csv_path = ctx.path(ocfg.csv_name)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            for field in _PATH_FIELDS:
                setattr(row, field, csv_path_for(ctx, getattr(row, field), csv_path.parent, ocfg.csv_paths))
        df = pd.DataFrame([row_to_csv_dict(r) for r in rows], columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)
        return StageResult(counts=counts)

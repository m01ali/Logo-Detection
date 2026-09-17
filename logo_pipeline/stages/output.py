"""``stages/output.py`` (07 MM1 Step 5.6, T9/T56): cuts and saves the crop /
enlarged-crop / loose-crop JPEGs (the only stage allowed to, 05 section C:
every path field there is ``Stage: output``), draws annotated frames for
every sampled frame ("drawing brand only"; decision colours arrive in MM7,
``spec docs/13``), and writes ``rows.csv`` in the frozen column order (05).

Wraps ``generate_hints.LABEL_COLORS`` / ``_load_font`` (the M2 visual style)
rather than ``generate_hints.annotate_frames`` / ``export_predictions_csv``
themselves, because those two functions are hard-coded to M2's
``extraction_output/`` layout and M2's narrower column set; the drawing
style and the CSV's column semantics are what carries over. ``rows.csv`` is
a strict superset of every M2 ★ column (01 §2), which is the MM1 acceptance
check ("M2 columns match ... in row count and column set", 07).
"""

from __future__ import annotations

import pandas as pd
from PIL import Image, ImageDraw

from ..crops import cut_crop, tight_and_enlarged
from ..schema import CSV_COLUMNS, Row, row_to_csv_dict
from .base import (ANNOTATED_FRAMES_DIR, CROPS_DIR, FRAMES_DIR, ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE,
                    RunContext, Stage, StageResult, det_dir_name, frame_file_name)

_ROW_FIELDS = set(Row.__dataclass_fields__)


def _save_jpeg(img, path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality)


def _draw_frame(ctx: RunContext, frame_idx: int, dets: list[Row], draw_cfg) -> None:
    from generate_hints import LABEL_COLORS, _load_font

    frame_path = ctx.path(f"{FRAMES_DIR}/{frame_file_name(frame_idx)}")
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    for row in dets:
        color = LABEL_COLORS.get(row.is_logo, LABEL_COLORS[None])
        box = [float(c) for c in row.selected_bbox]
        draw.rectangle(box, outline=color, width=3)
        parts = []
        if draw_cfg.brand:
            parts.append(row.canonical_brand or row.brand
                         or ("not a logo" if row.is_logo == "not_logo" else "unparsed"))
        if draw_cfg.det_id:
            parts.append(f"id={row.det_id}")
        if row.is_truncated:
            parts.append("[cut off]")
        if row.is_logo == "partial_logo":
            parts.append("(partial)")
        label = " · ".join(str(p) for p in parts) or "?"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        x0 = max(0, min(box[0], img.width - tw - 10))
        y0 = max(0, box[1] - th - 8)
        draw.rectangle([x0, y0, x0 + tw + 8, y0 + th + 8], fill=color)
        draw.text((x0 + 4, y0 + 3), label, fill=(255, 255, 255), font=font)
    out_path = ctx.path(f"{ANNOTATED_FRAMES_DIR}/{frame_file_name(frame_idx)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


class OutputStage(Stage):
    name = "output"
    device = "cpu"
    inputs = [ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE]
    outputs: list[str] = []   # csv_name is configurable; see outputs_exist

    def outputs_exist(self, ctx: RunContext) -> bool:
        return ctx.path(ctx.cfg.output.csv_name).exists()

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
                frame_cache[frame_idx] = ctx.open_frame_image(frame_idx)
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

        if ocfg.annotated_frames:
            for f in sampled:
                dets = by_frame.get(f["frame_idx"], [])
                _draw_frame(ctx, f["frame_idx"], dets, ocfg.draw)
                for row in dets:
                    row.annotated_frame_path = f"{ANNOTATED_FRAMES_DIR}/{frame_file_name(f['frame_idx'])}"

        csv_rows = [row_to_csv_dict(r) for r in rows]
        df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
        csv_path = ctx.path(ocfg.csv_name)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        return StageResult(counts={"rows": len(rows), "csv": ocfg.csv_name,
                                   "annotated_frames": len(sampled) if ocfg.annotated_frames else 0})

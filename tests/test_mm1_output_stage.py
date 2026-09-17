"""21_test_plan.md — MM1/MM7 groundwork: stages/output.py writes rows.csv
in CSV_COLUMNS order, cuts crop files, and draws annotated frames for every
sampled frame including ones with zero detections."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402

from logo_pipeline.schema import CSV_COLUMNS
from logo_pipeline.stages.canonicalise import CanonicaliseStage
from logo_pipeline.stages.merge import MergeStage
from logo_pipeline.stages.output import OutputStage
from logo_pipeline.stages.refine import RefineStage
from logo_pipeline.stages.router.engine import RouterStage
from logo_pipeline.stages.router_input import RouterInputStage
from logo_pipeline.stages.watermark import WatermarkStage


def _run_upto_router(ctx, dets: list[dict]) -> None:
    ctx.write_jsonl("raw_detections.jsonl", dets, header=ctx.header("detect_gdino"))
    MergeStage().run(ctx)
    RefineStage().run(ctx)
    CanonicaliseStage().run(ctx)
    WatermarkStage().run(ctx)
    ctx.write_jsonl("ocr.jsonl", [], header=ctx.header("ocr"))
    ctx.write_jsonl("faiss_hints.jsonl", [], header=ctx.header("faiss_query"))
    ctx.write_json("audio/audio_hints.json", {"status": "disabled", "brands": []})
    RouterInputStage().run(ctx)
    RouterStage().run(ctx)


def test_output_stage_writes_rows_csv_and_crops(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2, frame_size=(200, 100))
    dets = [
        {"raw_det_id": 0, "frame_idx": 0, "source": "gdino", "box_xyxy": [10, 10, 50, 50],
         "score": 0.9, "label": "a logo", "class_conf": None, "frame_width": 200, "frame_height": 100},
        # frame_idx 10 gets zero detections — must still get an annotated frame (T58).
    ]
    _run_upto_router(ctx, dets)

    result = OutputStage().run(ctx)
    assert result.counts["rows"] == 1

    csv_path = ctx.path("rows.csv")
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "crop_path"] == "crops/det_000000/crop.jpg"
    assert (ctx.run_dir / "crops" / "det_000000" / "crop.jpg").exists()
    assert (ctx.run_dir / "crops" / "det_000000" / "crop_enlarged.jpg").exists()
    # selected_bbox == loose_bbox in MM1 (refine is a stub) -> same file, per 05 C.
    assert df.loc[0, "loose_crop_path"] == df.loc[0, "crop_path"]

    # Annotated frames exist for every sampled frame, including the empty one.
    assert (ctx.run_dir / "annotated_frames" / "frame_000000.jpg").exists()
    assert (ctx.run_dir / "annotated_frames" / "frame_000010.jpg").exists()


def test_output_stage_handles_zero_detections_entirely(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=2)
    _run_upto_router(ctx, [])
    result = OutputStage().run(ctx)
    assert result.counts["rows"] == 0
    df = pd.read_csv(ctx.path("rows.csv"))
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 0
    assert (ctx.run_dir / "annotated_frames" / "frame_000000.jpg").exists()

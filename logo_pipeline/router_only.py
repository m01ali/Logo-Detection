"""T61 (``spec docs/17``): the router-only runner.

``python -m logo_pipeline router-only --config <yaml> --run <run_dir> [--input router_input.jsonl]
[--out reports/router_<tag>/]`` loads a run's ``router_input.jsonl``, rebuilds
``RouterInput`` under the given YAML, runs the MM5 engine and dedup, and writes
``router_output.jsonl``, ``firing_log.jsonl`` and a ``rows.csv`` copy carrying
the new decisions. Nothing upstream is touched, so a YAML diff plus these
files is enough to show what a rule change did. Deterministic: pure functions
and sorted iteration, so on the run's own YAML the decision lines equal the
pipeline's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .brands import load_brand_list
from .config import VideoCfg, build_dataclass, resolve_config
from .schema import CSV_COLUMNS
from .stages.base import FIRING_LOG_FILE, ROUTER_INPUT_FILE, ROUTER_OUTPUT_FILE, RUN_MANIFEST, dumps_line, utcnow_iso
from .stages.router.engine import enabled_rules, route_rows
from .stages.router.inputs import known_channel_brands

PATH_COLUMNS = ("crop_path", "enlarged_crop_path", "loose_crop_path", "frame_path", "annotated_frame_path")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not obj.get("_header"):
                rows.append(obj)
    return rows


def video_config_for(run_dir: Path, video_yaml: str | None) -> VideoCfg | dict:
    if video_yaml:
        return video_yaml
    manifest = json.loads((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))
    if "video_config" in manifest:
        return dict(manifest["video_config"])
    # Manifests written before ingest recorded the video file: identity only (Rule 5 then uses the
    # snapshot's channel brands, not the region entry's).
    return {"video_id": manifest["video_id"], "source": manifest.get("source") or "unknown",
            "channel": manifest.get("channel"), "channel_brand": manifest.get("channel_brand")}


def replay(run_dir: Path, pipeline_yaml: str, video_yaml: str | None = None, input_rel: str = ROUTER_INPUT_FILE,
           repo_root: Path | None = None):
    """``(outputs, firing, resolved)`` — the engine over ``input_rel`` under ``pipeline_yaml``."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    video = video_config_for(run_dir, video_yaml)
    manifest = json.loads((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))
    frames_dir = None
    if (manifest.get("mode") == "frames"):
        frames_dir = (manifest.get("config") or {}).get("ingest", {}).get("frames", {}).get("dir") or "."
    resolved = resolve_config(pipeline_yaml, video, frames_dir=frames_dir)
    cfg = resolved.pipeline
    bpath = Path(cfg.brands.list_path)
    brands = load_brand_list(bpath if bpath.is_absolute() else repo_root / bpath, cfg.brands.require_hash_match)
    channels = known_channel_brands(repo_root, resolved.video, brands, cfg.watermark)
    rows = _read_jsonl(run_dir / input_rel)
    outputs, firing = route_rows(rows, cfg, brands, channels)
    return outputs, firing, resolved


def write(run_dir: Path, out_dir: Path, outputs: list[dict], firing: list[dict], resolved) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {"_header": True, "stage": "router", "config_hash": resolved.config_hash,
              "video_id": resolved.video.video_id, "written_at": utcnow_iso(), "replay_of": str(run_dir),
              "not_logo_confidence_source": resolved.pipeline.router.not_logo_confidence_source,
              "enhanced_vlm_fields_available": resolved.pipeline.router.enhanced_vlm_fields_available,
              "enabled_rules": [r.name for r in enabled_rules(resolved.pipeline)]}
    for name, rows in ((ROUTER_OUTPUT_FILE, outputs), (FIRING_LOG_FILE, firing)):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            fh.write(dumps_line(header) + "\n")
            for r in rows:
                fh.write(dumps_line(r) + "\n")

    csv_written = False
    src_csv = run_dir / resolved.pipeline.output.csv_name
    if src_csv.exists():
        df = pd.read_csv(src_csv)
        by_det = {o["det_id"]: o for o in outputs}
        for col in [c for c in outputs[0] if c != "det_id"] if outputs else []:
            vals = [by_det.get(d, {}).get(col) for d in df["det_id"]]
            df[col] = [json.dumps(v) if isinstance(v, list) else v for v in vals]
        for col in PATH_COLUMNS:            # re-express run-relative paths for the new CSV location
            if col in df.columns:
                df[col] = [Path(os.path.relpath((src_csv.parent / p).resolve(), out_dir.resolve())).as_posix()
                           if isinstance(p, str) and p and not Path(p).is_absolute() else p for p in df[col]]
        df[[c for c in CSV_COLUMNS if c in df.columns]].to_csv(out_dir / "rows.csv", index=False)
        csv_written = True
    return {"rows": len(outputs), "out": str(out_dir), "rows_csv": csv_written}


def decision_lines(path: Path) -> list[str]:
    """The data lines of a router_output/firing_log file (the header carries a timestamp)."""
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and '"_header": true' not in ln]

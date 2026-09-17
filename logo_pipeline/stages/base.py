"""Stage contract, run context and the JSONL conventions (03 §3, §4).

* A stage reads only from the run directory and the config.
* Per-detection outputs are JSON Lines keyed by ``det_id``; per-frame outputs
  by ``frame_idx``. The first line of every JSONL file is a header
  ``{"_header": true, "stage": …, "config_hash": …}``.
* ``JsonlAppender`` gives per-record resumability to the slow GPU stages.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from ..config import PipelineCfg, VideoCfg
from ..logging_utils import StageLogger

# Run-directory file names (03 §4)
RUN_MANIFEST = "run_manifest.json"
STAGE_STATUS = "stage_status.json"
SHOTS_FILE = "shots.json"
SAMPLED_FRAMES_FILE = "sampled_frames.json"
RAW_DETECTIONS_FILE = "raw_detections.jsonl"
MERGED_DETECTIONS_FILE = "merged_detections.jsonl"
MERGE_LOG_FILE = "merge_log.jsonl"
MERGE_COUNTERS_FILE = "merge_counters.json"
REFINEMENT_FILE = "refinement.jsonl"
OCR_FILE = "ocr.jsonl"
FAISS_HINTS_FILE = "faiss_hints.jsonl"
AUDIO_HINTS_FILE = "audio/audio_hints.json"
AUDIO_WAV_FILE = "audio/audio.wav"
VLM_HINTED_FILE = "vlm/hinted.jsonl"
VLM_BLIND_FILE = "vlm/blind.jsonl"
VLM_HINT_COMPOSITION_FILE = "vlm/hint_composition.jsonl"
CANONICAL_FILE = "canonical.jsonl"
WATERMARK_FILE = "watermark.jsonl"
ROUTER_INPUT_FILE = "router_input.jsonl"
ROUTER_OUTPUT_FILE = "router_output.jsonl"
FIRING_LOG_FILE = "firing_log.jsonl"
FRAMES_DIR = "frames"
TEMPORAL_DIR = "temporal"
CROPS_DIR = "crops"
ANNOTATED_FRAMES_DIR = "annotated_frames"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def frame_file_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}.jpg"


def det_dir_name(det_id: int) -> str:
    return f"det_{det_id:06d}"


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in obj]
    if np is not None and isinstance(obj, np.integer):
        return int(obj)
    if np is not None and isinstance(obj, np.floating):
        return float(obj)
    if np is not None and isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return _json_safe(asdict(obj))
    return obj


def dumps_line(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


@dataclass
class StageResult:
    counts: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class RunContext:
    """Everything a stage may touch: the run directory and the config."""

    def __init__(self, run_dir: Path, resolved, brands, log: StageLogger, *, force: bool = False,
                 repo_root: Path | None = None):
        self.run_dir = Path(run_dir)
        self.resolved = resolved
        self.cfg: PipelineCfg = resolved.pipeline
        self.video: VideoCfg = resolved.video
        self.config_hash: str = resolved.config_hash
        self.brands = brands
        self.log = log
        self.force = force
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self._manifest: dict | None = None
        self._sampled: list[dict] | None = None

    # -- paths -------------------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.run_dir / rel

    def rel(self, p: Path) -> str:
        return Path(p).relative_to(self.run_dir).as_posix()

    def frame_file(self, frame_idx: int) -> Path:
        return self.run_dir / FRAMES_DIR / frame_file_name(frame_idx)

    def det_dir(self, det_id: int) -> Path:
        return self.run_dir / CROPS_DIR / det_dir_name(det_id)

    def resolve_repo_path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.repo_root / q

    # -- json ----------------------------------------------------------------
    def read_json(self, rel: str) -> Any:
        with open(self.path(rel), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def header(self, stage: str, **extra) -> dict:
        h = {"_header": True, "stage": stage, "config_hash": self.config_hash,
             "video_id": self.video.video_id, "written_at": utcnow_iso()}
        h.update(extra)
        return h

    def read_jsonl(self, rel: str, *, with_header: bool = False) -> list[dict] | tuple[dict | None, list[dict]]:
        p = self.path(rel)
        header: dict | None = None
        rows: list[dict] = []
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("_header"):
                        header = obj if header is None else header
                        continue
                    rows.append(obj)
        return (header, rows) if with_header else rows

    def write_jsonl(self, rel: str, rows: Iterable[dict], header: dict | None = None) -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            if header is not None:
                fh.write(dumps_line(header) + "\n")
            for r in rows:
                fh.write(dumps_line(r) + "\n")
        return p

    def jsonl_by_key(self, rel: str, key: str = "det_id") -> dict[Any, dict]:
        return {r[key]: r for r in self.read_jsonl(rel)}

    # -- run state ----------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self.read_json(RUN_MANIFEST)
        return self._manifest

    def update_manifest(self, **updates) -> None:
        m = self.manifest
        m.update(updates)
        self.write_json(RUN_MANIFEST, m)

    def invalidate(self) -> None:
        self._manifest = None
        self._sampled = None

    def sampled_frames(self) -> list[dict]:
        if self._sampled is None:
            self._sampled = self.read_json(SAMPLED_FRAMES_FILE)
        return self._sampled

    def open_frame_image(self, frame_idx: int) -> Image.Image:
        return Image.open(self.frame_file(frame_idx)).convert("RGB")

    def open_source(self):
        """The configured frame source when it is reachable from this machine, else ``None``."""
        from ..frame_source import open_frame_source
        try:
            return open_frame_source(self.cfg, self.video, self.repo_root)
        except (FileNotFoundError, RuntimeError):
            return None


class JsonlAppender:
    """Append-mode JSONL writer for per-record resumability.

    On open it reads the ids already present (unless ``truncate``) so the
    caller can skip them; every ``write`` is flushed so a crash keeps
    progress.
    """

    def __init__(self, path: Path, header: dict, key: str = "det_id", truncate: bool = False):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[Any] = set()
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists and not truncate:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not obj.get("_header") and key in obj:
                        self.done.add(obj[key])
            self._fh = open(self.path, "a", encoding="utf-8")
        else:
            self._fh = open(self.path, "w", encoding="utf-8")
            self._fh.write(dumps_line(header) + "\n")
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._fh.write(dumps_line(row) + "\n")
        self._fh.flush()
        self.done.add(row[self.key])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Stage(abc.ABC):
    name: str = ""
    device: str = "cpu"                # "cpu" | "gpu"
    inputs: list[str] = []             # run-dir relative files it reads
    outputs: list[str] = []            # run-dir relative files it writes

    @abc.abstractmethod
    def run(self, ctx: RunContext) -> StageResult: ...

    def outputs_exist(self, ctx: RunContext) -> bool:
        return all(ctx.path(o).exists() for o in self.outputs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Stage {self.name} ({self.device})>"


def iter_grouped_by_frame(rows: Iterable[dict]) -> Iterator[tuple[int, list[dict]]]:
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame_idx"]), []).append(r)
    for fi in sorted(by_frame):
        yield fi, by_frame[fi]

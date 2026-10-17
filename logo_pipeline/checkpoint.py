"""T77 checkpoint support (``spec docs/14``): the reports bundle, the
self-check run before showing the client, and the client package.

* ``reports`` runs every report that applies to a run (decisions by reason
  code, merge counters and distributions, refinement fallback, canonicalisation,
  Rule 5 branches, per-stage timings, ordering checks, acceptance caveats).
* ``check`` executes the acceptance self-checks of ``14`` and writes
  ``reports/checkpoint_checklist.md`` (exit status 1 on a failure).
* ``package`` copies what the client sees into one folder with a note listing
  what is and is not yet in the run.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

REPORT_SCRIPTS = (
    ("metrics", ["--no-labels"]),
    ("merge_ablation", []),
    ("refinement_report", []),
    ("watermark_report", []),
    ("ordering_checks", []),
    ("vlm_cost", []),
    ("acceptance_evidence", []),
)


def _eval(repo_root: Path, name: str):
    p = str(Path(repo_root) / "eval")
    if p not in sys.path:
        sys.path.insert(0, p)
    return __import__(name)


def run_reports(run_dir: Path, repo_root: Path) -> dict[str, str]:
    """Runs each report script's ``main`` on the run; returns ``{name: "ok" | error}``."""
    results = {}
    for name, extra in REPORT_SCRIPTS:
        mod = _eval(repo_root, name)
        try:
            with redirect_stdout(io.StringIO()):
                rc = mod.main(["--run", str(run_dir), *extra])
            results[name] = "ok" if rc in (0, None) else f"exit {rc}"
        except SystemExit as e:
            results[name] = "ok" if e.code in (0, None) else f"exit {e.code}"
        except FileNotFoundError as e:
            results[name] = f"skipped: {e}"
    if (run_dir / "merge_log.jsonl").exists():
        from .harness import write_report
        from .stages.base import RunContext
        results["harness"] = "ok" if _harness(run_dir, repo_root, write_report, RunContext) else "skipped"
    return results


def _harness(run_dir: Path, repo_root: Path, write_report, RunContext) -> bool:
    from .config import resolve_config
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    video = manifest.get("video_config") or {"video_id": manifest["video_id"], "source": "unknown"}
    pipeline = manifest.get("config")
    if not pipeline:
        return False
    resolved = resolve_config(pipeline, {k: v for k, v in video.items() if k != "overrides"})

    class _Log:
        def __getattr__(self, _):
            return lambda *a, **k: None
    write_report(RunContext(run_dir, resolved, None, _Log(), repo_root=repo_root))
    return True


def checklist(run_dir: Path) -> list[tuple[str, bool, str]]:
    """The ``14`` acceptance self-checks: ``[(check, passed, detail)]``."""
    import pandas as pd
    out: list[tuple[str, bool, str]] = []
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    sampled = json.loads((run_dir / "sampled_frames.json").read_text(encoding="utf-8"))
    annotated = list((run_dir / "annotated_frames").glob("frame_*.jpg"))
    want = {f"frame_{f['frame_idx']:06d}.jpg" for f in sampled}
    have = {p.name for p in annotated}
    out.append(("Every sampled frame has an annotated image", want <= have,
                f"{len(want & have)} of {len(want)} sampled frames annotated"))

    csv = run_dir / "rows.csv"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    out.append(("rows.csv loads without warnings", csv.exists() and not caught,
                f"{len(df)} rows; {len(caught)} warning(s)"))
    if len(df):
        null_dec = df["router_decision"].isna()
        unparsed = df["vlm_parse_ok"].astype(str).str.lower() != "true"
        unparsed_ok = (df.loc[unparsed, "reason_code"] == "vlm_output_unparsed").all()
        out.append(("No null router_decision; unparsed rows show vlm_output_unparsed",
                    not null_dec.any() and bool(unparsed_ok),
                    f"{int(null_dec.sum())} null decisions; {int(unparsed.sum())} unparsed rows"))
        n_review = int((df["router_decision"] == "REVIEW").sum())
        n_unknown = int((df["router_decision"] == "UNKNOWN_LOGO").sum())
        out.append(("REVIEW and UNKNOWN_LOGO queues both non-empty", n_review > 0 and n_unknown > 0,
                    f"REVIEW {n_review}, UNKNOWN_LOGO {n_unknown}"))
        n_ch = int(df["reason_code"].astype(str).str.startswith("channel_brand_").sum())
        out.append(("At least one channel_brand_* row", n_ch > 0, f"{n_ch} rows"))
        prov = {c: df[c].notna().all() and (df[c].astype(str) != "").all()
                for c in ("config_hash", "brand_list_version", "brand_list_hash", "git_sha") if c in df.columns}
        missing_cols = {"config_hash", "brand_list_version", "brand_list_hash", "git_sha"} - set(df.columns)
        out.append(("Config hash, brand list version/hash and git SHA on every row",
                    all(prov.values()) and not missing_cols,
                    ", ".join(f"{c}: {'ok' if v else 'missing'}" for c, v in prov.items())
                    + (f"; columns absent: {sorted(missing_cols)}" if missing_cols else "")))
    in_manifest = {k: bool(manifest.get(k)) for k in ("config_hash", "brand_list_version", "brand_list_hash",
                                                      "git_sha")}
    out.append(("Config hash, brand list version/hash and git SHA in the manifest", all(in_manifest.values()),
                ", ".join(f"{k}: {'ok' if v else 'missing'}" for k, v in in_manifest.items())))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from ordering_checks import check_firing_log, read_firing_log
    fl = run_dir / "firing_log.jsonl"
    if fl.exists():
        v = check_firing_log(read_firing_log(run_dir))
        out.append(("The two T65 ordering checks pass", not any(v.values()),
                    "; ".join(f"{k}: {len(d)} violation(s)" for k, d in v.items())))
    else:
        out.append(("The two T65 ordering checks pass", False, "firing_log.jsonl missing"))
    return out


def render_checklist(run_dir: Path, items: list[tuple[str, bool, str]], replay: tuple[bool, str] | None) -> str:
    lines = [f"# T77 checkpoint self-check — `{run_dir}`", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {c} | {'pass' if ok else 'FAIL'} | {d} |" for c, ok, d in items]
    if replay is not None:
        lines.append(f"| CPU replay rows.csv identical (step 3) | {'pass' if replay[0] else 'FAIL'} | {replay[1]} |")
    lines += ["", "Manual items (record by hand): code frozen at tag `m3-checkpoint` with its SHA in the manifest; "
              "the walkthrough (CSV in the review screen, both queues, a Rule 5 row's evidence panel, one Accept into "
              "FAISS on a copy of the index)."]
    return "\n".join(lines) + "\n"


def compare_csv(a: Path, b: Path) -> tuple[bool, str]:
    if not a.exists() or not b.exists():
        return False, f"missing: {a if not a.exists() else b}"
    same = a.read_bytes() == b.read_bytes()
    return same, "byte-identical" if same else "files differ"


def note(run_dir: Path) -> str:
    """The client note: what is and is not yet in the run (from the acceptance caveats)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from acceptance_evidence import caveats
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "stage_status.json").read_text()) if (run_dir / "stage_status.json").exists() else {}
    has_blind = (run_dir / "vlm" / "blind.jsonl").exists()
    has_scores = (run_dir / "vlm" / "scores.jsonl").exists()
    router = (status.get("router") or {}).get("counts", {})
    lines = [f"# Checkpoint T77 — {manifest.get('video_id')}", "",
             f"Run `{manifest.get('run_id')}`, config hash `{manifest.get('config_hash')}`, git `{manifest.get('git_sha')}`, "
             f"brand list `{manifest.get('brand_list_version')}`.", "",
             "## What is in this package", "",
             "- `annotated_frames/` — every sampled frame, boxes coloured by router decision; `annotated.mp4` plays them "
             "at the source's real time.",
             "- `rows.csv` — one row per detection in the frozen schema (`configs/row_schema` doc); loads in the review "
             "screen as is (REVIEW and UNKNOWN_LOGO queues preset).",
             "- `configs/` — the exact YAML, the per-video file, `reason_codes.md`.",
             "- `reports/` — decision distribution by reason code, merge counters and distributions, refinement "
             "fallback, canonicalisation modes, Rule 5 branches, per-stage timings, ordering checks.", "",
             "## Decisions in this run", ""]
    lines += [f"- {k}: {v}" for k, v in router.items() if k not in ("rows",)]
    lines += ["", "## What is not yet in this run", ""]
    lines += [f"- {c}" for c in caveats(manifest, status, has_blind, has_scores)]
    lines += ["", "## What your feedback can change before Group 10", "",
              "Thresholds (YAML only), reason-code wording, the annotated-frame legend, queue presets. A rule-logic "
              "defect is fixed in the router and replayed with the router-only runner (no GPU re-run)."]
    return "\n".join(lines) + "\n"


def package(run_dir: Path, out_dir: Path, repo_root: Path, video_yaml: str | None = None,
            pipeline_yaml: str = "configs/pipeline.yaml") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("annotated_frames", "reports"):
        if (run_dir / name).exists():
            shutil.copytree(run_dir / name, out_dir / name, dirs_exist_ok=True)
    for name in ("annotated.mp4", "rows.csv", "run_manifest.json"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out_dir / name)
    # rows.csv paths resolve against the CSV's folder: bring the images they name.
    for name in ("crops", "frames"):
        if (run_dir / name).exists():
            shutil.copytree(run_dir / name, out_dir / name, dirs_exist_ok=True)
    cfg_out = out_dir / "configs"
    cfg_out.mkdir(exist_ok=True)
    for src in (pipeline_yaml, "configs/reason_codes.md", video_yaml):
        if src and (Path(repo_root) / src).exists():
            shutil.copyfile(Path(repo_root) / src, cfg_out / Path(src).name)
    (out_dir / "CHECKPOINT_NOTE.md").write_text(note(run_dir), encoding="utf-8")
    return out_dir

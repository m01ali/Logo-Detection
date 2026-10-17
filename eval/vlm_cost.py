#!/usr/bin/env python3
"""T36 (``spec docs/15``): measured VLM cost, replacing Revision 3's planning
figure (~13 h of GPU for two passes over ~1,739 detections).

Reads the per-call ``elapsed_s`` on ``vlm/hinted.jsonl``, ``vlm/blind.jsonl``
and ``vlm/scores.jsonl`` (falling back to ``stage_status.json`` stage time /
calls for runs written before per-call timings existed), the spike report's
rows if present, and the detection counts; writes ``reports/vlm_cost.md``:
seconds per hinted call, per blind call and per scoring pass; detections per
sampled frame before and after the merge; projected hours for this run and for
the route-three tiers of ``17`` §T66.

Usage::

    python eval/vlm_cost.py --run runs/<video_id>/<run_id>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import md_table, read_json, read_jsonl, reports_dir  # noqa: E402

# 17 §T66: route-three tiers, approximate detections (hinted pass only).
TIERS = (("tier 1: videos 14, 1", 1000), ("tier 2: + video 27", 3000), ("tier 3: + videos 98, 100", 7000))


def per_call(records: list[dict]) -> list[float]:
    return [float(r["elapsed_s"]) for r in records if r.get("elapsed_s") is not None]


def summarise(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "p90": v[min(len(v) - 1, int(0.9 * len(v)))]}


def analyse(run_dir: Path) -> dict:
    status = read_json(run_dir, "stage_status.json", default={})
    hinted = read_jsonl(run_dir, "vlm/hinted.jsonl", required=False)
    blind = read_jsonl(run_dir, "vlm/blind.jsonl", required=False)
    scores = read_jsonl(run_dir, "vlm/scores.jsonl", required=False)
    spike = read_jsonl(run_dir, "reports/vlm_spike.jsonl", required=False)
    passes = {"hinted call": summarise(per_call(hinted)), "blind call": summarise(per_call(blind)),
              "scoring pass": summarise(per_call(scores) or per_call(spike))}
    fallback = None
    vlm = status.get("vlm", {})
    calls = (vlm.get("counts") or {}).get("calls") or 0
    if passes["hinted call"] is None and calls and vlm.get("elapsed_s"):
        # One stage timer over both passes when blind calls were made in the same stage.
        n_calls = calls + ((vlm.get("counts") or {}).get("blind_calls") or 0)
        fallback = vlm["elapsed_s"] / n_calls
    n_sampled = len(read_json(run_dir, "sampled_frames.json", default=[]))
    n_raw = len(read_jsonl(run_dir, "raw_detections.jsonl", required=False))
    n_reps = len(read_jsonl(run_dir, "merged_detections.jsonl", required=False))
    return {"passes": passes, "fallback_per_call": fallback, "n_sampled": n_sampled, "n_raw": n_raw,
            "n_reps": n_reps}


def render(run_dir: Path, a: dict) -> str:
    lines = ["# T36 VLM cost", "", f"Run: `{run_dir}`.", "", "## Seconds per call", ""]
    rows = []
    for name, s in a["passes"].items():
        rows.append((name, s["n"], f"{s['mean']:.2f}", f"{s['median']:.2f}", f"{s['p90']:.2f}") if s
                    else (name, 0, "—", "—", "—"))
    lines += md_table(["pass", "n", "mean s", "median s", "p90 s"], rows)
    if a["fallback_per_call"] is not None:
        lines += ["", f"This run predates per-call timings: `vlm` stage time / calls = "
                  f"{a['fallback_per_call']:.2f} s per call (hinted and blind not separated)."]
    per_frame_raw = a["n_raw"] / a["n_sampled"] if a["n_sampled"] else 0
    per_frame_reps = a["n_reps"] / a["n_sampled"] if a["n_sampled"] else 0
    lines += ["", "## Detections per sampled frame", "",
              f"- {a['n_sampled']} sampled frames; {a['n_raw']} raw boxes ({per_frame_raw:.1f}/frame) before the "
              f"merge, {a['n_reps']} representatives ({per_frame_reps:.1f}/frame) after.", ""]

    def secs(name):
        s = a["passes"][name]
        if s:
            return s["mean"]
        return a["fallback_per_call"] if name == "hinted call" else None

    h, b, sc = secs("hinted call"), secs("blind call"), secs("scoring pass")
    lines += ["## Projected GPU hours", ""]
    proj = [("this run", a["n_reps"])] + list(TIERS)
    prow = []
    for label, n in proj:
        hinted_h = n * h / 3600 if h else None
        both_h = n * (h + b) / 3600 if (h and b) else None
        score_h = n * sc / 3600 if sc else None
        prow.append((label, n, f"{hinted_h:.1f}" if hinted_h is not None else "—",
                     f"{both_h:.1f}" if both_h is not None else "—", f"{score_h:.1f}" if score_h is not None else "—"))
    lines += md_table(["scope", "detections", "hinted only (h)", "hinted + blind (h)", "+ scoring (h)"], prow)
    lines += ["", "Route three runs the hinted pass only (T64 does not need the blind pass there). "
              "Revision 3's planning figure was ~13 h for two passes over ~1,739 detections."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    out = reports_dir(run_dir, args.out)
    (out / "vlm_cost.md").write_text(render(run_dir, analyse(run_dir)), encoding="utf-8")
    print(f"wrote {out / 'vlm_cost.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

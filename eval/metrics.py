#!/usr/bin/env python3
"""T62 (``spec docs/17``): router metrics for a run.

Labels file schema (the review UI's export satisfies it):
``det_id, video_id, run_id, label ∈ {TP, FP, FN, wrong_brand}, true_brand (optional)``;
rows join on ``(video_id, run_id, det_id)``. Without ``--labels`` the run's own
``rows.csv`` ``label`` column is used when present; ``--no-labels`` ignores both
(the decision distribution published after the ``video_1`` run).

Writes ``reports/metrics.md`` + CSVs: the decision distribution with shares
and the review load (REVIEW + UNKNOWN_LOGO); per-rule (deciding) and
per-reason-code counts, with pre-empted counts from the firing log;
per-reason-code precision ``TP / (TP + FP [+ wrong_brand])`` with n; ACCEPT
precision on its own line; ``brand_evidence`` overall and by decision;
``canonical_match_mode``, ``refinement_status`` and Rule 5 branch counts.

``--wrong-brand-as fp`` counts ``wrong_brand`` as a false positive;
``separate`` leaves it out of the precision and reports it on its own.

Usage::

    python eval/metrics.py --run runs/<video_id>/<run_id> [--labels labels.csv] [--wrong-brand-as fp|separate]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import joined_rows, md_table, pct, read_jsonl, reports_dir, write_csv  # noqa: E402

DECISIONS = ("ACCEPT", "REVIEW", "UNKNOWN_LOGO", "REJECT", "DEDUPED")
LABELS = ("TP", "FP", "FN", "wrong_brand")


def load_labels(path: str | Path) -> dict[tuple, dict]:
    import pandas as pd
    df = pd.read_csv(path)
    missing = {"det_id", "video_id", "run_id", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"labels file lacks column(s) {sorted(missing)}")
    out = {}
    for r in df.to_dict("records"):
        lab = r.get("label")
        if isinstance(lab, str) and lab in LABELS:
            out[(str(r["video_id"]), str(r["run_id"]), int(r["det_id"]))] = r
    return out


def labels_from_csv(run_dir: Path, csv_name: str = "rows.csv") -> dict[tuple, dict]:
    p = Path(run_dir) / csv_name
    if not p.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(p)
    if "label" not in df.columns:
        return {}
    return {(str(r["video_id"]), str(r["run_id"]), int(r["det_id"])): r for r in df.to_dict("records")
            if isinstance(r.get("label"), str) and r["label"] in LABELS}


def precision(c: Counter, wrong_brand_as: str) -> tuple[float | None, int]:
    tp, fp, wb = c.get("TP", 0), c.get("FP", 0), c.get("wrong_brand", 0)
    denom = tp + fp + (wb if wrong_brand_as == "fp" else 0)
    return (tp / denom if denom else None), denom


def compute(rows: list[dict], firing: list[dict], labels: dict[tuple, dict], wrong_brand_as: str = "fp") -> dict:
    n = len(rows)
    decisions = Counter(r.get("router_decision") for r in rows)
    reasons = Counter((r.get("router_decision"), r.get("reason_code")) for r in rows)
    deciding = Counter(f.get("deciding_rule") or "default" for f in firing)
    preempted = Counter(m["rule"] for f in firing for m in f.get("rules_matched_preempted", []))
    rule5 = Counter(f.get("rule5_branch") for f in firing if f.get("rule5_branch"))

    by_code: dict[str, Counter] = defaultdict(Counter)
    accept = Counter()
    labelled = 0
    for r in rows:
        lab = labels.get((str(r.get("video_id")), str(r.get("run_id")), int(r["det_id"])))
        if not lab:
            continue
        labelled += 1
        by_code[r.get("reason_code")][lab["label"]] += 1
        if r.get("router_decision") == "ACCEPT":
            accept[lab["label"]] += 1
    evidence = Counter(r.get("brand_evidence") for r in rows)
    evidence_by_dec = Counter((r.get("router_decision"), r.get("brand_evidence")) for r in rows)
    return {
        "n": n, "decisions": decisions, "reasons": reasons, "deciding": deciding, "preempted": preempted,
        "review_load": decisions.get("REVIEW", 0) + decisions.get("UNKNOWN_LOGO", 0),
        "labelled": labelled, "by_code": dict(by_code), "accept": accept, "wrong_brand_as": wrong_brand_as,
        "evidence": evidence, "evidence_by_dec": evidence_by_dec,
        "match_mode": Counter(r.get("canonical_match_mode") for r in rows),
        "refinement": Counter(r.get("refinement_status") for r in rows), "rule5": rule5,
    }


def render(run_dir: Path, m: dict) -> str:
    n = m["n"]
    lines = [f"# Router metrics — `{run_dir}`", "", f"{n} detections; {m['labelled']} labelled.", "",
             "## Decision distribution", ""]
    lines += md_table(["decision", "rows", "share"], [(d, m["decisions"].get(d, 0), pct(m["decisions"].get(d, 0), n))
                                                      for d in DECISIONS] +
                      [(d, c, pct(c, n)) for d, c in m["decisions"].items() if d not in DECISIONS])
    lines += ["", f"**Review load** (REVIEW + UNKNOWN_LOGO): {m['review_load']} ({pct(m['review_load'], n)}).", "",
              "## Deciding rule", ""]
    lines += md_table(["rule", "rows deciding", "times pre-empted"],
                      [(r, c, m["preempted"].get(r, 0)) for r, c in sorted(m["deciding"].items(), key=lambda kv: -kv[1])]
                      + [(r, 0, c) for r, c in m["preempted"].items() if r not in m["deciding"]])
    lines += ["", "## Reason codes", ""]
    rows = []
    for (dec, code), c in sorted(m["reasons"].items(), key=lambda kv: -kv[1]):
        lc = m["by_code"].get(code, Counter())
        p, denom = precision(lc, m["wrong_brand_as"])
        rows.append((dec, code, c, sum(lc.values()), f"{p:.3f}" if p is not None else "—", denom,
                     lc.get("wrong_brand", 0)))
    lines += md_table(["decision", "reason code", "rows", "labelled", "precision", "n (denominator)", "wrong_brand"],
                      rows)
    p, denom = precision(m["accept"], m["wrong_brand_as"])
    lines += ["", f"**ACCEPT precision:** {f'{p:.3f}' if p is not None else 'n/a (no labelled ACCEPT rows)'} "
              f"(n = {denom}; wrong_brand counted {'as FP' if m['wrong_brand_as'] == 'fp' else 'separately'}).", "",
              "## brand_evidence", ""]
    lines += md_table(["brand_evidence", "rows", "share"], [(k, c, pct(c, n)) for k, c in m["evidence"].most_common()])
    lines += [""]
    lines += md_table(["decision", "brand_evidence", "rows"],
                      [(d, e, c) for (d, e), c in sorted(m["evidence_by_dec"].items(), key=lambda kv: (str(kv[0][0]),
                                                                                                      -kv[1]))])
    for title, key in (("canonical_match_mode", "match_mode"), ("refinement_status", "refinement"),
                       ("Rule 5 branch", "rule5")):
        lines += ["", f"## {title}", ""]
        lines += md_table([title, "rows"], [(k, c) for k, c in m[key].most_common()]) if m[key] else ["None."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--wrong-brand-as", choices=("fp", "separate"), default="fp")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    rows = joined_rows(run_dir)
    firing = read_jsonl(run_dir, "firing_log.jsonl", required=False)
    labels = {} if args.no_labels else (load_labels(args.labels) if args.labels else labels_from_csv(run_dir))
    m = compute(rows, firing, labels, args.wrong_brand_as)
    out = reports_dir(run_dir, args.out)
    (out / "metrics.md").write_text(render(run_dir, m), encoding="utf-8")
    write_csv(out / "metrics_decisions.csv", ["decision", "rows"], m["decisions"].items())
    write_csv(out / "metrics_reason_codes.csv", ["decision", "reason_code", "rows", "TP", "FP", "FN", "wrong_brand"],
              [(d, c, n, *(m["by_code"].get(c, Counter()).get(k, 0) for k in LABELS))
               for (d, c), n in m["reasons"].items()])
    print(f"wrote {out / 'metrics.md'} ({m['n']} rows, {m['labelled']} labelled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

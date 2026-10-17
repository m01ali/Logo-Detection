"""Shared run-directory readers for the ``eval/`` scripts: JSON Lines without
their header line, joined rows, and the ``reports/`` writer every script uses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(run_dir: Path, rel: str, *, required: bool = True) -> list[dict]:
    path = Path(run_dir) / rel
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} not found")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if not obj.get("_header"):
                    rows.append(obj)
    return rows


def read_json(run_dir: Path, rel: str, default=None):
    path = Path(run_dir) / rel
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def by_det(rows: Iterable[dict]) -> dict[int, dict]:
    return {r["det_id"]: r for r in rows}


def joined_rows(run_dir: Path) -> list[dict]:
    """``router_input.jsonl`` with ``router_output.jsonl``'s decisions laid over it, in ``det_id`` order."""
    inputs = by_det(read_jsonl(run_dir, "router_input.jsonl"))
    outputs = by_det(read_jsonl(run_dir, "router_output.jsonl", required=False))
    rows = []
    for det_id in sorted(inputs):
        r = dict(inputs[det_id])
        r.update({k: v for k, v in outputs.get(det_id, {}).items() if k != "det_id"})
        rows.append(r)
    return rows


def reports_dir(run_dir: Path, out: str | None = None) -> Path:
    d = Path(out) if out else Path(run_dir) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def md_table(header: list[str], rows: Iterable[Iterable]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"

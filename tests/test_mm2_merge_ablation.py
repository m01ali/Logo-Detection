"""spec docs/08 T19 — eval/merge_ablation.py: histograms + the
"would otherwise have merged" count that quantifies the area floor's extra
VLM-call cost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from eval.merge_ablation import analyse, main, read_merge_log  # noqa: E402

HEADER = {"_header": True, "stage": "merge", "iou_threshold": 0.70, "containment_threshold": 0.90,
          "area_ratio_threshold": 0.80, "min_merge_area": 0.0004}


def _write_log(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(HEADER) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_analyse_counts_floor_pairs_that_would_have_merged():
    rows = [
        {"outcome": "merged", "iou": 1.0, "contain": 1.0, "area_ratio": 1.0, "smaller_fraction": 0.01},
        {"outcome": "failed_size_precondition", "iou": 0.1, "contain": 0.95, "area_ratio": 0.4,
         "smaller_fraction": 0.02},
        {"outcome": "no_overlap", "iou": 0.0, "contain": 0.0, "area_ratio": 0.9, "smaller_fraction": 0.02},
        # skipped by the floor, but overlap+ratio would have satisfied "merged" if the floor didn't apply:
        {"outcome": "skipped_area_floor", "iou": 0.95, "contain": 0.95, "area_ratio": 0.95,
         "smaller_fraction": 0.0001},
        # skipped by the floor, and would NOT have merged anyway (fails overlap):
        {"outcome": "skipped_area_floor", "iou": 0.0, "contain": 0.0, "area_ratio": 0.5,
         "smaller_fraction": 0.0002},
    ]
    stats = analyse(HEADER, rows)
    assert stats["n_pairs_evaluated"] == 5
    assert stats["n_failed_size_precondition"] == 1
    assert stats["n_skipped_area_floor"] == 2
    assert stats["n_overlapping_pairs"] == 3   # merged + failed_size_precondition(contain 0.95) + floor(iou 0.95)
    assert stats["n_floor_overlapping"] == 1
    assert stats["n_floor_would_have_merged"] == 1
    hist = stats["histograms"]["area_ratio_failed_size_precondition"]
    assert sum(b["count"] for b in hist) == 1


def test_main_writes_reports(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_log(run_dir / "merge_log.jsonl", [
        {"outcome": "merged", "iou": 1.0, "contain": 1.0, "area_ratio": 1.0, "smaller_fraction": 0.01},
        {"outcome": "failed_size_precondition", "iou": 0.1, "contain": 0.95, "area_ratio": 0.4,
         "smaller_fraction": 0.02},
    ])
    import contextlib
    import io

    from unittest.mock import patch

    with patch("sys.argv", ["merge_ablation.py", "--run", str(run_dir)]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main()
    assert rc == 0
    assert (run_dir / "reports" / "merge_distributions.md").exists()
    assert (run_dir / "reports" / "merge_distributions.csv").exists()
    md = (run_dir / "reports" / "merge_distributions.md").read_text()
    assert "Pairs evaluated: 2" in md


def test_read_merge_log_skips_header(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_log(run_dir / "merge_log.jsonl", [
        {"outcome": "merged", "iou": 1.0, "contain": 1.0, "area_ratio": 1.0, "smaller_fraction": 0.01},
    ])
    header, rows = read_merge_log(run_dir)
    assert header["iou_threshold"] == 0.70
    assert len(rows) == 1

"""spec docs/09 T24 — eval/refinement_report.py: status shares, fallback
rate, rejected-safety-check breakdown, and the OCR-brand before/after
metric (only becomes non-trivial once MM4 populates ocr_brand_match)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from eval.refinement_report import analyse, main  # noqa: E402

ROWS = [
    {"det_id": 0, "refinement_status": "applied", "refinement_area_ratio": 0.2,
     "ocr_boxes": [{"matched_brand": "nike"}]},
    {"det_id": 1, "refinement_status": "applied", "refinement_area_ratio": 0.4,
     "ocr_boxes": [{"matched_brand": "bmw"}]},
    {"det_id": 2, "refinement_status": "rejected_safety_check", "refinement_area_ratio": None,
     "safety_results": {"inside_loose": True, "min_side_px": True, "area_ratio_vs_loose": False,
                         "max_aspect": True}},
    {"det_id": 3, "refinement_status": "rejected_safety_check", "refinement_area_ratio": None,
     "safety_results": {"inside_loose": True, "min_side_px": False, "area_ratio_vs_loose": False,
                         "max_aspect": True}},
    {"det_id": 4, "refinement_status": "oversized_region", "refinement_area_ratio": None},
    {"det_id": 5, "refinement_status": "no_candidate", "refinement_area_ratio": None},
]


def test_analyse_without_ocr_matching_available():
    stats = analyse(ROWS, [])
    assert stats["n_rows"] == 6
    assert stats["applied"] == 2
    assert stats["fallback_rate"] == 1 - 2 / 6
    assert stats["failing_checks"] == {"area_ratio_vs_loose": 2, "min_side_px": 1}
    assert stats["oversized_region"] == 1
    assert abs(stats["mean_applied_area_ratio"] - 0.3) < 1e-9
    assert stats["ocr_matching_available"] is False


def test_analyse_with_ocr_matching_counts_tight_vs_loose_disagreement():
    ocr_rows = [
        {"det_id": 0, "ocr_brand_match": "nike"},    # tight agrees with loose (nike found both places)
        {"det_id": 1, "ocr_brand_match": None},       # tight missed it, loose found "bmw"
    ]
    stats = analyse(ROWS, ocr_rows)
    assert stats["ocr_matching_available"] is True
    assert stats["n_applied_rows"] == 2
    assert stats["tight_found_not_loose"] == 0
    assert stats["loose_found_not_tight"] == 1   # det_id 1


def test_main_writes_report_and_flags_high_fallback(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [{"det_id": i, "refinement_status": "no_candidate", "refinement_area_ratio": None} for i in range(11)]
    rows[0]["refinement_status"] = "applied"
    rows[0]["refinement_area_ratio"] = 0.3
    with open(run_dir / "refinement.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": True, "stage": "refine"}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    import contextlib
    import io
    from unittest.mock import patch

    with patch("sys.argv", ["refinement_report.py", "--run", str(run_dir)]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main()
    assert rc == 0
    md = (run_dir / "reports" / "refinement.md").read_text(encoding="utf-8")
    assert "Total rows: 11" in md
    assert "exceeds 90%" in md   # 10/11 no_candidate -> ~90.9% fallback, clears the 90% flag threshold

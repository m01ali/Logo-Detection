"""21_test_plan.md — MM1: CSV_COLUMNS covers every Row field and nothing
else; export_schema_doc.py renders 05 without error."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields

from logo_pipeline.schema import CSV_COLUMNS, Row, SECTION_TITLES, row_to_csv_dict


def test_csv_columns_covers_every_field_exactly_once():
    field_names = [f.name for f in fields(Row)]
    assert CSV_COLUMNS == field_names
    assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS))


def test_every_field_has_a_known_section():
    for f in fields(Row):
        assert f.metadata.get("section") in SECTION_TITLES, f"{f.name} has no/unknown section tag"


def test_row_to_csv_dict_serialises_lists_as_json_and_keeps_scalars():
    row = Row(video_id="v", run_id="r", det_id=0, frame_idx=0, timecode_s=0.0,
              frame_width=100, frame_height=100, det_score=0.5,
              box_xyxy=[1.0, 2.0, 3.0, 4.0], audio_brands=["rai", "verisure"], is_truncated=True)
    d = row_to_csv_dict(row)
    assert d["box_xyxy"] == "[1.0, 2.0, 3.0, 4.0]"
    assert d["audio_brands"] == '["rai", "verisure"]'
    assert d["video_id"] == "v"
    assert d["is_truncated"] is True   # booleans are not JSON-string-encoded


def test_export_schema_doc_runs_and_covers_every_field(tmp_path, repo_root):
    out = tmp_path / "schema.md"
    result = subprocess.run([sys.executable, str(repo_root / "scripts" / "export_schema_doc.py"), "--out", str(out)],
                             capture_output=True, text=True, cwd=repo_root)
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    for f in fields(Row):
        assert f"`{f.name}`" in text, f"{f.name} missing from generated schema doc"

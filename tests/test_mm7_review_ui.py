"""MM7 T73/T75: the review screen loads an M2 CSV unchanged and an M3 rows.csv
with queue presets, the router and Rule 5 panels, and Accept into FAISS
(smoke tests via streamlit.testing.v1.AppTest; the FAISS class is replaced by
an in-memory index so no model is loaded)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from logo_pipeline.schema import CSV_COLUMNS, Row, row_to_csv_dict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "app" / "review_ui.py")

M2_COLUMNS = ["det_id", "frame_idx", "timecode_s", "det_score", "box_xyxy", "crop_path", "enlarged_crop_path",
              "frame_path", "is_logo", "brand", "confidence", "signals_used", "reasoning", "faiss_top1",
              "faiss_top1_score", "faiss_top5_brands", "faiss_top5_scores", "audio_brands", "label"]


def _m2_csv(tmp_path) -> Path:
    rows = [[i, 44, 1.5, 0.3, "[1, 2, 3, 4]", "missing.jpg", "missing.jpg", "missing.jpg", lab, brand, 80, "[]",
             "r", "x", 0.9, "[]", "[]", "[]", None] for i, (lab, brand) in
            enumerate([("logo", "rai"), ("not_logo", None), ("partial_logo", "verisure")])]
    p = tmp_path / "vlm_predictions.csv"
    pd.DataFrame(rows, columns=M2_COLUMNS).to_csv(p, index=False)
    return p


def _m3_csv(tmp_path) -> Path:
    (tmp_path / "crops").mkdir()
    specs = [  # det_id, decision, reason, canonical_brand
        (0, "REVIEW", "channel_brand_outside_watermark_region", "rai sport"),
        (1, "UNKNOWN_LOGO", "logo_like_but_brand_unknown", "UNKNOWN"),
        (2, "REJECT", "high_confidence_not_logo", None),
        (3, "ACCEPT", "clear_logo_with_strong_ocr_match", "hp"),
    ]
    rows = []
    for det_id, dec, code, brand in specs:
        Image.new("RGB", (16, 16), (det_id * 40, 0, 0)).save(tmp_path / "crops" / f"c{det_id}.jpg")
        rows.append(row_to_csv_dict(Row(
            video_id="video_1", run_id="r1", det_id=det_id, frame_idx=0, is_logo="logo", vlm_decision="logo",
            vlm_parse_ok=True, confidence=80, logo_probability=90, router_decision=dec, reason_code=code,
            router_reason=code, router_rule=5 if dec == "REVIEW" else None, canonical_brand=brand,
            crop_path=f"crops/c{det_id}.jpg", loose_crop_path=f"crops/c{det_id}.jpg", ocr_text="rai",
            ocr_fuzzy_score=40, watermark_overlap=False, watermark_overlap_ratio=0.1, rules_matched_preempted=[16])))
    p = tmp_path / "rows.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(p, index=False)
    return p


def _open(csv: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input(key="csv_input").set_value(str(csv)).run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def test_m2_csv_still_loads(tmp_path):
    at = _open(_m2_csv(tmp_path))
    assert "Detection" in _text(at) and "det_id=0" in _text(at)
    keys = {w.key for w in at.multiselect}
    assert "f_decision" not in keys                      # no router columns: no queue widgets


def test_m3_csv_defaults_to_review_and_unknown_queue(tmp_path):
    at = _open(_m3_csv(tmp_path))
    assert at.multiselect(key="f_decision").value == ["REVIEW", "UNKNOWN_LOGO"]
    assert "Detection <b>1</b> / 2" in _text(at)


def test_queue_presets_filter(tmp_path):
    at = _open(_m3_csv(tmp_path))
    at.button(key="preset_unknown").click().run()
    assert at.multiselect(key="f_decision").value == ["UNKNOWN_LOGO"]
    assert "det_id=1" in _text(at) and "/ 1" in _text(at)
    at.button(key="preset_review").click().run()
    assert at.multiselect(key="f_decision").value == ["REVIEW"]


def test_router_and_rule5_panels_on_a_channel_brand_row(tmp_path):
    at = _open(_m3_csv(tmp_path))
    text = _text(at)
    assert "channel_brand_outside_watermark_region" in text and "Pre-empted rules:** 16" in text
    assert "Rule 5 evidence" in text
    panels = [d.value for d in at.dataframe if "watermark_overlap" in d.value.columns]
    assert len(panels) == 1
    assert {"ocr_text", "ocr_fuzzy_score", "faiss_top1", "brand_from_audio", "watermark_overlap_ratio"} <= \
        set(panels[0].columns)


def _fake_db_factory(tmp_path):
    import faiss

    from models.faiss_db import LogoDatabaseNew

    def make(index_path, metadata_path, batch_size=8, device="cpu"):
        db = object.__new__(LogoDatabaseNew)
        db.index_path, db.metadata_path, db.device, db.batch_size = index_path, metadata_path, "cpu", 4
        db.index, db.metadata = faiss.IndexFlatL2(4), []
        db._embed_batch = lambda imgs: np.ones((len(imgs), 4), dtype="float32") / 2.0
        return db
    return make


def test_accept_into_faiss_queues_then_commits_with_provenance(tmp_path):
    csv = _m3_csv(tmp_path)
    faiss_dir = tmp_path / "FAISS"
    faiss_dir.mkdir()
    with patch("models.faiss_db.LogoDatabaseNew", side_effect=_fake_db_factory(tmp_path)):
        at = _open(csv)
        at.text_input(key="faiss_dir").set_value(str(faiss_dir)).run()
        assert at.text_input(key="enrol_brand_0").value == "rai sport"
        at.button(key="enrol_btn_0").click().run()
        assert len(at.session_state["pending_enrol"]) == 1
        at.button(key="commit_enrol").click().run()
        assert not at.exception, at.exception
    recs = [json.loads(ln) for ln in (faiss_dir / "enrolments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 1
    r = recs[0]
    assert (r["origin"], r["video_id"], r["run_id"], r["det_id"], r["brand"]) == (
        "review_ui.accept", "video_1", "r1", 0, "rai sport")
    df = pd.read_csv(csv)
    assert bool(df.loc[df["det_id"] == 0, "enrolled_to_faiss"].iloc[0]) is True


def test_accept_is_offered_only_on_review_and_unknown_rows(tmp_path):
    at = _open(_m3_csv(tmp_path))
    at.multiselect(key="f_decision").set_value(["ACCEPT"]).run()
    assert not [b for b in at.button if (b.key or "").startswith("enrol_btn_")]
    at.checkbox(key="enrol_accept_rows").check().run()      # the [ASSUMPTION] toggle for ACCEPT rows
    assert [b for b in at.button if (b.key or "").startswith("enrol_btn_")]

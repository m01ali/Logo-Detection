"""MM7 (spec docs/13): rows.csv (T56), annotated frames (T58), the annotated
video (T59), the harness (T60), and the FAISS enrolment API behind T75 (with the
T55 provenance records)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import make_ctx  # noqa: E402
from test_mm1_output_stage import _run_upto_router  # noqa: E402

from logo_pipeline.annotate import draw_frame, frame_holds, label_text, wrap, write_video
from logo_pipeline.config import OutputDrawCfg
from logo_pipeline.schema import CSV_COLUMNS, Row
from logo_pipeline.stages.output import OutputStage, csv_path_for

REPO = Path(__file__).resolve().parents[1]


def _det(i, frame, box):
    return {"raw_det_id": i, "frame_idx": frame, "source": "gdino", "box_xyxy": box, "score": 0.9, "label": "a logo",
            "class_conf": None, "frame_width": 200, "frame_height": 100}


@pytest.fixture
def output_run(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=3, frame_size=(200, 100))
    _run_upto_router(ctx, [_det(0, 0, [10, 10, 60, 60]), _det(1, 0, [100, 20, 150, 70]), _det(2, 10, [5, 5, 40, 40])])
    OutputStage().run(ctx)          # frame 20 has no detection
    return ctx


# -- T56 ----------------------------------------------------------------------------

def test_rows_csv_columns_order_lists_and_relative_paths(output_run):
    df = pd.read_csv(output_run.path("rows.csv"))
    assert list(df.columns) == CSV_COLUMNS
    assert json.loads(df.loc[0, "box_xyxy"]) == [10, 10, 60, 60]
    for col in ("crop_path", "frame_path", "annotated_frame_path"):
        p = df.loc[0, col]
        assert "\\" not in p and not Path(p).is_absolute()
        assert (output_run.run_dir / p).exists()
    assert df["router_decision"].notna().all()


def test_csv_paths_absolute_and_relative_to_a_subfolder(output_run):
    sub = output_run.run_dir / "out"
    assert csv_path_for(output_run, "crops/x.jpg", sub, "relative") == "../crops/x.jpg"
    assert Path(csv_path_for(output_run, "crops/x.jpg", sub, "absolute")).is_absolute()


# -- T58 ----------------------------------------------------------------------------

def test_every_sampled_frame_is_annotated_including_empty_ones(output_run):
    sampled = output_run.sampled_frames()
    files = sorted(p.name for p in output_run.path("annotated_frames").glob("*.jpg"))
    assert files == [f"frame_{f['frame_idx']:06d}.jpg" for f in sampled]
    empty = Image.open(output_run.path("annotated_frames/frame_000020.jpg"))
    assert empty.size[1] > 100                       # the footer strip is appended below the frame


def test_label_text_and_wrapping():
    row = Row(det_id=7, canonical_brand="verisure", router_decision="REVIEW", reason_code="partial_logo_default_review",
              is_logo="partial_logo", is_truncated=True)
    t = label_text(row, OutputDrawCfg())
    assert t == "verisure · REVIEW · partial_logo_default_review · id=7 [cut off] (partial)"
    img = Image.new("RGB", (120, 60))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    from logo_pipeline.annotate import _font
    font = _font(12)
    lines = wrap(d, t, font, 110)
    assert len(lines) > 1
    # a line may overflow only when it is one unbreakable word (the "·" separator stays with the word after it)
    assert all(d.textlength(ln, font=font) <= 110 or ln.replace("· ", "·").count(" ") == 0 for ln in lines)


def test_draw_frame_colours_deduped_grey_and_draws_footer():
    frame = Image.new("RGB", (960, 560), (0, 0, 0))
    rows = [Row(det_id=0, selected_bbox=[100, 200, 200, 300], loose_bbox=[90, 190, 220, 320], router_decision="ACCEPT",
                reason_code="clear_logo_with_strong_ocr_match", canonical_brand="hp"),
            Row(det_id=1, selected_bbox=[600, 200, 700, 300], loose_bbox=[600, 200, 700, 300], router_decision="DEDUPED")]
    out = draw_frame(frame, rows, {"frame_idx": 3, "timecode_s": 1.5, "shot_id": 2}, "v", [[0.9, 0.0, 1.0, 0.1]],
                     OutputDrawCfg())
    px = np.asarray(out)
    assert out.size == (960, 560 + 560 // 40 + 12)
    assert (px[250, 100] == [0, 170, 0]).all()          # ACCEPT outline, left edge of the selected box
    assert (px[250, 600] == [150, 150, 150]).all()      # DEDUPED: thin grey
    assert (px[250, 602] == [0, 0, 0]).all()            # ...one pixel wide
    magenta = np.all(px == [255, 0, 255], axis=-1)
    region = magenta[:60, 860:]
    assert region.any() and not region.all(axis=1).any()   # watermark region drawn, and dotted (no solid edge)
    assert px[560:, :, :].max() > 200                   # footer text drawn


# -- T59 ----------------------------------------------------------------------------

def test_frame_holds_per_mode():
    sampled = [{"timecode_s": t, "shot_id": s} for t, s in ((0.0, 0), (2.0, 0), (4.0, 1))]
    assert frame_holds(sampled, "interval", 2.0, 4.0, None) == [2.0, 2.0, 2.0]
    assert frame_holds(sampled, "segment", 2.0, 4.0, None)[-1] == 4.0
    assert frame_holds(sampled, "interval", 2.0, 4.0, 1.0)[-1] == 1.0          # frames mode step
    shots = [{"shot_id": 0, "start_s": 0.0, "end_s": 3.5}, {"shot_id": 1, "start_s": 3.5, "end_s": 9.0}]
    assert frame_holds(sampled, "per_shot", 2.0, 4.0, None, shots) == [3.5, 3.5, 5.5]


def test_video_duration_matches_frames_times_interval(tmp_path):
    import cv2
    paths = []
    for i in range(5):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (101, 51), (40 * i, 0, 0)).save(p)     # odd size: padded to even for H.264
        paths.append(p)
    info = write_video(paths, [2.0] * 5, tmp_path / "a.mp4")
    assert abs(info["duration_s"] - 10.0) <= 1.0
    cap = cv2.VideoCapture(str(tmp_path / "a.mp4"))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(n / fps - 10.0) <= 1.0
    assert not (tmp_path / "a.mp4v.mp4").exists()


def test_output_stage_writes_the_video(output_run):
    status = OutputStage().outputs_exist(output_run)
    assert status and output_run.path("annotated.mp4").stat().st_size > 0


def test_two_cpu_replays_give_identical_rows_csv(output_run):
    first = output_run.path("rows.csv").read_bytes()
    OutputStage().run(output_run)
    assert output_run.path("rows.csv").read_bytes() == first


# -- T60 ----------------------------------------------------------------------------

def test_harness_report_renders(output_run):
    from logo_pipeline.harness import write_report
    path = write_report(output_run)
    text = path.read_text()
    assert "T18 merge counters (video)" in text and "Decision distribution" in text
    assert output_run.path("reports/merge_distributions.md").exists()


# -- T55 / T75 enrolment ----------------------------------------------------------------

def _real_db(tmp_path, dim=8):
    """A real LogoDatabaseNew without its models: add_logos/save are the production methods."""
    import faiss

    from models.faiss_db import LogoDatabaseNew
    db = object.__new__(LogoDatabaseNew)
    db.index_path, db.metadata_path = str(tmp_path / "logo_index.faiss"), str(tmp_path / "metadata.json")
    db.device, db.batch_size = "cpu", 4
    db.index, db.metadata = faiss.IndexFlatL2(dim), ["legacy a", "legacy b"]
    rng = np.random.default_rng(0)
    db.index.add(rng.normal(size=(2, dim)).astype("float32"))

    def embed(images):
        v = np.stack([np.full(dim, float(np.asarray(im).mean()) + i, dtype="float32") for i, im in enumerate(images)])
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    db._embed_batch = embed
    return db


def _rec(det_id=0, origin="review_ui.accept"):
    from logo_pipeline.faiss_store import EnrolmentRecord
    return EnrolmentRecord(brand="verisure", source_id=f"v/r/det_{det_id}", enrolled_at="t", origin=origin,
                           video_id="v", run_id="r", det_id=det_id)


def test_enrolment_record_round_trip():
    from logo_pipeline.faiss_store import EnrolmentRecord
    r = _rec(3)
    assert EnrolmentRecord.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_enrol_with_augmentations_records_augmented_of_and_flushes_on_save(tmp_path):
    import albumentations as A

    from logo_pipeline.faiss_store import enrol, enrolments_path, read_enrolments
    db = _real_db(tmp_path)
    img = Image.new("RGB", (32, 32), (120, 30, 30))
    idx = enrol(db, [img], [_rec()], augmentations=A.Compose([A.HorizontalFlip(p=1.0)]), num_augments=2)
    assert idx == [2] and db.index.ntotal == 5
    assert not enrolments_path(db).exists()            # nothing is written before the index is saved
    db.save()                                           # the production save calls the hook
    recs = read_enrolments(enrolments_path(db))
    assert sorted(recs) == [2, 3, 4]
    assert recs[2].augmented_of is None and recs[3].augmented_of == 2 and recs[4].augmented_of == 2
    assert all(r.origin == "review_ui.accept" for r in recs.values())
    assert json.loads(Path(db.metadata_path).read_text())[2:] == ["verisure"] * 3


def test_enrol_marks_indices_unresolved_when_a_batch_fails(tmp_path):
    from logo_pipeline.faiss_store import enrol
    db = _real_db(tmp_path)
    db._embed_batch = lambda images: np.array([])        # production behaviour on an embedding error
    with pytest.warns(UserWarning):
        idx = enrol(db, [Image.new("RGB", (8, 8))], [_rec()])
    assert idx == [None] and db._pending_enrolments[0].vector_index is None


def test_enrol_rejects_unknown_origin(tmp_path):
    from logo_pipeline.faiss_store import enrol
    with pytest.raises(ValueError):
        enrol(_real_db(tmp_path), [Image.new("RGB", (8, 8))], [_rec(origin="somewhere")])


def test_legacy_adapter():
    from logo_pipeline.faiss_store import record_for
    r = record_for(1, ["a", "b"], {})
    assert (r.origin, r.brand, r.vector_index) == ("legacy", "b", 1)


def test_add_logos_is_called_only_inside_faiss_store():
    """T55: the FAISS manager and the review UI write through enrol(); nothing else calls add_logos."""
    files = [REPO / "app" / "streamlit_faiss_app.py", REPO / "app" / "review_ui.py",
             *(REPO / "logo_pipeline").rglob("*.py")]
    offenders = [str(p.relative_to(REPO)) for p in files if re.search(r"\.add_logos\(", p.read_text(encoding="utf-8"))
                 and p.name != "enrol.py"]
    assert offenders == []
    app = (REPO / "app" / "streamlit_faiss_app.py").read_text(encoding="utf-8")
    origins = set(re.findall(r'"(faiss_manager\.[a-z_]+)"', app))
    assert len(origins) == 8


def test_commit_enrolments_from_the_review_queue(tmp_path):
    from logo_pipeline.faiss_store import enrolments_path, read_enrolments
    from logo_pipeline.review_support import commit_enrolments
    db = _real_db(tmp_path)
    crop = tmp_path / "crop.jpg"
    Image.new("RGB", (16, 16), (200, 0, 0)).save(crop)
    row = {"video_id": "video_1", "run_id": "r1", "det_id": 42, "frame_idx": 300, "crop_path": "crop.jpg",
           "label": "TP", "canonical_brand": "verisure", "router_decision": "REVIEW"}
    before = db.index.ntotal
    commit_enrolments(db, [{"row": row, "brand": "verisure"}], lambda p: tmp_path / p)
    assert db.index.ntotal == before + 1
    rec = read_enrolments(enrolments_path(db))[before]
    assert (rec.video_id, rec.run_id, rec.det_id, rec.source_id, rec.review_label) == (
        "video_1", "r1", 42, "video_1/r1/det_42", "TP")

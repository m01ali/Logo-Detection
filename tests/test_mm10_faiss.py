"""MM10 (spec docs/16): Roboflow box denormalisation, the T51 self-match
diagnostic and T53 exclusions, and the T54 re-benchmark — on a 20-crop subset
with a tiny in-memory index (random vectors), no CLIP needed."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import faiss_rebenchmark as rb  # noqa: E402
import faiss_selfmatch as sm  # noqa: E402

from logo_pipeline.brands import BrandEntry, build_brand_list  # noqa: E402

LABELS = Path("~/Downloads/logo2.yolov11/train/labels").expanduser()
NAMES = Path("~/Downloads/logo2.yolov11/data.yaml").expanduser()


def test_denormalise():
    assert sm.denormalise(0.5, 0.5, 0.2, 0.4, 1000, 500) == (400.0, 150.0, 600.0, 350.0)


def test_frame_stem():
    assert sm.frame_stem(Path("14_frame_0001_png.rf.kGVj8ZmRiu1wbA74uzzu.txt")) == "14_frame_0001"
    assert sm.frame_stem(Path("README.txt")) is None


@pytest.mark.skipif(not (LABELS.is_dir() and NAMES.exists() and Path("fixed-images").is_dir()),
                    reason="Roboflow export or fixed-images not present")
def test_video_14_boxes_denormalise_to_the_expected_list():
    names = sm.load_names(NAMES)
    boxes = [b for b in sm.iter_gold_boxes(LABELS, Path("fixed-images"), names, {"14"})
             if b.gold_id.split("#")[0] in ("14_frame_0001", "14_frame_0002")]
    assert [(b.gold_id, b.class_name) for b in boxes] == [("14_frame_0001#0", names[236]),
                                                         ("14_frame_0002#0", names[236])]
    for b in boxes:                            # 1866 x 1012 frames
        assert b.box == pytest.approx((101.21, 42.79, 172.03, 115.79), abs=0.05)


def _gold(tmp_path, n=20):
    """``n`` synthetic gold boxes over two frames, alternating two classes, plus their crops."""
    frames = []
    for f in range(2):
        p = tmp_path / f"9_frame_{f + 1:04d}.png"
        Image.new("RGB", (400, 200), (10 * f, 0, 0)).save(p)
        frames.append(p)
    out = []
    for i in range(n):
        x = (i % 10) * 38
        out.append(sm.GoldBox(f"9_frame_{i % 2 + 1:04d}#{i}", "9", frames[i % 2],
                              "verisure" if i % 2 else "zzz unknown brand", (x, 10, x + 30, 60), 400, 200))
    out.append(sm.GoldBox("9_frame_0001#tiny", "9", frames[0], "verisure", (0, 0, 8, 8), 400, 200))
    return out


class _Embedder:
    """Deterministic unit vectors per crop; crop k returns exactly index vector k for k < n_self."""

    def __init__(self, base: np.ndarray, n_self: int):
        self.base, self.n_self, self.k = base, n_self, 0

    def __call__(self, crops):
        out = []
        for _ in crops:
            if self.k < self.n_self:
                v = self.base[self.k]
            else:
                v = np.random.default_rng(1000 + self.k).normal(size=self.base.shape[1])
                v = v / np.linalg.norm(v)
            out.append(v.astype("float32"))
            self.k += 1
        return np.stack(out)


def _index(dim=16, n=30):
    import faiss
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = faiss.IndexFlatL2(dim)
    idx.add(v)
    return idx, v, ["Verisure" if i % 3 == 0 else f"brand {i}" for i in range(n)]


def test_selfmatch_reports_counts_exclusions_and_the_t52_statement(tmp_path):
    index, vecs, metadata = _index()
    rows = sm.run(_gold(tmp_path), _Embedder(vecs, n_self=3), index, metadata, tmp_path / "rep", batch=8)
    assert len(rows) == 20                                         # the 8 px box was skipped
    with open(tmp_path / "rep" / "faiss_selfmatch_excluded.csv") as fh:
        excluded = list(csv.DictReader(fh))
    assert [r["gold_id"] for r in excluded] == [rows[i]["gold_id"] for i in range(3)]
    text = (tmp_path / "rep" / "faiss_selfmatch.md").read_text()
    assert "Self-match (raw top-1 similarity ≥ 0.999): **3**" in text and "skipped" in text
    assert "3 crops self-match" in text and "a rebuild is not proposed" in text
    assert np.load(tmp_path / "rep" / "faiss_gold_embeddings.npy").shape == (20, 16)


def test_selfmatch_without_matches_says_the_benchmark_proceeds(tmp_path):
    index, vecs, metadata = _index()
    sm.run(_gold(tmp_path), _Embedder(vecs, n_self=0), index, metadata, tmp_path / "rep")
    assert "No crop self-matches; the benchmark proceeds on the existing index." in \
        (tmp_path / "rep" / "faiss_selfmatch.md").read_text()


def test_selfmatch_refuses_misaligned_embeddings(tmp_path):
    index, vecs, metadata = _index()
    with pytest.raises(RuntimeError):
        sm.run(_gold(tmp_path), lambda crops: np.array([]), index, metadata, tmp_path / "rep")


BRANDS = build_brand_list("2026-01-01", [BrandEntry("verisure", ("very sure",)), BrandEntry("rai", (), True)])


def _row(cls, top, sims):
    return {"class_name": cls, "top_brands": top, "top_sims": sims}


def test_rebenchmark_populations_accuracy_and_channel_variant():
    rows = [_row("verisure", ["Verisure", "x"], [0.98, 0.80]),      # present, correct
            _row("very sure", ["x", "Verisure"], [0.95, 0.90]),     # present via alias, correct at k=3 only
            _row("rai", ["RAI", "y"], [0.99, 0.70]),                # present, channel class
            _row("nobody", ["x", "y"], [0.85, 0.80])]               # absent
    keys = {"verisure", "rai", "x", "y"}
    present, absent = rb.populations(rows, keys, BRANDS)
    assert len(present) == 3 and len(absent) == 1
    e = rb.evaluate(present, absent, BRANDS)
    assert e["accuracy"][1] == pytest.approx(2 / 3) and e["accuracy"][3] == 1.0
    assert e["rule7_n"] == 2 and e["rule7_precision"] == 1.0         # verisure 0.98/0.18 and rai 0.99/0.29
    assert e["absent_top1"] == [0.85]
    no_ch = rb.evaluate(*rb.populations([r for r in rows if r["class_name"] != "rai"], keys, BRANDS), BRANDS)
    assert no_ch["n_present"] == 2
    text = rb.render(e, no_ch, excluded=1, n_brands=2)
    assert "Without channel classes" in text and "Rule 7 remains disabled" in text


def test_rebenchmark_states_the_kill_criterion_when_nothing_separates():
    rows = [_row("verisure", ["x"] if i % 2 else ["Verisure"], [0.9]) for i in range(30)]
    e = rb.evaluate(*rb.populations(rows, {"verisure", "x"}, BRANDS), BRANDS)
    assert e["separation"] is None
    assert "Kill criterion" in rb.render(e, e, 0, 1)

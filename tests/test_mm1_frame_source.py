"""21_test_plan.md — MM1: frame sources. Folder source orders ordinals
numerically and reports fps_is_assumed; video source on video_3-trimmed
returns 340 frames at 30 fps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from logo_pipeline.frame_source import FolderFrameSource, VideoFrameSource, ordinal_of


def _write_png(path: Path, size=(8, 8)):
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def test_ordinal_of_parses_suffix():
    assert ordinal_of("14_frame_0007.png") == 7
    assert ordinal_of("9_frame_0010.png") == 10
    assert ordinal_of("not_a_match.png") is None


def test_folder_source_orders_ordinals_numerically(tmp_path: Path):
    # 9_frame_0010 must sort after 9_frame_0009 (numeric, not lexicographic:
    # lexicographic would already get this right for zero-padded names, so
    # also check an unpadded pair that would break under naive string sort).
    for n in (9, 10, 100, 2):
        _write_png(tmp_path / f"9_frame_{n:04d}.png")
    src = FolderFrameSource(tmp_path, video_id="9", assumed_fps=1.0)
    assert src.indices() == [2, 9, 10, 100]
    assert src.fps_is_assumed is True
    assert src.frame_count == 4


def test_folder_source_duplicate_ordinal_rejected(tmp_path: Path):
    # Both parse to ordinal 1 (int("0001") == int("00001")): two distinct
    # filenames genuinely colliding on the same frame index.
    _write_png(tmp_path / "9_frame_0001.png")
    _write_png(tmp_path / "9_frame_00001.png")
    with pytest.raises(RuntimeError, match="duplicate ordinal"):
        FolderFrameSource(tmp_path, video_id="9", assumed_fps=1.0)


def test_folder_source_neighbour_and_timecode(tmp_path: Path):
    for n in (1, 2, 3, 4, 5):
        _write_png(tmp_path / f"5_frame_{n:04d}.png")
    src = FolderFrameSource(tmp_path, video_id="5", assumed_fps=1.0)
    assert src.neighbour(3, +1) == 4
    assert src.neighbour(3, -1) == 2
    assert src.neighbour(1, -1) is None    # past the start
    assert src.neighbour(5, +1) is None    # past the end
    assert src.timecode(1) == 0.0          # ordinals start at 1 (02 §3.1)
    assert src.timecode(3) == 2.0


def test_folder_source_on_real_fixed_images_video_14(repo_root: Path):
    d = repo_root / "fixed-images"
    if not d.is_dir():
        pytest.skip("fixed-images/ not present on this machine")
    src = FolderFrameSource(d, video_id="14", assumed_fps=1.0)
    assert src.frame_count == 46
    assert src.fps_is_assumed is True
    frame = src.read(src.indices()[0])
    assert frame is not None


def test_video_source_on_video_3_trimmed(repo_root: Path):
    path = repo_root / "video_3-trimmed.mp4"
    if not path.exists():
        pytest.skip("video_3-trimmed.mp4 not present on this machine")
    src = VideoFrameSource(path)
    assert src.frame_count == 340
    assert abs(src.fps - 30.0) < 0.01
    assert src.fps_is_assumed is False
    assert src.width == 960 and src.height == 560
    frame = src.read(0)
    assert frame is not None
    assert frame.shape[:2] == (560, 960)

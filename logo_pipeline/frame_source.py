"""Frame sources for the two ingest modes (03 §5).

``VideoFrameSource`` wraps OpenCV; ``FolderFrameSource`` reads a
``fixed-images``-style folder of ``{video_id}_frame_{ordinal}.png`` files in
numeric ordinal order with an assumed frame rate. Both expose the same
interface so the stages never branch on the mode.
"""

from __future__ import annotations

import abc
import re
from pathlib import Path

import cv2
import numpy as np

_ORDINAL_RE = re.compile(r"_frame_(\d+)\.[A-Za-z0-9]+$")


class FrameSource(abc.ABC):
    mode: str
    fps: float
    fps_is_assumed: bool
    frame_count: int
    width: int
    height: int
    source_path: str

    @abc.abstractmethod
    def indices(self) -> list[int]:
        """Every frame index of the source in order (video: 0..N-1; folder: ordinals)."""

    @abc.abstractmethod
    def read(self, frame_idx: int) -> np.ndarray | None:
        """BGR array for ``frame_idx`` or ``None`` when unreadable / out of range."""

    def timecode(self, frame_idx: int) -> float:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "source": self.source_path,
            "fps": self.fps,
            "fps_is_assumed": self.fps_is_assumed,
            "frame_count": self.frame_count,
            "frame_width": self.width,
            "frame_height": self.height,
        }


class VideoFrameSource(FrameSource):
    mode = "video"

    def __init__(self, path: str | Path):
        self.source_path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)   # M2 fallback
        self.fps_is_assumed = False
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_pos = 0

    def indices(self) -> list[int]:
        return list(range(self.frame_count))

    def read(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        if frame_idx != self._next_pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self._cap.read()
        self._next_pos = frame_idx + 1 if ok else -1
        return bgr if ok else None

    def timecode(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def close(self) -> None:
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None


def ordinal_of(path: str | Path) -> int | None:
    m = _ORDINAL_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


class FolderFrameSource(FrameSource):
    mode = "frames"

    def __init__(self, directory: str | Path, video_id: str, pattern: str = "{video_id}_frame_*.png",
                 assumed_fps: float = 1.0):
        self.directory = Path(directory)
        self.video_id = str(video_id)
        self.pattern = pattern
        self.source_path = str(self.directory / pattern.format(video_id=self.video_id))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        files: dict[int, Path] = {}
        for p in self.directory.glob(pattern.format(video_id=self.video_id)):
            o = ordinal_of(p)
            if o is None:
                continue
            if o in files:
                raise RuntimeError(f"duplicate ordinal {o} for video {self.video_id}: {files[o]} and {p}")
            files[o] = p
        if not files:
            raise RuntimeError(f"no frames matching {self.source_path}")
        self._files = files
        self._ordinals = sorted(files)          # numeric order: 9_frame_0010 after 9_frame_0009
        self.fps = float(assumed_fps)
        self.fps_is_assumed = True
        self.frame_count = len(self._ordinals)
        first = cv2.imread(str(files[self._ordinals[0]]))
        if first is None:
            raise RuntimeError(f"cannot read {files[self._ordinals[0]]}")
        self.height, self.width = first.shape[:2]

    def indices(self) -> list[int]:
        return list(self._ordinals)

    def path_of(self, ordinal: int) -> Path | None:
        return self._files.get(ordinal)

    def position_of(self, ordinal: int) -> int:
        return self._ordinals.index(ordinal)

    def read(self, frame_idx: int) -> np.ndarray | None:
        p = self._files.get(frame_idx)
        if p is None:
            return None
        return cv2.imread(str(p))

    def timecode(self, frame_idx: int) -> float:
        # 02 §3.1: ordinals start at 1; timecode = (ordinal − 1) / assumed_fps
        return (frame_idx - 1) / self.fps

    def neighbour(self, frame_idx: int, offset: int) -> int | None:
        """Ordinal ``offset`` positions away in the folder, or ``None`` past the ends."""
        try:
            pos = self._ordinals.index(frame_idx) + offset
        except ValueError:
            return None
        if 0 <= pos < len(self._ordinals):
            return self._ordinals[pos]
        return None


def open_frame_source(cfg, video, repo_root: Path | None = None) -> FrameSource:
    """Build the source described by the resolved config (video mode or frames mode)."""
    root = Path(repo_root) if repo_root else Path.cwd()
    if cfg.ingest.mode == "frames":
        d = Path(cfg.ingest.frames.dir)
        if not d.is_absolute():
            d = root / d
        return FolderFrameSource(d, video.video_id, cfg.ingest.frames.pattern, cfg.ingest.frames.assumed_fps)
    src = Path(video.source)
    if not src.is_absolute():
        src = root / src
    return VideoFrameSource(src)

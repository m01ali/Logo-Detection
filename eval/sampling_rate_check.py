#!/usr/bin/env python3
"""T66 route three (``spec docs/17``): confirm the rate at which ``fixed-images``
frames were extracted from a source video, once a source video arrives.

For ``--samples`` ordinals ``N`` spread over the folder, the video frame at
``t = (N − 1) / rate`` is compared with ``fixed-images/{v}_frame_{N}.png`` by
perceptual hash (DCT pHash, Hamming distance ≤ ``--max-distance``). The
believed rate passes when at least 90 % of samples match; otherwise the
candidate rates are swept. Writes ``reports/sampling_rate_{v}.md`` with the
confirmed rate (or none). Route-three box matching does not wait on this; the
rate only changes ``timecode_s``.

Usage::

    python eval/sampling_rate_check.py --video path/to/14.mp4 --video-id 14 [--frames-dir fixed-images]
        [--rate 1.0] [--candidates 0.5,1,2,5,10,25,30] [--samples 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runio import md_table, reports_dir, write_csv  # noqa: E402

PASS_SHARE = 0.9


def phash(image) -> np.ndarray:
    """64-bit DCT perceptual hash of a BGR/RGB/grey array or PIL image."""
    import cv2
    a = np.asarray(image)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(a.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small)[:8, :8]
    return (low > np.median(low)).flatten()


def distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def check_rate(read_video_frame, frames: dict[int, np.ndarray], rate: float, max_distance: int) -> tuple[float, list]:
    """Share of sampled ordinals whose video frame at ``(N-1)/rate`` matches, with per-sample distances."""
    rows = []
    for n, img in sorted(frames.items()):
        vid = read_video_frame((n - 1) / rate)
        d = distance(phash(vid), phash(img)) if vid is not None else None
        rows.append((n, d))
    ok = sum(1 for _, d in rows if d is not None and d <= max_distance)
    return (ok / len(rows) if rows else 0.0), rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--frames-dir", default="fixed-images")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--candidates", default="0.5,1,2,3,5,10,12.5,15,24,25,29.97,30")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--max-distance", type=int, default=10)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)

    import cv2
    files = sorted(Path(args.frames_dir).glob(f"{args.video_id}_frame_*.png"))
    if not files:
        print(f"no frames for video {args.video_id} in {args.frames_dir}")
        return 1
    picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(args.samples, len(files)))]
    frames = {int(p.stem.split("_frame_")[1]): cv2.imread(str(p)) for p in picks}
    cap = cv2.VideoCapture(args.video)

    def read_at(t: float):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        return f if ok else None

    share, rows = check_rate(read_at, frames, args.rate, args.max_distance)
    tried = [(args.rate, share)]
    confirmed = args.rate if share >= PASS_SHARE else None
    if confirmed is None:
        for c in (float(x) for x in args.candidates.split(",")):
            if c == args.rate:
                continue
            s, _ = check_rate(read_at, frames, c, args.max_distance)
            tried.append((c, s))
            if s >= PASS_SHARE:
                confirmed = c
                break
    cap.release()
    lines = [f"# Sampling-rate check — video {args.video_id}", "",
             f"{len(frames)} ordinals compared by pHash (match: Hamming ≤ {args.max_distance}; pass: ≥ "
             f"{int(PASS_SHARE * 100)}% of samples).", ""]
    lines += md_table(["rate (fps)", "share matching"], [(r, f"{s:.2f}") for r, s in tried])
    lines += ["", "Believed-rate distances: " + ", ".join(f"{n}: {d}" for n, d in rows), ""]
    if confirmed is not None:
        lines.append(f"**Confirmed rate: {confirmed} fps.** Set `ingest.frames.assumed_fps: {confirmed}` in the "
                     f"per-video overrides and re-run `--from-stage sampling`; with `frame_ordinals` or "
                     "`process_all_frames` the same frames are sampled, so only `timecode_s` changes.")
    else:
        lines.append("**No candidate rate matched.** Timecodes stay on the assumed rate; box matching is unaffected.")
    out = reports_dir(Path("."), args.out)
    path = out / f"sampling_rate_{args.video_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(out / f"sampling_rate_{args.video_id}.csv", ["kind", "key", "value"],
              [("rate_share", r, round(s, 4)) for r, s in tried]
              + [("believed_rate_distance", n, "" if d is None else d) for n, d in rows]
              + [("confirmed_rate", "", "" if confirmed is None else confirmed)])
    print(f"{'confirmed ' + str(confirmed) if confirmed else 'no rate confirmed'}; wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

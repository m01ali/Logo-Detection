"""Annotated frames (T58) and the annotated video (T59), ``spec docs/13``.

Frames: every sampled frame gets an image, with or without detections.
Outline colour follows ``router_decision``; DEDUPED boxes are thin and grey;
``selected_bbox`` is solid and ``loose_bbox`` dashed where refinement moved
it; watermark regions are dotted; a footer strip carries
``video_id · frame · t · shot · n detections``. Font loading reuses
``generate_hints._load_font`` (the M2 style); the label text is the router's.

Video: built from the annotated frames only (drawing on unprocessed frames
would misrepresent recall). Each frame is held for its share of the source
time at 25 fps and the writer streams frame by frame; it is re-encoded to
H.264 by ffmpeg when one is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

DECISION_COLOURS = {
    "ACCEPT": (0, 170, 0),
    "REVIEW": (255, 150, 0),
    "REJECT": (220, 40, 40),
    "UNKNOWN_LOGO": (40, 110, 255),
    "DEDUPED": (150, 150, 150),
    None: (255, 255, 255),
}
DRAW_ORDER = {"DEDUPED": 0, "REJECT": 1, None: 2, "REVIEW": 3, "UNKNOWN_LOGO": 4, "ACCEPT": 5}
REGION_COLOUR = (255, 0, 255)
VIDEO_FPS = 25


def _font(size: int):
    try:
        from generate_hints import _load_font
        return _load_font(size)
    except Exception:  # pragma: no cover - generate_hints imports torch-free code only, but be safe
        from PIL import ImageFont
        return ImageFont.load_default()


def _broken_rect(draw: ImageDraw.ImageDraw, box, colour, width: int, dash: int, gap: int) -> None:
    x1, y1, x2, y2 = (float(c) for c in box)
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            continue
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            fa, fb = pos / length, end / length
            draw.line([(ax + (bx - ax) * fa, ay + (by - ay) * fa), (ax + (bx - ax) * fb, ay + (by - ay) * fb)],
                      fill=colour, width=width)
            pos += dash + gap


def label_text(row, draw_cfg) -> str:
    parts = []
    if draw_cfg.brand:
        parts.append(row.canonical_brand or row.brand or "—")
    if draw_cfg.decision:
        parts.append(row.router_decision or "unrouted")
    if draw_cfg.reason_code and row.reason_code:
        parts.append(row.reason_code)
    if draw_cfg.det_id:
        parts.append(f"id={row.det_id}")
    text = " · ".join(str(p) for p in parts)
    if row.is_truncated:
        text += " [cut off]"
    if row.is_logo == "partial_logo":
        text += " (partial)"
    return text or "?"


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Wraps at `` · `` separators first, then at spaces, so a label never runs off the frame."""
    words = text.replace(" · ", " · ").split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}" if cur else w
        if cur and draw.textlength(cand, font=font) > max_width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return [ln.replace(" ", " ") for ln in lines]


def draw_frame(frame: Image.Image, rows: list, frame_info: dict, video_id: str, regions: list, draw_cfg) -> Image.Image:
    """The annotated image for one sampled frame (``rows`` are ``schema.Row`` of that frame)."""
    w, h = frame.size
    font_size = max(12, h // 40)
    font = _font(font_size)
    footer_h = font_size + 12
    out = Image.new("RGB", (w, h + footer_h), (0, 0, 0))
    out.paste(frame.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)

    ordered = sorted(rows, key=lambda r: (DRAW_ORDER.get(r.router_decision, 2), r.det_id))
    for row in ordered:
        colour = DECISION_COLOURS.get(row.router_decision, DECISION_COLOURS[None])
        deduped = row.router_decision == "DEDUPED"
        selected = [float(c) for c in (row.selected_bbox or row.loose_bbox)]
        draw.rectangle(selected, outline=colour, width=1 if deduped else 3)
        if row.loose_bbox and list(row.loose_bbox) != list(row.selected_bbox or row.loose_bbox):
            _broken_rect(draw, row.loose_bbox, colour, 1 if deduped else 2, 8, 6)

    for r in regions:          # normalised [x1, y1, x2, y2]; over the boxes so a bug's own box cannot hide it
        _broken_rect(draw, [r[0] * w, r[1] * h, r[2] * w, r[3] * h], REGION_COLOUR, 2, 3, 4)

    for row in ordered:
        colour = DECISION_COLOURS.get(row.router_decision, DECISION_COLOURS[None])
        deduped = row.router_decision == "DEDUPED"
        selected = [float(c) for c in (row.selected_bbox or row.loose_bbox)]
        lines = wrap(draw, label_text(row, draw_cfg), font, max(40, w - 10))
        line_h = font_size + 4
        block_h = line_h * len(lines) + 4
        block_w = max(draw.textlength(ln, font=font) for ln in lines) + 8
        x0 = max(0, min(selected[0], w - block_w))
        y0 = selected[1] - block_h if selected[1] - block_h >= 0 else min(selected[3], h - block_h)
        draw.rectangle([x0, y0, x0 + block_w, y0 + block_h], fill=colour)
        for i, ln in enumerate(lines):
            draw.text((x0 + 4, y0 + 2 + i * line_h), ln, fill=(255, 255, 255) if not deduped else (30, 30, 30),
                      font=font)

    shot = frame_info.get("shot_id")
    footer = (f"{video_id} · frame {frame_info['frame_idx']} · t={float(frame_info.get('timecode_s', 0.0)):.1f}s · "
              f"shot {shot if shot is not None else '—'} · {len(rows)} detections")
    draw.text((6, h + 6), footer, fill=(255, 255, 255), font=font)
    return out


# ---------------------------------------------------------------------------
# T59 video
# ---------------------------------------------------------------------------

def frame_holds(sampled: list[dict], mode: str, interval_s: float, segment_s: float, step_s: float | None,
                shots: list[dict] | None = None) -> list[float]:
    """Seconds each sampled frame is held, so the video runs at the source's real time.

    ``per_shot`` holds each frame for its shot's duration; the other modes hold a frame until the
    next sampled timecode, the last one for the mode's nominal step.
    """
    if not sampled:
        return []
    if mode == "per_shot" and shots:
        dur = {s["shot_id"]: max(0.0, float(s["end_s"]) - float(s["start_s"])) for s in shots}
        return [dur.get(f.get("shot_id"), interval_s) or interval_s for f in sampled]
    nominal = step_s if step_s else (segment_s if mode == "segment" else interval_s)
    times = [float(f.get("timecode_s", 0.0)) for f in sampled]
    holds = [max(0.0, b - a) for a, b in zip(times, times[1:])] + [nominal]
    return [hd if hd > 0 else nominal for hd in holds]


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_video(frame_paths: list[Path], holds: list[float], out_path: Path, codec: str = "mp4v",
                fps: int = VIDEO_FPS, reencode: bool = True) -> dict:
    """Streams the frames to ``out_path``; returns ``{frames_written, duration_s, h264}``."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    hgt, wid = first.shape[:2]
    wid2, hgt2 = wid + wid % 2, hgt + hgt % 2                        # H.264 needs even dimensions
    raw_path = out_path.with_name(out_path.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*codec), fps, (wid2, hgt2))
    n = 0
    carry = 0.0
    try:
        for p, hold in zip(frame_paths, holds):
            img = cv2.imread(str(p))
            if img.shape[:2] != (hgt, wid):
                img = cv2.resize(img, (wid, hgt))
            if (wid2, hgt2) != (wid, hgt):
                img = cv2.copyMakeBorder(img, 0, hgt2 - hgt, 0, wid2 - wid, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            exact = hold * fps + carry                          # carry rounding so the total stays on time
            reps = max(1, int(round(exact)))
            carry = exact - reps
            for _ in range(reps):
                writer.write(img)
            n += reps
    finally:
        writer.release()

    h264 = False
    exe = ffmpeg_exe() if reencode else None
    if exe:
        res = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                             capture_output=True, text=True)
        h264 = res.returncode == 0 and out_path.exists()
    if h264:
        raw_path.unlink(missing_ok=True)
    else:
        raw_path.replace(out_path)
    return {"frames_written": n, "duration_s": round(n / fps, 2), "h264": h264}

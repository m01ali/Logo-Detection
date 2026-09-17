"""``stages/shots.py`` (T57, 07 MM1 Step 2): PySceneDetect scene list ->
``shots.json``.

Video mode uses ``AdaptiveDetector`` (the client-facing default,
``shots.detector: adaptive``); frames mode always forces ``ContentDetector``
since adaptive detection needs a continuous stream (03 §5.2, C10). Each
scene gets an ordinal ``shot_id`` starting at 0; this file is also what
``per_shot`` sampling reads.
"""

from __future__ import annotations

from .base import SHOTS_FILE, RunContext, Stage, StageResult


def _detect_scenes_video(path: str, detector_name: str, cfg) -> list[tuple[float, float, int, int]]:
    from scenedetect import ContentDetector, AdaptiveDetector, detect

    if detector_name == "content":
        detector = ContentDetector(threshold=cfg.shots.content_threshold,
                                    min_scene_len=cfg.shots.min_scene_len_frames)
    else:
        detector = AdaptiveDetector(adaptive_threshold=cfg.shots.adaptive_threshold,
                                     min_scene_len=cfg.shots.min_scene_len_frames)
    scene_list = detect(path, detector, show_progress=False)
    out = []
    for start_tc, end_tc in scene_list:
        out.append((start_tc.get_seconds(), end_tc.get_seconds(), start_tc.get_frames(), end_tc.get_frames()))
    return out


def _detect_scenes_frames(ordinals: list[int], fps: float, threshold: float, min_len: int, read_fn) -> list[dict]:
    """``ContentDetector`` driven frame-by-frame over an ordinal sequence
    that is not a continuous video (03 §5.2)."""
    import cv2

    from scenedetect import ContentDetector

    detector = ContentDetector(threshold=threshold, min_scene_len=min_len)
    cuts: list[int] = []  # positions (0-based, within `ordinals`) where a new scene starts
    for pos, ordinal in enumerate(ordinals):
        bgr = read_fn(ordinal)
        if bgr is None:
            continue
        cut_positions = detector.process_frame(pos, bgr)
        cuts.extend(cut_positions)
    try:
        cuts.extend(detector.post_process(len(ordinals) - 1))
    except Exception:
        pass
    cut_set = sorted(set(c for c in cuts if 0 < c < len(ordinals)))
    boundaries = [0] + cut_set + [len(ordinals)]
    scenes = []
    for i in range(len(boundaries) - 1):
        start_pos, end_pos = boundaries[i], boundaries[i + 1] - 1
        scenes.append({
            "shot_id": i,
            "start_frame": ordinals[start_pos], "end_frame": ordinals[end_pos],
            "start_s": (ordinals[start_pos] - 1) / fps, "end_s": (ordinals[end_pos] - 1) / fps,
        })
    return scenes


class ShotsStage(Stage):
    name = "shots"
    device = "cpu"
    outputs = [SHOTS_FILE]

    def run(self, ctx: RunContext) -> StageResult:
        cfg = ctx.cfg
        manifest = ctx.manifest
        detector_name = ctx.resolved.effective.get("shots.detector", cfg.shots.detector)

        if not cfg.shots.enabled:
            ctx.write_json(SHOTS_FILE, [])
            return StageResult(counts={"shots": 0, "enabled": False})

        if manifest["mode"] == "video":
            triples = _detect_scenes_video(manifest["source"], detector_name, cfg)
            scenes = [
                {"shot_id": i, "start_frame": sf, "end_frame": ef, "start_s": ss, "end_s": es}
                for i, (ss, es, sf, ef) in enumerate(triples)
            ]
            if not scenes:  # PySceneDetect returns [] for a video with no cuts detected
                scenes = [{"shot_id": 0, "start_frame": 0, "end_frame": manifest["frame_count"] - 1,
                          "start_s": 0.0, "end_s": manifest["frame_count"] / manifest["fps"]}]
        else:
            source = ctx.open_source()
            if source is None:
                raise FileNotFoundError("frames-mode shot detection needs the frame folder")
            ordinals = source.indices()
            scenes = _detect_scenes_frames(ordinals, source.fps, cfg.shots.content_threshold,
                                            cfg.shots.min_scene_len_frames, source.read)
            source.close()

        ctx.write_json(SHOTS_FILE, scenes)
        ctx.update_manifest(shots_detector=detector_name)
        return StageResult(counts={"shots": len(scenes), "detector": detector_name})

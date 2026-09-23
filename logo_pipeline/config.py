"""The single YAML (T8): loader, strict schema, per-video overlay, config hash.

Every key in ``spec docs/04`` is a field of a dataclass below. Unknown keys
fail validation, enum-typed keys are checked against their allowed values,
and the resolved (defaults filled, overrides applied) configuration is what
gets hashed and written into ``run_manifest.json``.

Two keys are not in ``04`` and were added here because the specs use their
values without listing them: ``crops.enlarge_factor`` (the 1.5× enlarged
crop of ``05`` C) and ``crops.jpeg_quality``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

CONFIG_SCHEMA_VERSION = 1

DEFAULT_STAGES: tuple[str, ...] = (
    "ingest", "shots", "sampling", "detect_gdino", "detect_yolo", "merge", "refine", "ocr",
    "faiss_query", "audio", "vlm", "canonicalise", "watermark", "router_input", "router", "output",
)

# Keys that do not influence any output and are therefore excluded from the
# config hash (so a run directory copied between the Mac and the GPU machine,
# or re-run with a different log cadence, keeps its stage status valid).
HASH_EXCLUDED_KEYS: tuple[tuple[str, ...], ...] = (
    ("run", "run_id"),
    ("run", "output_root"),
    ("run", "log_every_n"),
    ("run", "device"),
)

GDINO_QUERIES_M2: tuple[str, ...] = (
    "a television network watermark",
    "a channel logo in the corner of the screen",
    "a digital graphic logo",
    "a sports broadcast logo",
    "a brand logo on a sports shirt",
    "a sponsor logo on athletic shorts",
    "a logo on a tennis racket",
    "a sportswear brand logo",
    "a sponsor logo on a sports uniform",
    "an advertising board sponsor logo",
    "a logo printed on a sports court surface",
    "a sponsor logo on a billiard table border",
    "a tournament logo on a wall",
    "a brand name printed on a banner",
    "a brand name on medicine packaging",
    "a company logo on a food box",
    "a brand logo on a frozen food bag",
    "a toy brand logo",
    "a corporate logo on a product label",
    "a logo on a security camera device",
    "a corporate trademark",
    "a small brand icon",
    "a company symbol",
    "a text-based brand logo",
    "a commercial brand mark",
    "an enterprise emblem",
    "a distinctive brand insignia",
    "a recognizable company badge",
    "a generic logo",
    "an organization's crest",
    "a commercial watermark",
    "a manufacturer's mark",
    "an official logo graphic",
    "a product",
    "a potential logo",
    "a company",
    "a brand",
)


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types, bad enum values or locked values."""


# ---------------------------------------------------------------------------
# Schema dataclasses (one per YAML section, in 04 order)
# ---------------------------------------------------------------------------


@dataclass
class RunCfg:
    output_root: str = "runs"
    run_id: str | None = None
    log_every_n: int = 10
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42
    stages: list[str] = field(default_factory=lambda: list(DEFAULT_STAGES))


@dataclass
class IngestFramesCfg:
    dir: str | None = None
    pattern: str = "{video_id}_frame_*.png"
    assumed_fps: float = 1.0


@dataclass
class IngestCfg:
    mode: Literal["video", "frames"] = "video"
    frames: IngestFramesCfg = field(default_factory=IngestFramesCfg)
    save_frame_jpeg_quality: int = 95


@dataclass
class ShotsCfg:
    enabled: bool = True
    detector: Literal["adaptive", "content"] = "adaptive"
    adaptive_threshold: float = 3.0
    content_threshold: float = 27.0
    min_scene_len_frames: int = 15


@dataclass
class PerShotCfg:
    position: Literal["middle", "first"] = "middle"


@dataclass
class SegmentCfg:
    duration_s: float = 4.0


@dataclass
class SamplingCfg:
    mode: Literal["interval", "per_shot", "segment"] = "interval"
    interval_s: float = 2.0
    per_shot: PerShotCfg = field(default_factory=PerShotCfg)
    segment: SegmentCfg = field(default_factory=SegmentCfg)
    temporal_window: int = 5
    temporal_offsets_for_vlm: list[int] = field(default_factory=lambda: [-2, -1, 1, 2])


@dataclass
class GdinoCfg:
    enabled: bool = True
    model_id: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.20
    text_threshold: float = 0.20
    queries: list[str] = field(default_factory=lambda: list(GDINO_QUERIES_M2))
    min_crop_px: int = 16


@dataclass
class YoloCfg:
    enabled: bool = True
    weights: str = "weights/best.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou_nms: float = 0.7
    class_agnostic: bool = True
    log_predicted_class: bool = True


@dataclass
class DetectorsCfg:
    gdino: GdinoCfg = field(default_factory=GdinoCfg)
    yolo: YoloCfg = field(default_factory=YoloCfg)


@dataclass
class SmallBoxCfg:
    enabled: bool = False
    pixel_area_cutoff: int = 400
    centre_distance_threshold: float = 0.5


@dataclass
class MergeCfg:
    enabled: bool = True
    iou_threshold: float = 0.70
    containment_threshold: float = 0.90
    area_ratio_threshold: float = 0.80
    min_merge_area: float = 0.0004
    representative: Literal["tighter", "higher_score"] = "tighter"
    small_box: SmallBoxCfg = field(default_factory=SmallBoxCfg)
    log_failed_pairs: bool = True
    legacy_filters_enabled: bool = False


@dataclass
class Heuristic2Cfg:
    enabled: bool = True
    ocr_min_confidence: float = 0.40
    brand_match_min_score: int = 80
    padding_ratio: float = 0.10
    min_area_ratio_vs_loose: float = 0.05
    max_area_ratio_vs_loose: float = 0.95
    min_side_px: int = 12
    max_aspect: float = 8.0


@dataclass
class Heuristic5Cfg:
    enabled: bool = True
    area_ratio: float = 0.45


@dataclass
class RefinementCfg:
    enabled: bool = True
    heuristic_1_provenance: bool = True
    heuristic_2_ocr_tightening: Heuristic2Cfg = field(default_factory=Heuristic2Cfg)
    heuristic_5_oversized_guard: Heuristic5Cfg = field(default_factory=Heuristic5Cfg)
    heuristics_3_4_enabled: bool = False


@dataclass
class OcrNormaliseCfg:
    lowercase: bool = True
    strip_non_alphanumeric: bool = True
    collapse_whitespace: bool = True


@dataclass
class OcrMatchingCfg:
    short_name_cutoff: int = 4
    exact_match_short: bool = True
    general_fuzzy_threshold: int = 90
    scorer: Literal["WRatio", "token_set_ratio", "smart_score"] = "WRatio"
    match_lists: list[str] = field(default_factory=lambda: ["canonical", "audio"])


@dataclass
class OcrCfg:
    languages: list[str] = field(default_factory=lambda: ["en", "it"])
    gpu: Literal["auto", True, False] = "auto"
    upscale_min_side: int = 80
    upscale_factor: int = 3
    fallback_to_enlarged_crop: bool = True
    normalise: OcrNormaliseCfg = field(default_factory=OcrNormaliseCfg)
    matching: OcrMatchingCfg = field(default_factory=OcrMatchingCfg)
    cheap_call_keep_boxes: bool = True


@dataclass
class FaissCfg:
    index_path: str = "FAISS/logo_index.faiss"
    metadata_path: str = "FAISS/metadata.json"
    embedder: Literal["clip", "siglip2"] = "clip"
    top_k: int = 5
    overfetch_factor: int = 3
    similarity_threshold: float = 0.0
    rule7_top1_threshold: float = 0.97
    rule7_score_gap: float = 0.08
    rule12_top1_threshold: float = 0.97
    rule14_top1_threshold: float = 0.90
    abstain_threshold: float | None = None


@dataclass
class AudioCfg:
    enabled: bool = True
    whisper_model_id: str = "openai/whisper-large-v3"
    task: Literal["transcribe", "translate"] = "transcribe"
    language: str | None = "italian"
    text_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    text_decoding: Literal["greedy"] = "greedy"
    brand_prompt: Literal["m2", "task2_italian"] = "task2_italian"


@dataclass
class ImageSignalCfg:
    include_in_vlm_prompt: bool = True


@dataclass
class TextSignalCfg:
    compute_and_log: bool = True
    include_in_vlm_prompt: bool = True


@dataclass
class SignalsCfg:
    tight_crop: ImageSignalCfg = field(default_factory=ImageSignalCfg)
    enlarged_crop: ImageSignalCfg = field(default_factory=ImageSignalCfg)
    full_frame: ImageSignalCfg = field(default_factory=ImageSignalCfg)
    temporal_context: ImageSignalCfg = field(default_factory=ImageSignalCfg)
    audio: TextSignalCfg = field(default_factory=TextSignalCfg)
    faiss: TextSignalCfg = field(default_factory=TextSignalCfg)
    ocr: TextSignalCfg = field(default_factory=TextSignalCfg)


@dataclass
class VlmScoringCfg:
    enabled: bool = False
    labels: list[str] = field(default_factory=lambda: ["logo", "partial_logo", "not_logo", "uncertain"])
    method: Literal["sequence_logprob", "first_token_softmax"] = "sequence_logprob"
    distribution_check_required: bool = True


@dataclass
class VlmCfg:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantisation: Literal["4bit", "8bit", "none"] = "4bit"
    max_pixels: int = 100352
    max_new_tokens: int = 256
    decoding: Literal["greedy"] = "greedy"
    brand_mode: Literal["open", "closed"] = "open"
    unknown_explicit_option: bool = True
    channel_watermark_line: bool = True
    hinted_pass: bool = True
    blind_pass: bool = True
    blind_pass_signals: list[str] = field(
        default_factory=lambda: ["tight_crop", "enlarged_crop", "full_frame", "temporal_context"])
    log_hint_composition: bool = True
    scoring: VlmScoringCfg = field(default_factory=VlmScoringCfg)


@dataclass
class LlmMergeCfg:
    enabled: bool = False
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class CanonicalisationCfg:
    alias_first: bool = True
    exact_then_fuzzy: bool = True
    fuzzy_threshold: int = 90
    short_name_cutoff: int = 4
    audio_brands_as_targets: bool = True
    llm_merge: LlmMergeCfg = field(default_factory=LlmMergeCfg)


@dataclass
class WatermarkCfg:
    margin: float = 0.02
    containment_threshold: float = 0.50
    regions_ref_required: bool = False


@dataclass
class RouterRulesCfg:
    rule_1: bool = True
    rule_1_fallback_logprob: bool = False
    rule_2: bool = False          # needs brand_type (enhanced VLM fields, not produced in M3 v1)
    rule_3: bool = False          # needs is_brand_identifier (same)
    rule_4: bool = True
    rule_5_review: bool = True
    rule_5_reject: bool = False
    rule_6: bool = True
    rule_7: bool = False
    rule_8: bool = False
    rule_9: bool = True
    rule_10: bool = True
    rule_11_review: bool = True
    rule_11_reject: bool = True
    rule_12: bool = True
    rule_13: bool = True
    rule_14: bool = True
    rule_15: bool = True
    rule_16: bool = True
    rule_17: bool = True
    rule_18: bool = True


@dataclass
class RouterThresholdsCfg:
    rule1_not_logo_confidence: float = 0.80
    rule1_logo_probability_max: int = 20
    rule4_area_ratio: float = 0.45
    rule5_ocr_fuzzy: int = 90
    rule5_reject_ocr_fuzzy: int = 70
    rule6_ocr_fuzzy_threshold: int = 95
    rule8_logo_probability: int = 95
    rule9_strong_ocr: int = 95
    rule11_ocr_fuzzy: int = 80
    rule12_ocr_fuzzy: int = 80
    rule12_ocr_fuzzy_second: int = 90
    rule13_ocr_fuzzy: int = 85
    rule16_ocr_fuzzy: int = 80


@dataclass
class RouterDedupCfg:
    dedup_iou_threshold: float = 0.50
    dedup_containment_threshold: float = 0.80
    dedup_area_ratio_threshold: float = 0.80
    post_vlm_min_merge_area_enabled: bool = False


@dataclass
class TrustWeightsCfg:
    """18 July §8 trust constants. The ``brand_type`` / ``is_brand_identifier`` terms only
    contribute when ``router.enhanced_vlm_fields_available`` is true."""
    vlm_logo: int = 3
    brand_type_brand: int = 2
    is_brand_identifier: int = 2
    ocr_strong: int = 4
    ocr_strong_threshold: int = 90
    brand_in_database: int = 1
    faiss_strong_agreement: int = 1
    small_box: int = 1
    small_box_area_ratio: float = 0.20
    tight_bbox: int = 1


@dataclass
class RiskWeightsCfg:
    """18 July §8 risk constants (same enhanced-field caveat as the trust weights)."""
    partial_logo: int = 3
    uncertain: int = 4
    truncated: int = 4
    non_brand_type: int = 6
    audio_weak_ocr: int = 4
    faiss_weak_ocr: int = 4
    weak_ocr_threshold: int = 80
    conflicting_ocr: int = 5
    faiss_conflict: int = 3
    oversized: int = 4
    oversized_area_ratio: float = 0.45
    channel_outside_watermark: int = 5


@dataclass
class ScoringLayerCfg:
    compute_trust_risk: bool = True
    trust: TrustWeightsCfg = field(default_factory=TrustWeightsCfg)
    risk: RiskWeightsCfg = field(default_factory=RiskWeightsCfg)


@dataclass
class BrandEvidenceCfg:
    compute: bool = True


@dataclass
class RouterCfg:
    enabled: bool = True
    not_logo_confidence_source: Literal["generated", "scored"] = "generated"   # Part E; flips after T32
    enhanced_vlm_fields_available: bool = False        # brand_type / is_brand_identifier (24 July; M2 prompt lacks them)
    rules: RouterRulesCfg = field(default_factory=RouterRulesCfg)
    thresholds: RouterThresholdsCfg = field(default_factory=RouterThresholdsCfg)
    dedup: RouterDedupCfg = field(default_factory=RouterDedupCfg)
    scoring_layer: ScoringLayerCfg = field(default_factory=ScoringLayerCfg)
    brand_evidence: BrandEvidenceCfg = field(default_factory=BrandEvidenceCfg)
    firing_log: bool = True


@dataclass
class OutputDrawCfg:
    brand: bool = True
    decision: bool = True
    reason_code: bool = True
    det_id: bool = True


@dataclass
class OutputCfg:
    csv_name: str = "rows.csv"
    csv_paths: Literal["relative", "absolute"] = "relative"
    annotated_frames: bool = True
    annotated_video: bool = True
    video_codec: str = "mp4v"
    draw: OutputDrawCfg = field(default_factory=OutputDrawCfg)


@dataclass
class CropsCfg:
    """Not in 04; added because 05 C needs the enlarged-crop factor."""
    enlarge_factor: float = 1.5
    jpeg_quality: int = 95


@dataclass
class BrandsCfg:
    list_path: str = "configs/brands/roboflow_2026-04-28.yaml"
    require_hash_match: bool = False


@dataclass
class EvaluationCfg:
    labels_path: str | None = None
    iou_match_threshold: float = 0.50
    roboflow_labels_dir: str | None = None
    channel_classes: list[str] = field(default_factory=lambda: [
        "rai", "rai fiction", "mediaset", "rete 4", "boing", "nove", "sport italia", "sky", "sky wifi",
        "discovery plus", "infinity plus", "paramount plus", "m6", "x factor", "tim"])


@dataclass
class PipelineCfg:
    version: int = CONFIG_SCHEMA_VERSION
    run: RunCfg = field(default_factory=RunCfg)
    ingest: IngestCfg = field(default_factory=IngestCfg)
    shots: ShotsCfg = field(default_factory=ShotsCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    detectors: DetectorsCfg = field(default_factory=DetectorsCfg)
    merge: MergeCfg = field(default_factory=MergeCfg)
    refinement: RefinementCfg = field(default_factory=RefinementCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    faiss: FaissCfg = field(default_factory=FaissCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    vlm: VlmCfg = field(default_factory=VlmCfg)
    canonicalisation: CanonicalisationCfg = field(default_factory=CanonicalisationCfg)
    watermark: WatermarkCfg = field(default_factory=WatermarkCfg)
    router: RouterCfg = field(default_factory=RouterCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    crops: CropsCfg = field(default_factory=CropsCfg)
    brands: BrandsCfg = field(default_factory=BrandsCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)


@dataclass
class VideoCfg:
    """``configs/videos/<video_id>.yaml`` (04 per-video file)."""
    video_id: str = ""
    source: str | None = None
    channel: str | None = None
    channel_brand: str | None = None
    watermark_regions_ref: str | None = None
    overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strict builder
# ---------------------------------------------------------------------------


def _type_name(t: Any) -> str:
    return getattr(t, "__name__", str(t))


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if annotation is Any:
        return value
    if is_dataclass(annotation) and isinstance(annotation, type):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {_type_name(type(value))}")
        return build_dataclass(annotation, value, path)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path}: {value!r} is not one of {list(args)}")
        return value
    if origin in (types.UnionType, typing.Union):
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed")
        last_err: Exception | None = None
        for a in non_none:
            try:
                return _coerce(value, a, path)
            except ConfigError as e:  # try the next alternative
                last_err = e
        raise ConfigError(str(last_err) if last_err else f"{path}: invalid value {value!r}")
    if origin in (list, typing.List):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {_type_name(type(value))}")
        inner = args[0] if args else Any
        return [_coerce(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if annotation is dict or origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {_type_name(type(value))}")
        return value
    return value


def build_dataclass(cls: type, data: dict, path: str = "") -> Any:
    """Instantiate ``cls`` from ``data`` rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = path or cls.__name__
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        sub = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], sub)
        elif f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[misc]
            continue
        else:
            raise ConfigError(f"{sub}: required key missing")
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def deep_merge(base: dict, overlay: dict, path: str = "overrides") -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        sub = f"{path}.{k}"
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v, sub)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Post-build validation (locked decisions and cross-key rules)
# ---------------------------------------------------------------------------


def validate_pipeline_cfg(cfg: PipelineCfg) -> None:
    if cfg.version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"version: expected {CONFIG_SCHEMA_VERSION}, got {cfg.version}")
    unknown_stages = [s for s in cfg.run.stages if s not in DEFAULT_STAGES]
    if unknown_stages:
        raise ConfigError(f"run.stages: unknown stage(s) {unknown_stages}")
    order = {s: i for i, s in enumerate(DEFAULT_STAGES)}
    if [order[s] for s in cfg.run.stages] != sorted(order[s] for s in cfg.run.stages):
        raise ConfigError("run.stages: stages must be listed in pipeline order")
    # T10 two-switch rule: a signal cannot be in the prompt if it is not computed.
    for name in ("audio", "faiss", "ocr"):
        sw: TextSignalCfg = getattr(cfg.signals, name)
        if sw.include_in_vlm_prompt and not sw.compute_and_log:
            raise ConfigError(
                f"signals.{name}: include_in_vlm_prompt=true requires compute_and_log=true")
    # Locked decisions (Revision 3 §3).
    if cfg.merge.legacy_filters_enabled:
        raise ConfigError("merge.legacy_filters_enabled must remain false (T17)")
    if cfg.merge.representative != "tighter":
        raise ConfigError("merge.representative is locked to 'tighter'")
    if not cfg.detectors.yolo.class_agnostic:
        raise ConfigError("detectors.yolo.class_agnostic must stay true (decision locked)")
    if cfg.router.rules.rule_5_reject:
        raise ConfigError("router.rules.rule_5_reject is locked off for all of M3")
    if cfg.router.rules.rule_8:
        raise ConfigError("router.rules.rule_8 is locked off for all of M3")
    if cfg.router.dedup.post_vlm_min_merge_area_enabled:
        raise ConfigError("router.dedup.post_vlm_min_merge_area_enabled is locked off (T45)")
    if cfg.router.not_logo_confidence_source == "scored" and not cfg.vlm.scoring.enabled:
        raise ConfigError("router.not_logo_confidence_source 'scored' requires vlm.scoring.enabled (T31/T32)")
    if not cfg.ocr.matching.exact_match_short:
        raise ConfigError("ocr.matching.exact_match_short must stay true (T27/T28 has no alternative branch)")
    if cfg.ingest.mode == "frames" and not cfg.ingest.frames.dir:
        raise ConfigError("ingest.frames.dir is required when ingest.mode is 'frames'")
    unknown_blind = [s for s in cfg.vlm.blind_pass_signals
                     if s not in ("tight_crop", "enlarged_crop", "full_frame", "temporal_context")]
    if unknown_blind:
        raise ConfigError(f"vlm.blind_pass_signals: only image signals allowed, got {unknown_blind}")
    bad_lists = [m for m in cfg.ocr.matching.match_lists if m not in ("canonical", "audio")]
    if bad_lists:
        raise ConfigError(f"ocr.matching.match_lists: only canonical/audio allowed (T26), got {bad_lists}")
    if cfg.sampling.temporal_window < 1:
        raise ConfigError("sampling.temporal_window must be >= 1")
    if any(abs(o) > cfg.sampling.temporal_window or o == 0 for o in cfg.sampling.temporal_offsets_for_vlm):
        raise ConfigError("sampling.temporal_offsets_for_vlm must be non-zero and within temporal_window")


# ---------------------------------------------------------------------------
# Resolution and hashing
# ---------------------------------------------------------------------------


def _strip_excluded(d: dict) -> dict:
    d = copy.deepcopy(d)
    for keypath in HASH_EXCLUDED_KEYS:
        node = d
        for k in keypath[:-1]:
            node = node.get(k, {})
        node.pop(keypath[-1], None)
    return d


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(resolved_pipeline: dict, video: dict) -> str:
    doc = {"pipeline": _strip_excluded(resolved_pipeline), "video": video}
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


@dataclass
class ResolvedConfig:
    pipeline: PipelineCfg
    video: VideoCfg
    resolved: dict            # to_dict(pipeline) after overlay, defaults filled
    config_hash: str
    pipeline_path: str | None
    video_path: str | None
    effective: dict = field(default_factory=dict)   # e.g. shots detector forced in frames mode

    @property
    def hash8(self) -> str:
        return self.config_hash[:8]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def resolve_config(
    pipeline_yaml: str | Path | dict,
    video_yaml: str | Path | dict | None = None,
    *,
    frames_dir: str | None = None,
    video_id: str | None = None,
    source: str | None = None,
    run_id: str | None = None,
) -> ResolvedConfig:
    """Load, overlay, validate and hash. CLI arguments override the files."""
    if isinstance(pipeline_yaml, dict):
        pipeline_data, pipeline_path = copy.deepcopy(pipeline_yaml), None
    else:
        pipeline_data, pipeline_path = load_yaml(pipeline_yaml), str(pipeline_yaml)

    if video_yaml is None:
        video_data, video_path = {}, None
    elif isinstance(video_yaml, dict):
        video_data, video_path = copy.deepcopy(video_yaml), None
    else:
        video_data, video_path = load_yaml(video_yaml), str(video_yaml)

    video = build_dataclass(VideoCfg, video_data, "video")
    if video_id:
        video.video_id = str(video_id)
    if source:
        video.source = source
    if not video.video_id:
        raise ConfigError("video_id is required (per-video file or --video-id)")

    merged = deep_merge(pipeline_data, video.overrides or {})
    if frames_dir:
        merged = deep_merge(merged, {"ingest": {"mode": "frames", "frames": {"dir": frames_dir}}})
    if run_id:
        merged = deep_merge(merged, {"run": {"run_id": run_id}})

    cfg = build_dataclass(PipelineCfg, merged, "")
    validate_pipeline_cfg(cfg)
    if cfg.ingest.mode == "video" and not video.source:
        raise ConfigError("video.source is required when ingest.mode is 'video'")

    effective: dict[str, Any] = {}
    if cfg.ingest.mode == "frames" and cfg.shots.detector != "content":
        effective["shots.detector"] = "content"   # 03 §5.2, 04 shots.detector note

    resolved = to_dict(cfg)
    video_dict = {k: v for k, v in to_dict(video).items() if k != "overrides"}
    config_hash = compute_config_hash(resolved, video_dict)
    return ResolvedConfig(cfg, video, resolved, config_hash, pipeline_path, video_path, effective)


def default_pipeline_dict() -> dict:
    return to_dict(PipelineCfg())


def signal_switches(cfg: PipelineCfg) -> dict:
    """The T10 switch table as written into the manifest."""
    out: dict[str, dict] = {}
    for name in ("tight_crop", "enlarged_crop", "full_frame", "temporal_context"):
        out[name] = {"include_in_vlm_prompt": getattr(cfg.signals, name).include_in_vlm_prompt}
    for name in ("audio", "faiss", "ocr"):
        sw = getattr(cfg.signals, name)
        out[name] = {"compute_and_log": sw.compute_and_log, "include_in_vlm_prompt": sw.include_in_vlm_prompt}
    return out


def text_signal_in_prompt(cfg: PipelineCfg, name: str) -> bool:
    sw: TextSignalCfg = getattr(cfg.signals, name)
    return bool(sw.compute_and_log and sw.include_in_vlm_prompt)

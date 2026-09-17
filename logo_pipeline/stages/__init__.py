"""Stage registry. Stage modules import their heavy dependencies lazily."""

from __future__ import annotations

from .base import Stage


def get_stage(name: str) -> Stage:
    # Imports are local so that importing the registry never pulls torch etc.
    if name == "ingest":
        from .ingest import IngestStage as S
    elif name == "shots":
        from .shots import ShotsStage as S
    elif name == "sampling":
        from .sampling import SamplingStage as S
    elif name == "detect_gdino":
        from .detect_gdino import DetectGdinoStage as S
    elif name == "detect_yolo":
        from .detect_yolo import DetectYoloStage as S
    elif name == "merge":
        from .merge import MergeStage as S
    elif name == "refine":
        from .refine import RefineStage as S
    elif name == "ocr":
        from .ocr import OcrStage as S
    elif name == "faiss_query":
        from .faiss_query import FaissQueryStage as S
    elif name == "audio":
        from .audio import AudioStage as S
    elif name == "vlm":
        from .vlm import VlmStage as S
    elif name == "canonicalise":
        from .canonicalise import CanonicaliseStage as S
    elif name == "watermark":
        from .watermark import WatermarkStage as S
    elif name == "router_input":
        from .router_input import RouterInputStage as S
    elif name == "router":
        from .router.engine import RouterStage as S
    elif name == "output":
        from .output import OutputStage as S
    else:
        raise KeyError(f"unknown stage {name!r}")
    return S()


__all__ = ["Stage", "get_stage"]

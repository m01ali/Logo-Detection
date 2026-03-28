# post_filtering.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image

# ---- Generic Strategy Interface ------------------------------------------------

class PostFilterStrategy(ABC):
    """
    Strategy interface: apply a post-filter to detections and return:
      - annotated_rows: input rows + postfilter metadata (verdict/method/applied)
      - verdict_by_logo_id: mapping from logo_id -> verdict for quick lookup in frame_results
    Each 'row' is expected to be a dict with at least:
      - 'brand' (str)
      - 'filename' (str)  e.g., f"{logo_id}.png"
      - 'source' (str)    e.g., "Top_2K_Brands"
    """
    name: str = "PostFilterStrategy"

    @abstractmethod
    def apply(
        self,
        detected_rows: List[dict],
        image_dir: str
    ) -> Tuple[List[dict], Dict[str, str]]:
        ...


# ---- Qwen-based "Correct | Other | Incorrect" Strategy ------------------------

class QwenCorrectnessStrategy(PostFilterStrategy):
    """
    Applies a Qwen-VL-style triage ('Correct' | 'Other' | 'Incorrect') to a subset
    of detections selected by `sources_to_filter`.

    Only rows with source in `sources_to_filter` are sent to the model.
    Everything else is skipped but still annotated (applied=False, verdict='Skipped').

    Parameters
    ----------
    model : object with .chunked_run_examples(prompts, images, batch_size=..., **model_kwargs)
    sources_to_filter : set of sources to run the LLM on (e.g., {"Top_2K_Brands"})
    prompt_template : string template with '{brand}'
    image_size : tuple (w, h) to resize crops for the model
    batch_size : int for batched inference
    model_kwargs : dict forwarded to the model generation call
    """

    name = "Qwen2.5-VL-Correctness"

    def __init__(
        self,
        model,
        sources_to_filter: Optional[Set[str]] = None,
        prompt_template: str = "",
        image_size: Tuple[int, int] = (240, 240),
        batch_size: int = 8,
        model_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.sources_to_filter = set(sources_to_filter or {"Top_2K_Brands"})
        self.prompt_template = prompt_template
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs or {"temperature": 0.1, "do_sample": True}

    @staticmethod
    def _normalize_verdict(s: str) -> str:
        # Be robust to accidental extra tokens/spaces
        s_clean = (s or "").strip().lower()
        if "other" in s_clean:
            return "Other"
        if "incorrect" in s_clean:
            return "Incorrect"
        if "correct" in s_clean:
            return "Correct"
        # Fallback safety
        return "Incorrect"

    def apply(
        self,
        detected_rows: List[dict],
        image_dir: str
    ) -> Tuple[List[dict], Dict[str, str]]:
        # 1) Build batches only for rows whose source we want to filter
        idxs_to_check: List[int] = []
        prompts: List[str] = []
        images: List[Image.Image] = []

        for i, row in enumerate(detected_rows):
            src = str(row.get("source", "unknown"))
            if src in self.sources_to_filter:
                brand = str(row.get("brand", "")).strip()
                fname = row.get("filename")
                if not fname:
                    continue
                path = os.path.join(image_dir, fname)
                if not os.path.exists(path):
                    continue
                try:
                    img = Image.open(path).convert("RGB").resize(self.image_size)
                except Exception:
                    continue

                prompts.append(self.prompt_template.format(brand=brand))
                images.append(img)
                idxs_to_check.append(i)

        # 2) Run the model if we have anything to check
        verdicts: List[str] = []
        if idxs_to_check:
            outputs: Sequence[str] = self.model.chunked_run_examples(
                prompts, images, batch_size=self.batch_size, **self.model_kwargs
            )
            verdicts = [self._normalize_verdict(o) for o in outputs]

        # 3) Annotate rows and build verdict_by_logo_id
        verdict_by_logo_id: Dict[str, str] = {}

        # First, default annotation for all rows (not applied)
        for row in detected_rows:
            row.setdefault("postfilter_method", self.name)
            row.setdefault("postfilter_applied", False)
            row.setdefault("postfilter_verdict", "Skipped")

        # Then, fill in for those we actually checked
        for local_idx, row_idx in enumerate(idxs_to_check):
            v = verdicts[local_idx] if local_idx < len(verdicts) else "Incorrect"
            row = detected_rows[row_idx]
            row["postfilter_applied"] = True
            row["postfilter_verdict"] = v

            # recover logo_id from filename like f"{logo_id}.png"
            logo_id = os.path.splitext(row.get("filename", ""))[0]
            if logo_id:
                verdict_by_logo_id[logo_id] = v

        return detected_rows, verdict_by_logo_id


# ---- Orchestrator -------------------------------------------------------------

# def apply_post_filters(
#     detected_rows: List[dict],
#     image_dir: str,
#     frame_results: List[dict],
#     strategies: Iterable[PostFilterStrategy],
# ) -> Tuple[List[dict], List[dict], List[dict]]:
#     """
#     Runs one or more post-filter strategies (in order).

#     Returns:
#       filtered_rows  : rows after removing items that failed *any* applied strategy
#       annotated_rows : every row with postfilter annotations (keep-all / diagnostics)
#       frame_results  : updated in-place with postfilter verdicts (non-correct -> UNKNOWN)
#     """
#     annotated_rows = detected_rows

#     # aggregate logo_id -> last verdict (if multiple strategies, last one wins)
#     logo_verdicts: Dict[str, str] = {}

#     for strat in strategies:
#         annotated_rows, vmap = strat.apply(annotated_rows, image_dir=image_dir)
#         logo_verdicts.update(vmap)

#     # Filter logic: if a row had a postfilter_applied and verdict != 'Correct' -> drop.
#     filtered_rows: List[dict] = []
#     for row in annotated_rows:
#         applied = row.get("postfilter_applied", False)
#         verdict = row.get("postfilter_verdict", "Skipped")
#         if applied:
#             if verdict == "Correct":
#                 filtered_rows.append(row)
#             # else drop
#         else:
#             # strategy didn't apply -> keep row unchanged
#             filtered_rows.append(row)

#     # Update frame_results so downstream steps (e.g., annotate_video) reflect filtering
#     for fr in frame_results:
#         results = fr.get("results", {})
#         for logo_id, info in results.items():
#             v = logo_verdicts.get(logo_id)
#             if v:
#                 # attach metadata
#                 info["postfilter"] = {
#                     "method": " | ".join(s.name for s in strategies),
#                     "verdict": v,
#                     "applied": True,
#                 }
#                 # if not Correct => blank it out so it's treated as unknown/ignored later
#                 if v != "Correct":
#                     info["brand"] = "UNKNOWN"
#             else:
#                 info["postfilter"] = {
#                     "method": " | ".join(s.name for s in strategies),
#                     "verdict": "Skipped",
#                     "applied": False,
#                 }

#     return filtered_rows, annotated_rows, frame_results

def apply_post_filters(
    detected_rows: List[dict],
    image_dir: str,
    frame_results: List[dict],
    strategies: Iterable[PostFilterStrategy],
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Runs one or more post-filter strategies (in order).

    Returns:
      filtered_rows            : rows after removing items that failed *any* applied strategy
      annotated_rows           : every row with postfilter annotations (keep-all / diagnostics)
      frame_results            : updated in-place with postfilter verdicts (non-correct -> brand='UNKNOWN')
      filtered_frame_results   : frame_results-like structure containing only 'Skipped' and 'Correct' items
                                 (per-frame 'results' dict is filtered accordingly; frames are kept in order)
    """
    annotated_rows = detected_rows

    # Aggregate logo_id -> last verdict (if multiple strategies, last one wins)
    logo_verdicts: Dict[str, str] = {}
    for strat in strategies:
        annotated_rows, vmap = strat.apply(annotated_rows, image_dir=image_dir)
        logo_verdicts.update(vmap)

    # Filter logic for rows: if postfilter_applied and verdict != 'Correct' -> drop.
    filtered_rows: List[dict] = []
    for row in annotated_rows:
        applied = row.get("postfilter_applied", False)
        verdict = row.get("postfilter_verdict", "Skipped")
        if applied:
            if verdict == "Correct":
                filtered_rows.append(row)
            # else drop
        else:
            # Strategy didn't apply -> keep row unchanged
            filtered_rows.append(row)

    # Update frame_results so downstream steps (e.g., annotate_video) reflect filtering
    methods_str = " | ".join(s.name for s in strategies) if strategies else ""
    for fr in frame_results:
        results = fr.get("results", {})
        for logo_id, info in results.items():
            v = logo_verdicts.get(logo_id)
            if v:
                # attach metadata
                info["postfilter"] = {
                    "method": methods_str,
                    "verdict": v,
                    "applied": True,
                }
                # if not Correct => blank it out so it's treated as unknown/ignored later
                if v != "Correct":
                    info["brand"] = "UNKNOWN"
            else:
                info["postfilter"] = {
                    "method": methods_str,
                    "verdict": "Skipped",
                    "applied": False,
                }

    # Build filtered_frame_results (keep only 'Skipped' and 'Correct')
    filtered_frame_results: List[dict] = []
    for fr in frame_results:
        results = fr.get("results", {})
        kept: Dict[str, dict] = {}
        for logo_id, info in results.items():
            pf = info.get("postfilter", {})
            applied = pf.get("applied", False)
            verdict = pf.get("verdict", "Skipped")
            # Keep if:
            # - applied and verdict == 'Correct'
            # - not applied (i.e., 'Skipped' by definition)
            if (applied and verdict == "Correct") or (not applied):
                kept[logo_id] = info
        # Preserve frame/timestamp; keep empty frames too (safe for downstream alignment)
        filtered_frame_results.append({
            "frame_number": fr.get("frame_number"),
            "timestamp": fr.get("timestamp"),
            "results": kept
        })

    return filtered_rows, annotated_rows, frame_results, filtered_frame_results
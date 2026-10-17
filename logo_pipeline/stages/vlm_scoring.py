"""``stages/vlm_scoring.py`` (T30/T31, ``spec docs/15``): closed-set decision
scoring under the hinted context.

For each representative the hinted prompt is rebuilt exactly as the production
call, followed by an assistant prefix that opens the JSON (``{"is_logo": "``).
Each label ``L`` of ``vlm.scoring.labels`` is teacher-forced as ``L + '"'``
and scored:

* ``sequence_logprob``: the summed log-probability of the label's tokens;
* ``first_token_softmax``: the log-probability of the first token where the
  labels' token sequences diverge.

``decision_scores = softmax(scores over the labels)``,
``logo_probability_scored = 100 * (P(logo) + P(partial_logo))`` [ASSUMPTION C7]
and ``not_logo_confidence = P(not_logo)``.

Tokeniser boundaries: the prefix ends in ``"`` and a BPE vocabulary may merge
that quote with the label's first letters. Every label is therefore scored
from the same token position — the longest prefix all labels share with the
context — so each label pays for exactly the same leading text.

The prefix is taken from the model's own hinted reply where it contains
``"is_logo": "`` (pretty-printed JSON starts ``{\\n  "is_logo": "``), else the
canonical ``{"is_logo": "``; the one used is recorded per row.

Runs only when ``vlm.scoring.enabled``; the router keeps the generated field
until T32's distribution check passes (``router.not_logo_confidence_source``).
"""

from __future__ import annotations

import math
import re
import time

from .base import (MERGED_DETECTIONS_FILE, VLM_HINTED_FILE, VLM_SCORES_FILE, JsonlAppender, RunContext, Stage,
                    StageResult)

DEFAULT_PREFIX = '{"is_logo": "'
_PREFIX_RE = re.compile(r'^(.*?"is_logo"\s*:\s*")', re.DOTALL)


def prefix_from_reply(raw: str | None) -> str:
    """The text the model emitted up to the opening quote of the ``is_logo`` value."""
    if raw:
        m = _PREFIX_RE.match(raw)
        if m and m.group(1).lstrip().startswith("{"):
            return m.group(1).lstrip()
    return DEFAULT_PREFIX


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def derive_scores(labels: list[str], raw_scores: list[float]) -> dict:
    """``decision_scores``, ``logo_probability_scored``, ``not_logo_confidence`` from per-label scores."""
    probs = dict(zip(labels, softmax(raw_scores)))
    any_logo = probs.get("logo", 0.0) + probs.get("partial_logo", 0.0)
    return {
        "decision_scores": {k: round(v, 6) for k, v in probs.items()},
        "logo_probability_scored": round(100.0 * any_logo, 4) if "logo" in probs else None,
        "not_logo_confidence": round(probs["not_logo"], 6) if "not_logo" in probs else None,
    }


def common_prefix_len(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def label_spans(ctx_ids, label_ids: list, method: str) -> tuple[int, list[tuple[int, int]]]:
    """Positions to score in each label's token sequence (pure; unit-tested).

    Returns ``(k, spans)``: ``k`` is the token index every label shares with the context, and
    ``spans[b] = (start, end)`` the half-open range of label ``b``'s tokens to score.
    """
    k = min(common_prefix_len(ctx_ids, ids) for ids in label_ids)
    if method == "sequence_logprob":
        return k, [(k, len(ids)) for ids in label_ids]
    # first_token_softmax: the first position (>= k) where the label sequences disagree.
    d = k
    while all(d < len(ids) for ids in label_ids) and len({int(ids[d]) for ids in label_ids}) == 1:
        d += 1
    return k, [(d, min(d + 1, len(ids))) for ids in label_ids]


def score_labels(model, processor, messages: list[dict], prefix: str, labels: list[str], method: str) -> list[float]:
    """One batched forward pass over ``context + prefix + label + '"'`` for every label."""
    import torch
    from qwen_vl_utils import process_vision_info

    text_ctx = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
    image_inputs, video_inputs = process_vision_info(messages)
    n = len(labels)
    ctx_batch = processor(text=[text_ctx], images=image_inputs, videos=video_inputs, return_tensors="pt")
    ctx_ids = ctx_batch["input_ids"][0].tolist()
    batch = processor(text=[text_ctx + lab + '"' for lab in labels],
                      images=(image_inputs or []) * n or None, videos=(video_inputs or []) * n or None,
                      padding=True, return_tensors="pt")
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    total = ids.shape[1]
    real = [ids[b][mask[b].bool()].tolist() for b in range(n)]       # left padding: real tokens at the end
    _, spans = label_spans(ctx_ids, real, method)
    keep = max(len(r) - start for r, (start, _) in zip(real, spans)) + 1

    batch = batch.to(model.device)
    with torch.no_grad():
        try:
            logits = model(**batch, logits_to_keep=keep).logits
        except TypeError:                      # older transformers have no logits_to_keep
            logits = model(**batch).logits
    if logits.shape[1] != keep:                # ...and some accept it via **kwargs and ignore it
        logits = logits[:, -keep:, :]
    logp = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    for b in range(n):
        pad = total - len(real[b])
        start, end = spans[b]
        s = 0.0
        for t in range(start, end):
            pos = pad + t - 1 - (total - keep)          # logits at t-1 predict token t
            s += float(logp[b, pos, real[b][t]])
        scores.append(s)
    return scores


class VlmScoringStage(Stage):
    name = "vlm_scoring"
    device = "gpu"
    inputs = [MERGED_DETECTIONS_FILE, VLM_HINTED_FILE]
    outputs: list[str] = []

    def outputs_exist(self, ctx: RunContext) -> bool:
        if not ctx.cfg.vlm.scoring.enabled:
            return True
        if not ctx.path(VLM_SCORES_FILE).exists():
            return False
        det_ids = {r["det_id"] for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE)}
        return det_ids <= {r["det_id"] for r in ctx.read_jsonl(VLM_SCORES_FILE)}

    def run(self, ctx: RunContext) -> StageResult:
        scfg = ctx.cfg.vlm.scoring
        if not scfg.enabled:
            ctx.log.stage_skipped(self.name, "vlm.scoring.enabled is false")
            return StageResult(counts={"scored": 0, "enabled": False})
        det_ids = sorted(r["det_id"] for r in ctx.read_jsonl(MERGED_DETECTIONS_FILE))
        out = JsonlAppender(ctx.path(VLM_SCORES_FILE), ctx.header(
            "vlm_scoring", method=scfg.method, labels=list(scfg.labels), model_id=ctx.cfg.vlm.model_id,
            quantisation=ctx.cfg.vlm.quantisation, constrained_decoding=False), truncate=ctx.force)
        try:
            todo = [d for d in det_ids if d not in out.done]
            if not todo:
                return StageResult(counts={"scored": 0, "skipped": len(det_ids)})
            n = score_detections(ctx, todo, out.write)
        finally:
            out.close()
        return StageResult(counts={"scored": n})


def score_detections(ctx: RunContext, det_ids: list[int], write) -> int:
    """Scores ``det_ids`` under the hinted context, calling ``write(record)`` for each."""
    from .vlm import DetectionInputs, _get_model, build_messages

    vcfg, scfg = ctx.cfg.vlm, ctx.cfg.vlm.scoring
    hinted = {r["det_id"]: r for r in ctx.read_jsonl(VLM_HINTED_FILE)} if ctx.path(VLM_HINTED_FILE).exists() else {}
    inp = DetectionInputs(ctx)
    model, processor = _get_model(vcfg.model_id, vcfg.quantisation, vcfg.max_pixels)
    labels = list(scfg.labels)
    for i, det_id in enumerate(det_ids, 1):
        frame_img = ctx.open_frame_image(inp.reps[det_id]["frame_idx"])
        content, _, _ = inp.content(det_id, frame_img, blind=False)
        prefix = prefix_from_reply((hinted.get(det_id) or {}).get("raw"))
        t0 = time.perf_counter()
        raw_scores = score_labels(model, processor, build_messages(content), prefix, labels, scfg.method)
        rec = {"det_id": det_id, **derive_scores(labels, raw_scores),
               "raw_scores": {lab: round(v, 6) for lab, v in zip(labels, raw_scores)},
               "method": scfg.method, "prefix": prefix, "elapsed_s": round(time.perf_counter() - t0, 3)}
        write(rec)
        ctx.log.progress(i, len(det_ids), unit="det")
    return len(det_ids)


# ---------------------------------------------------------------------------
# T30 spike (``python -m logo_pipeline vlm-spike``)
# ---------------------------------------------------------------------------

SPIKE_FILE = "reports/vlm_spike.jsonl"
SPIKE_REPORT = "reports/vlm_spike.md"
POSITIVE = ("logo", "partial_logo")


def spike_sample(hinted: dict[int, dict], n: int, seed: int = 42) -> list[int]:
    """``n`` det_ids spanning the hinted ``is_logo`` values (round-robin over the strata, each
    shuffled with ``seed``), so a sample is not all ``not_logo`` when most rows are."""
    import random
    strata: dict[str, list[int]] = {}
    for det_id in sorted(hinted):
        label = ((hinted[det_id].get("parsed") or {}).get("is_logo")) or "unparsed"
        strata.setdefault(str(label), []).append(det_id)
    rng = random.Random(seed)
    for ids in strata.values():
        rng.shuffle(ids)
    out: list[int] = []
    order = sorted(strata)
    while len(out) < n and any(strata.values()):
        for k in order:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return sorted(out)


def spike_verdict(records: list[dict], hinted: dict[int, dict], n_requested: int,
                  min_distinct: int = 20) -> dict:
    """The T30 exit criteria over spike records (pure)."""
    scored = [r for r in records if r.get("decision_scores")]
    tracked = considered = 0
    for r in scored:
        is_logo = ((hinted.get(r["det_id"]) or {}).get("parsed") or {}).get("is_logo")
        if is_logo is None:
            continue
        considered += 1
        ds = r["decision_scores"]
        mass_logo = ds.get("logo", 0.0) + ds.get("partial_logo", 0.0)
        if (is_logo in POSITIVE and mass_logo > 0.5) or (is_logo == "not_logo" and ds.get("not_logo", 0.0) > 0.5):
            tracked += 1
    distinct = len({round(r["logo_probability_scored"], 2) for r in scored
                    if r.get("logo_probability_scored") is not None})
    checks = {
        "all_scored": len(scored) == n_requested,
        "mass_tracks_is_logo_in_majority": considered > 0 and tracked / considered > 0.5,
        f"at_least_{min_distinct}_distinct_values": distinct >= min_distinct,
    }
    return {"n_requested": n_requested, "n_scored": len(scored), "tracked": tracked, "considered": considered,
            "distinct_values": distinct, "checks": checks, "passed": all(checks.values())}


def spike_report(verdict: dict, records: list[dict], hinted: dict[int, dict], ctx_info: dict) -> str:
    from collections import Counter
    secs = sorted(r["elapsed_s"] for r in records if r.get("elapsed_s") is not None)
    hinted_secs = sorted(h["elapsed_s"] for h in hinted.values() if h.get("elapsed_s") is not None)
    prefixes = Counter(r.get("prefix") for r in records)

    def med(v):
        return f"{v[len(v) // 2]:.2f} s" if v else "n/a"

    lines = [
        "# T30 log-prob spike", "",
        f"**Result: {'PASS' if verdict['passed'] else 'FAIL'}.** "
        + ("The scored field can be enabled for the full run (T31); T32's distribution check still gates any rule."
           if verdict["passed"] else
           "Rules 1-fallback and 8 stay off; the categorical fallback of 29 July is to be scoped and the client informed."),
        "",
        "## Setup", "",
        f"- Checkpoint: `{ctx_info.get('model_id')}`, quantisation `{ctx_info.get('quantisation')}`, transformers.",
        f"- Method: `{ctx_info.get('method')}`; labels {ctx_info.get('labels')}.",
        "- Constrained / JSON decoding: **not active** (plain greedy generation produces the text output; "
        "scoring is a separate teacher-forced pass).",
        "- Assistant prefix(es) used (taken from the model's own hinted reply where present):",
    ]
    lines += [f"  - `{p!r}` × {c}" for p, c in prefixes.most_common()]
    lines += ["", "## Exit criteria", "", "| check | result |", "|---|---|"]
    lines += [f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in verdict["checks"].items()]
    lines += ["",
              f"- Scored {verdict['n_scored']} of {verdict['n_requested']}.",
              f"- Mass tracks `is_logo` on {verdict['tracked']} of {verdict['considered']} rows with a parsed decision.",
              f"- Distinct `logo_probability_scored` values (2 dp): {verdict['distinct_values']}.",
              "", "## Cost (feeds T36)", "",
              f"- Scoring pass per detection: median {med(secs)} over {len(secs)} rows.",
              f"- Hinted generation call per detection (this run): median {med(hinted_secs)}"
              + ("" if hinted_secs else " (older run without per-call timings)") + ".",
              "", "## Rows", "", "| det_id | hinted is_logo | P(logo) | P(partial) | P(not_logo) | P(uncertain) | "
              "logo_probability_scored |", "|---|---|---|---|---|---|---|"]
    for r in records:
        ds = r.get("decision_scores") or {}
        is_logo = ((hinted.get(r["det_id"]) or {}).get("parsed") or {}).get("is_logo")
        lines.append(f"| {r['det_id']} | {is_logo} | {ds.get('logo', 0):.3f} | {ds.get('partial_logo', 0):.3f} | "
                     f"{ds.get('not_logo', 0):.3f} | {ds.get('uncertain', 0):.3f} | {r.get('logo_probability_scored')} |")
    return "\n".join(lines) + "\n"


def run_spike(ctx: RunContext, n: int = 50, seed: int = 42) -> dict:
    """Scores a stratified sample of an existing run's representatives and writes the spike report."""
    if not ctx.path(VLM_HINTED_FILE).exists():
        raise FileNotFoundError(f"{VLM_HINTED_FILE} is required: the spike scores under the hinted context")
    hinted = {r["det_id"]: r for r in ctx.read_jsonl(VLM_HINTED_FILE)}
    sample = spike_sample(hinted, n, seed)
    records: list[dict] = []
    ctx.write_jsonl(SPIKE_FILE, [], header=ctx.header("vlm_spike", n=n, seed=seed))
    with open(ctx.path(SPIKE_FILE), "a", encoding="utf-8") as fh:
        from .base import dumps_line

        def keep(rec):
            records.append(rec)
            fh.write(dumps_line(rec) + "\n")
            fh.flush()
        score_detections(ctx, sample, keep)
    scfg = ctx.cfg.vlm.scoring
    verdict = spike_verdict(records, hinted, len(sample))
    info = {"model_id": ctx.cfg.vlm.model_id, "quantisation": ctx.cfg.vlm.quantisation, "method": scfg.method,
            "labels": list(scfg.labels)}
    ctx.path(SPIKE_REPORT).parent.mkdir(parents=True, exist_ok=True)
    ctx.path(SPIKE_REPORT).write_text(spike_report(verdict, records, hinted, info), encoding="utf-8")
    return verdict

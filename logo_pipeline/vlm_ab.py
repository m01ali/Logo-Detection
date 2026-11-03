"""19 A12: re-run the hinted VLM call on a sample of a route-three run under changed
VLM settings, and say what the change would do.

``python -m logo_pipeline vlm-ab --run runs/14/route3 --tag upscale --set vlm.crop_min_pixels=50176
[--set vlm.wordmark_rule=true] [--n-rescue 40] [--n-held 20] [--n-neg 40] [--seed 42]``

Three seeded groups, read from ``eval/iou_match.py``'s ``iou_match_matched.csv`` /
``iou_match_unmatched.csv`` and the run's ``vlm/hinted.jsonl``:

* ``rescue`` — boxes matching a gold logo that the hinted pass called ``not_logo``
  (Rule 1 rejected them);
* ``held`` — boxes matching a gold logo that the hinted pass called ``logo`` or
  ``partial_logo``;
* ``negative`` — detections whose best IoU with the frame's gold boxes is below 0.1
  and that the hinted pass called ``not_logo``: boxes no labeller drew, the best
  stand-in for true non-logos.

Each call is the stage's own hinted call (same inputs, same parser) under the run's
manifest config with the ``--set`` overrides on top. Nothing in the run directory
changes except ``vlm/ab_<tag>.jsonl`` (appended per det_id, so a stopped run resumes)
and ``reports/vlm_ab_<tag>.md`` + ``.csv``. The router is not re-run: Rule 1 rejects a
``not_logo`` row at generated confidence ≥ 80, which every rescue row has, so a flip to
``logo`` / ``partial_logo`` is a box that escapes Rule 1.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import yaml

from .brands import load_brand_list, normalise
from .config import deep_merge, resolve_config
from .logging_utils import StageLogger
from .router_only import video_config_for
from .stages.base import RUN_MANIFEST, VLM_HINTED_FILE, JsonlAppender, RunContext

GROUPS = ("rescue", "held", "negative")
LOGO_LIKE = ("logo", "partial_logo")
NEG_MAX_IOU = 0.1


def parse_sets(items: list[str]) -> dict:
    """``["vlm.crop_min_pixels=50176", ...]`` -> a nested overrides dict (values read as YAML)."""
    out: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `eval iou_match` on this run first")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _decision(rec: dict | None) -> str | None:
    return ((rec or {}).get("parsed") or {}).get("is_logo")


def sample_groups(matched: list[dict], unmatched: list[dict], hinted: dict[int, dict], n: dict[str, int],
                  seed: int) -> tuple[dict[str, list[int]], dict[str, int]]:
    """``({group: det_ids}, {group: pool size})``; each group a seeded sample of its pool."""
    def ids(rows, keep):
        return sorted({int(r["det_id"]) for r in rows if r.get("det_id") not in ("", None) and keep(r)})
    pools = {
        "rescue": ids(matched, lambda r: _decision(hinted.get(int(r["det_id"]))) == "not_logo"),
        "held": ids(matched, lambda r: _decision(hinted.get(int(r["det_id"]))) in LOGO_LIKE),
        "negative": ids(unmatched, lambda r: float(r["max_iou_with_gold"]) < NEG_MAX_IOU
                        and _decision(hinted.get(int(r["det_id"]))) == "not_logo"),
    }
    rng = random.Random(seed)
    groups = {g: sorted(rng.sample(pools[g], min(n.get(g, 0), len(pools[g])))) for g in GROUPS}
    return groups, {g: len(p) for g, p in pools.items()}


def open_ab_context(run_dir: Path, overrides: dict, repo_root: Path, pipeline_yaml: str | None = None) -> RunContext:
    """The run's own settings (its manifest's resolved config, else ``pipeline_yaml``) plus ``overrides``."""
    manifest = json.loads((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))
    pipeline = manifest.get("config") or str(pipeline_yaml or repo_root / "configs" / "pipeline.yaml")
    video = video_config_for(run_dir, None)
    video = dict(video, overrides=deep_merge(video.get("overrides") or {}, overrides))
    resolved = resolve_config(pipeline, video)
    bpath = Path(resolved.pipeline.brands.list_path)
    brands = load_brand_list(bpath if bpath.is_absolute() else repo_root / bpath,
                             resolved.pipeline.brands.require_hash_match)
    return RunContext(run_dir, resolved, brands, StageLogger(None, log_every_n=10), repo_root=repo_root)


def run_ab(run_dir: Path, sets: list[str], tag: str, n: dict[str, int], seed: int, repo_root: Path,
           pipeline_yaml: str | None = None, generate_fn=None) -> Path:
    """Runs the calls (resuming per det_id) and writes the report; returns the report path."""
    from .stages.vlm import DetectionInputs, _get_model, generate, parse_reply

    run_dir = Path(run_dir)
    overrides = parse_sets(sets)
    ctx = open_ab_context(run_dir, overrides, repo_root, pipeline_yaml)
    hinted = {r["det_id"]: r for r in ctx.read_jsonl(VLM_HINTED_FILE)}
    matched = _read_csv(run_dir / "reports" / "iou_match_matched.csv")
    unmatched = _read_csv(run_dir / "reports" / "iou_match_unmatched.csv")
    groups, pools = sample_groups(matched, unmatched, hinted, n, seed)

    rel = f"vlm/ab_{tag}.jsonl"
    if ctx.path(rel).exists():
        head, _ = ctx.read_jsonl(rel, with_header=True)
        if head and (head.get("overrides") != overrides or head.get("seed") != seed):
            raise ValueError(f"{rel} was written with other settings ({head.get('overrides')}, seed "
                             f"{head.get('seed')}); use another --tag or delete the file")
    out = JsonlAppender(ctx.path(rel), ctx.header("vlm_ab", tag=tag, overrides=overrides, seed=seed,
                                                  groups=groups))
    todo = [(g, d) for g in GROUPS for d in groups[g] if d not in out.done]
    try:
        if todo:
            inp = DetectionInputs(ctx)
            if generate_fn is None:
                vcfg = ctx.cfg.vlm
                model, processor = _get_model(vcfg.model_id, vcfg.quantisation, vcfg.max_pixels)
                generate_fn = lambda content: generate(model, processor, content, vcfg)  # noqa: E731
            ctx.log.stage_start("vlm_ab", "gpu")
            for i, (group, det_id) in enumerate(todo, 1):
                frame_img = ctx.open_frame_image(inp.reps[det_id]["frame_idx"])
                content, _, _ = inp.content(det_id, frame_img, blind=False)
                t0 = time.perf_counter()
                raw = generate_fn(content)
                parsed = parse_reply(raw)
                out.write({"det_id": det_id, "group": group, "raw": raw, "parsed": parsed,
                           "vlm_parse_ok": parsed is not None, "elapsed_s": round(time.perf_counter() - t0, 3)})
                ctx.log.progress(i, len(todo), unit="det")
            ctx.log.stage_end("vlm_ab", {"calls": len(todo)})
    finally:
        out.close()
    return write_report(ctx, tag, overrides, groups, pools, hinted, matched)


def summarise(records: dict[int, dict], groups: dict[str, list[int]], hinted: dict[int, dict],
              gold: dict[int, str], brands, ccfg) -> tuple[dict, list[tuple]]:
    """Per-group counts and one CSV row per call."""
    from .stages.canonicalise import canonicalise

    def canon(brand):
        return canonicalise(brand, brands, ccfg).canonical if brand else None

    def gold_canon(cls):
        return brands.map_exact(cls) or normalise(cls)

    stats, rows = {}, []
    for g in GROUPS:
        s = {"n": 0, "base": Counter(), "variant": Counter(), "flipped": 0, "brand_correct_base": 0,
             "brand_correct_variant": 0, "unparsed": 0, "elapsed": [], "elapsed_base": []}
        for d in groups[g]:
            rec = records.get(d)
            if rec is None:
                continue
            base, var = _decision(hinted.get(d)), _decision(rec)
            s["n"] += 1
            s["base"][base or "unparsed"] += 1
            s["variant"][var or "unparsed"] += 1
            s["unparsed"] += var is None
            s["flipped"] += (var in LOGO_LIKE) if g != "held" else (var == "not_logo")
            vb = (rec.get("parsed") or {}).get("brand")
            bb = ((hinted.get(d) or {}).get("parsed") or {}).get("brand")
            vc, bc = canon(vb), canon(bb)
            truth = gold_canon(gold[d]) if d in gold else None
            if truth is not None:
                s["brand_correct_variant"] += vc == truth and var in LOGO_LIKE
                s["brand_correct_base"] += bc == truth and base in LOGO_LIKE
            s["elapsed"].append(rec.get("elapsed_s") or 0.0)
            if (hinted.get(d) or {}).get("elapsed_s") is not None:
                s["elapsed_base"].append(hinted[d]["elapsed_s"])
            rows.append((d, g, gold.get(d, ""), base, var, bb, vb, vc, truth is not None and vc == truth,
                         (rec.get("parsed") or {}).get("confidence"), rec.get("elapsed_s"),
                         (rec.get("parsed") or {}).get("reasoning")))
        stats[g] = s
    return stats, rows


def render(run_dir: Path, tag: str, overrides: dict, stats: dict, pools: dict) -> str:
    def mean(v):
        return f"{sum(v) / len(v):.1f}" if v else "—"

    def share(a, b):
        return f"{100 * a / b:.1f}%" if b else "—"

    r, h, ng = stats["rescue"], stats["held"], stats["negative"]
    lines = [f"# VLM A/B `{tag}` — `{run_dir}`", "",
             f"Overrides: `{json.dumps(overrides)}` on the run's own settings. The router is not re-run: a "
             "`not_logo` call here is a Rule 1 REJECT, a `logo` / `partial_logo` call escapes it.", "",
             "| group | what it is | sampled / pool | flipped | share |", "|---|---|---|---|---|",
             f"| rescue | true logos the run REJECTed (hinted `not_logo`) | {r['n']} / {pools['rescue']} | "
             f"{r['flipped']} now logo or partial | {share(r['flipped'], r['n'])} |",
             f"| held | true logos the run kept as logo | {h['n']} / {pools['held']} | "
             f"{h['flipped']} now not_logo | {share(h['flipped'], h['n'])} |",
             f"| negative | boxes no labeller drew (IoU < {NEG_MAX_IOU}), hinted `not_logo` | "
             f"{ng['n']} / {pools['negative']} | {ng['flipped']} now logo or partial | "
             f"{share(ng['flipped'], ng['n'])} |", ""]
    if r["n"] and pools["rescue"]:
        lines.append(f"- Projected over the run: about {round(pools['rescue'] * r['flipped'] / r['n'])} of "
                     f"{pools['rescue']} rejected true logos escape Rule 1"
                     + (f", against about {round(pools['negative'] * ng['flipped'] / ng['n'])} of "
                        f"{pools['negative']} unlabelled boxes that would start to as well."
                        if ng["n"] else "."))
    lines += [f"- Brand correct among rescued boxes: {r['brand_correct_variant']} of {r['flipped']}; among held "
              f"boxes: {h['brand_correct_variant']} of {h['n']} (was {h['brand_correct_base']}).",
              f"- Unparsed replies: {sum(s['unparsed'] for s in stats.values())}.",
              f"- Seconds per call: {mean([e for s in stats.values() for e in s['elapsed']])} here, "
              f"{mean([e for s in stats.values() for e in s['elapsed_base']])} in the run for the same boxes.", "",
              "## Decisions by group", "", "| group | run (hinted) | this variant |", "|---|---|---|"]
    for g in GROUPS:
        fmt = lambda c: ", ".join(f"{k}: {v}" for k, v in c.most_common()) or "—"  # noqa: E731
        lines.append(f"| {g} | {fmt(stats[g]['base'])} | {fmt(stats[g]['variant'])} |")
    lines += ["", "Each call is in the CSV with both decisions, both brands and the variant's reasoning."]
    return "\n".join(lines) + "\n"


def write_report(ctx: RunContext, tag: str, overrides: dict, groups: dict, pools: dict, hinted: dict,
                 matched: list[dict]) -> Path:
    records = {r["det_id"]: r for r in ctx.read_jsonl(f"vlm/ab_{tag}.jsonl")}
    gold = {int(m["det_id"]): m["gold_class"] for m in matched if m.get("det_id") not in ("", None)}
    stats, rows = summarise(records, groups, hinted, gold, ctx.brands, ctx.cfg.canonicalisation)
    out = ctx.path("reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vlm_ab_{tag}.md"
    path.write_text(render(ctx.run_dir, tag, overrides, stats, pools), encoding="utf-8")
    with open(out / f"vlm_ab_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["det_id", "group", "gold_class", "run_is_logo", "variant_is_logo", "run_brand", "variant_brand",
                    "variant_brand_canonical", "variant_brand_correct", "variant_confidence", "elapsed_s",
                    "variant_reasoning"])
        w.writerows(rows)
    return path

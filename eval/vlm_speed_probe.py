#!/usr/bin/env python3
"""19 A12 re-run cost: where a VLM call's time goes on this machine.

Loads the model as the ``vlm`` stage does (the run's own settings plus any
``--set`` overrides, as ``vlm-ab``), then for ``--n`` detections of the run
times three phases separately:

* Preprocessing: chat template, image fetch and resize, processor (CPU);
* Prefill: ``generate`` with ``max_new_tokens=1`` (the image and prompt pass);
* the full ``generate``, whose time beyond prefill is decoding, reported as
  new tokens per second.

It also records:

* where the model's modules sit (``hf_device_map``);
* a bare fp16 matmul (what the GPU does when it is fed) and the cost of one
  small kernel launch (the per-token overhead a 4-bit decode pays hundreds of
  times per token);
* a pure-Python loop (CPU speed);
* CPU, GPU utilisation, power and SM clock sampled through the calls.

Writes ``reports/vlm_speed_probe_<label>.md``. Nothing in the run is changed.

Usage::

    python eval/vlm_speed_probe.py --run runs/<video_id>/<run_id> [--n 5] [--set KEY=VALUE ...] [--label x]
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runio import md_table, reports_dir  # noqa: E402

SMI_FIELDS = ("utilization.gpu", "power.draw", "clocks.sm", "clocks.max.sm", "pstate")


def parse_smi(line: str) -> dict | None:
    """One ``nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits`` line."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != len(SMI_FIELDS):
        return None
    out = {}
    for k, v in zip(SMI_FIELDS, parts):
        try:
            out[k] = v if k == "pstate" else float(v)
        except ValueError:
            return None
    return out


class Sampler:
    """CPU (system and this process) and ``nvidia-smi`` readings every ``period`` seconds, in a thread."""

    def __init__(self, period: float = 1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            import psutil
            proc = psutil.Process()
            psutil.cpu_percent(None)
            proc.cpu_percent(None)
        except Exception:  # noqa: BLE001 - psutil is optional
            psutil = proc = None
        while not self._stop.wait(self.period):
            s = {}
            if psutil is not None:
                s["cpu_system"] = psutil.cpu_percent(None)
                s["cpu_process"] = proc.cpu_percent(None)
            try:
                line = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                      timeout=5).stdout.splitlines()
                if line:
                    s.update(parse_smi(line[0]) or {})
            except Exception:  # noqa: BLE001 - no nvidia-smi (e.g. the Mac)
                pass
            self.samples.append(s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


def summarise_calls(calls: list[dict]) -> dict:
    """Medians over the timed calls; decode rate from (new tokens − 1) over (full − prefill)."""
    def med(key):
        v = [c[key] for c in calls if c.get(key) is not None]
        return statistics.median(v) if v else None
    rates = [(c["new_tokens"] - 1) / (c["full_s"] - c["prefill_s"]) for c in calls
             if c["new_tokens"] > 1 and c["full_s"] > c["prefill_s"]]
    return {"n": len(calls), "preprocess_s": med("preprocess_s"), "prefill_s": med("prefill_s"),
            "full_s": med("full_s"), "input_tokens": med("input_tokens"), "image_tokens": med("image_tokens"),
            "new_tokens": med("new_tokens"), "decode_tok_s": statistics.median(rates) if rates else None,
            "per_call_s": med("total_s")}


def summarise_samples(samples: list[dict]) -> dict:
    out = {}
    for k in ("cpu_system", "cpu_process", "utilization.gpu", "power.draw", "clocks.sm", "clocks.max.sm"):
        v = [s[k] for s in samples if k in s]
        out[k] = (statistics.median(v), max(v)) if v else None
    out["pstate"] = Counter(s["pstate"] for s in samples if "pstate" in s).most_common()
    return out


def micro_benchmarks() -> dict:
    """fp16 matmul TFLOPS, µs per small kernel launch, and a pure-Python loop."""
    import torch
    out = {}
    t = time.perf_counter()
    acc = 0
    for i in range(10_000_000):
        acc += i
    out["python_loop_s"] = time.perf_counter() - t
    if not torch.cuda.is_available():
        return out
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    out["matmul_tflops"] = 20 * 2 * n ** 3 / (time.perf_counter() - t) / 1e12
    x = torch.zeros(16, device="cuda")
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5000):
        x.add_(1)
    torch.cuda.synchronize()
    out["launch_us"] = (time.perf_counter() - t) / 5000 * 1e6
    del a, b, x
    torch.cuda.empty_cache()
    return out


def environment(model) -> dict:
    import importlib.metadata as md

    import torch

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return "—"
    dm = getattr(model, "hf_device_map", None) or {}
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "torch cuda": torch.version.cuda, "transformers": ver("transformers"), "bitsandbytes": ver("bitsandbytes"),
            "accelerate": ver("accelerate"), "qwen-vl-utils": ver("qwen-vl-utils"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "logical CPUs": __import__("os").cpu_count(), "torch threads": torch.get_num_threads(),
            "model dtype": str(model.dtype), "attention": getattr(model.config, "_attn_implementation", "—"),
            "device map": ", ".join(f"{d}: {n} modules" for d, n in Counter(map(str, dm.values())).most_common())
                          or "single device"}


def time_call(model, processor, content: list[dict], vcfg) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    from logo_pipeline.stages.vlm import build_messages

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    messages = build_messages(content)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                       return_tensors="pt").to(model.device, model.dtype)
    sync()
    t1 = time.perf_counter()
    greedy = vcfg.decoding == "greedy"
    kw = {"do_sample": not greedy, **({"temperature": None} if greedy else {})}
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, **kw)
    sync()
    t2 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=vcfg.max_new_tokens, **kw)
    sync()
    t3 = time.perf_counter()
    image_id = getattr(model.config, "image_token_id", None)
    n_in = int(inputs.input_ids.shape[1])
    return {"preprocess_s": t1 - t0, "prefill_s": t2 - t1, "full_s": t3 - t2, "total_s": (t1 - t0) + (t3 - t2),
            "input_tokens": n_in, "new_tokens": int(out.shape[1]) - n_in,
            "image_tokens": int((inputs.input_ids == image_id).sum()) if image_id is not None else None}


def render(run_dir: Path, label: str, sets: list[str], env: dict, load_s: float, bench: dict, calls: list[dict],
           summary: dict, load: dict) -> str:
    def f(x, nd=1):
        return "—" if x is None else f"{x:.{nd}f}"

    def mm(pair, nd=0):
        return "—" if pair is None else f"{pair[0]:.{nd}f} / {pair[1]:.{nd}f}"

    lines = [f"# VLM speed probe `{label}` — `{run_dir}`", "",
             f"Overrides: `{' '.join(sets) or 'none'}`. Model load: {load_s:.0f} s.", ""]
    lines += md_table(["item", "value"], list(env.items()))
    lines += ["", "## Per call (median of the timed calls)", ""]
    lines += md_table(["preprocess s", "prefill s", "decode tok/s", "new tokens", "input tokens", "image tokens",
                       "per call s"],
                      [(f(summary["preprocess_s"], 2), f(summary["prefill_s"], 2), f(summary["decode_tok_s"]),
                        f(summary["new_tokens"], 0), f(summary["input_tokens"], 0), f(summary["image_tokens"], 0),
                        f(summary["per_call_s"]))])
    lines += [""]
    lines += md_table(["det_id", "preprocess s", "prefill s", "full generate s", "new tokens"],
                      [(c["det_id"], f(c["preprocess_s"], 2), f(c["prefill_s"], 2), f(c["full_s"]), c["new_tokens"])
                       for c in calls])
    lines += ["", "## Machine during the calls (median / max)", ""]
    lines += md_table(["reading", "value"],
                      [("system CPU %", mm(load["cpu_system"])), ("this process CPU % (100 = one core)",
                                                                  mm(load["cpu_process"])),
                       ("GPU utilisation %", mm(load["utilization.gpu"])), ("power W", mm(load["power.draw"])),
                       ("SM clock MHz (max possible)", f"{mm(load['clocks.sm'])} ({mm(load['clocks.max.sm'])})"),
                       ("performance state", ", ".join(f"{p}: {n}" for p, n in load["pstate"]) or "—")])
    lines += ["", "## Bare hardware", ""]
    lines += md_table(["test", "value", "healthy RTX 4090, for scale"],
                      [("fp16 matmul 8192², TFLOPS", f(bench.get("matmul_tflops")), "about 150"),
                       ("one small kernel launch, µs", f(bench.get("launch_us")), "about 5–15 under WDDM"),
                       ("pure-Python loop of 10⁷ additions, s", f(bench.get("python_loop_s"), 2),
                        "about 0.3–0.6 on a current desktop CPU")])
    return "\n".join(lines) + "\n"


def pick_det_ids(reps: dict, n: int) -> list[int]:
    ids = sorted(reps)
    if len(ids) <= n:
        return ids
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=5, help="detections to time (one more is used to warm up)")
    ap.add_argument("--set", action="append", default=[], dest="sets", metavar="KEY=VALUE")
    ap.add_argument("--label", default=None, help="names reports/vlm_speed_probe_<label>.md (default: default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from logo_pipeline.stages.vlm import DetectionInputs, _get_model
    from logo_pipeline.vlm_ab import open_ab_context, parse_sets

    run_dir, repo_root = Path(args.run), Path(__file__).resolve().parents[1]
    label = args.label or "default"
    ctx = open_ab_context(run_dir, parse_sets(args.sets), repo_root)
    vcfg = ctx.cfg.vlm
    bench = micro_benchmarks()
    print(f"bare hardware: {bench}")
    t = time.perf_counter()
    model, processor = _get_model(vcfg.model_id, vcfg.quantisation, vcfg.max_pixels)
    load_s = time.perf_counter() - t
    env = environment(model)
    print(f"environment: {env}")
    inp = DetectionInputs(ctx)
    ids = pick_det_ids(inp.reps, args.n + 1)

    def content(det_id):
        return inp.content(det_id, ctx.open_frame_image(inp.reps[det_id]["frame_idx"]), blind=False)[0]

    time_call(model, processor, content(ids[0]), vcfg)                         # warm-up, not counted
    calls = []
    with Sampler() as sampler:
        for i, det_id in enumerate(ids[1:], 1):
            c = dict(time_call(model, processor, content(det_id), vcfg), det_id=det_id)
            calls.append(c)
            print(f"{i}/{len(ids) - 1} det {det_id}: preprocess {c['preprocess_s']:.2f} s, prefill "
                  f"{c['prefill_s']:.2f} s, generate {c['full_s']:.1f} s, {c['new_tokens']} new tokens")
    summary, load = summarise_calls(calls), summarise_samples(sampler.samples)
    out = reports_dir(run_dir, args.out) / f"vlm_speed_probe_{label}.md"
    out.write_text(render(run_dir, label, args.sets, env, load_s, bench, calls, summary, load), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

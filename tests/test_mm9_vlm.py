"""MM9 (spec docs/15): blind prompt (T67/T33), paired passes and --blind-only,
closed-set scoring (T30/T31) with a tiny fake model, the spike verdict, the
scores reaching the router, and the T32/T36 reports."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from helpers import make_ctx  # noqa: E402
from mm5_helpers import make_cfg, route, row  # noqa: E402
from test_mm1_vlm_stage import _prep  # noqa: E402

import generate_hints as gh
from logo_pipeline.stages.router.inputs import MissingRouterInput
from logo_pipeline.stages.vlm import VlmStage
from logo_pipeline.stages.vlm_scoring import (derive_scores, label_spans, prefix_from_reply, score_labels,
                                              spike_sample, spike_verdict)

HINT_HEADERS = ("## Signal: audio_brands", "## Signal: faiss_matches", "## Signal: ocr_text")
ALL_SIGNALS = set(gh.CANONICAL_SIGNALS)


def _content(blind: bool) -> str:
    content = gh._build_vlm_content(
        crop_img=None, enlarged_img=None, frame_img=None, temporal_pairs=[],
        detection={"score": 0.5, "timecode_seconds": 1.0}, audio_brands=["verisure"],
        faiss_hits=[{"brand_name": "rai", "similarity": 0.9}],
        ocr_data={"source": "tight_crop", "lines": [{"text": "RAI", "confidence": 0.9}]},
        signals=ALL_SIGNALS, brand_mode="open", channel_watermark="rai sport", blind=blind)
    return "".join(c["text"] for c in content if c["type"] == "text")


# -- T67 / T33 blind prompt ----------------------------------------------------------

def test_blind_prompt_renders_no_hint_sections_and_no_watermark_line():
    text = _content(blind=True)
    for h in HINT_HEADERS:
        assert h not in text
    assert "Known context: the channel watermark" not in text
    assert "hint" not in text.lower()
    assert "Base your answer ONLY on the images shown above" in text
    assert '"audio_brands"' not in text            # signal vocabulary is image signals only
    assert '"UNKNOWN"' in text                     # T67 option kept


def test_hinted_prompt_unchanged():
    text = _content(blind=False)
    for h in HINT_HEADERS:
        assert h in text
    assert "Known context: the channel watermark" in text
    assert "Base your answer ONLY on the signals shown above" in text
    assert "soft evidence" in text


# -- T33 paired passes ---------------------------------------------------------------

REPLY = ('{"is_logo": "logo", "brand": "rai", "is_truncated": false, "logo_probability": 90, '
         '"confidence": 80, "signals_used": ["tight_crop"], "reasoning": "x"}')


def _mocks(reply=REPLY):
    model = MagicMock(device="cpu", dtype=None)
    model.generate.return_value = [[1, 2, 3]]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"
    processor.batch_decode.return_value = [reply]
    inputs = MagicMock(input_ids=[[1]])
    inputs.to.return_value = inputs
    processor.return_value = inputs
    calls = []
    real = gh._build_vlm_content

    def spy(*a, **kw):
        calls.append(kw.get("blind", False))
        return real(*a, **kw)
    return [patch("logo_pipeline.stages.vlm._get_model", return_value=(model, processor)),
            patch("qwen_vl_utils.process_vision_info", return_value=([], [])),
            patch("generate_hints._build_vlm_content", side_effect=spy)], calls


def _run_vlm(ctx, **opts):
    patches, calls = _mocks()
    for p in patches:
        p.start()
    try:
        ctx.options.update(opts)
        result = VlmStage().run(ctx)
    finally:
        for p in patches:
            p.stop()
    return result, calls


def test_both_passes_write_paired_records(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    _prep(ctx)
    result, calls = _run_vlm(ctx)
    assert result.counts == {"calls": 1, "blind_calls": 1}
    assert sorted(calls) == [False, True]
    blind = ctx.read_jsonl("vlm/blind.jsonl")
    assert len(blind) == 1 and blind[0]["parsed"]["brand"] == "rai"
    assert set(blind[0]["enabled_signals"]) <= {"tight_crop", "enlarged_crop", "full_frame", "temporal_context"}
    assert "elapsed_s" in blind[0] and "elapsed_s" in ctx.read_jsonl("vlm/hinted.jsonl")[0]
    comp = ctx.read_jsonl("vlm/hint_composition.jsonl")[0]
    assert comp["mode"] == "open" and "signals_rendered" in comp
    assert VlmStage().outputs_exist(ctx)


def test_blind_only_adds_the_blind_pass_without_repeating_hinted(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, overrides={"vlm": {"blind_pass": False}})
    _prep(ctx)
    _run_vlm(ctx)                                    # checkpoint-style run: hinted only
    assert not ctx.path("vlm/blind.jsonl").exists()
    ctx2 = make_ctx(tmp_path, repo_root, n_frames=1)  # same run dir, blind_pass back on
    assert not VlmStage().outputs_exist(ctx2)
    result, calls = _run_vlm(ctx2, vlm_pass="blind")
    assert result.counts == {"calls": 0, "blind_calls": 1} and calls == [True]
    assert len(ctx2.read_jsonl("vlm/hinted.jsonl")) == 1


def test_blind_only_requires_blind_pass_enabled(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, overrides={"vlm": {"blind_pass": False}})
    _prep(ctx)
    with pytest.raises(ValueError):
        _run_vlm(ctx, vlm_pass="blind")


def test_passes_resume_per_det_id(tmp_path, repo_root):
    ctx = make_ctx(tmp_path, repo_root, n_frames=1)
    _prep(ctx)
    _run_vlm(ctx)
    result, calls = _run_vlm(ctx)
    assert result.counts["calls"] == 0 and calls == []


# -- T30 scoring maths -----------------------------------------------------------------

def test_label_spans_score_every_label_from_the_same_position():
    ctx_ids = [1, 2, 3, 9]                      # context ends with a token the labels re-tokenise
    labels = [[1, 2, 3, 20, 21], [1, 2, 3, 30], [1, 2, 3, 9, 40]]
    k, spans = label_spans(ctx_ids, labels, "sequence_logprob")
    assert k == 3 and spans == [(3, 5), (3, 4), (3, 5)]


def test_label_spans_first_token_is_the_first_divergence():
    labels = [[1, 2, 7, 20], [1, 2, 7, 30], [1, 2, 7, 40, 41]]
    k, spans = label_spans([1, 2], labels, "first_token_softmax")
    assert k == 2 and spans == [(3, 4), (3, 4), (3, 4)]


def test_derive_scores():
    d = derive_scores(["logo", "partial_logo", "not_logo", "uncertain"], [0.0, 0.0, math.log(2), -50.0])
    assert math.isclose(sum(d["decision_scores"].values()), 1.0, abs_tol=1e-5)
    assert math.isclose(d["not_logo_confidence"], 0.5, abs_tol=1e-4)
    assert math.isclose(d["logo_probability_scored"], 50.0, abs_tol=1e-2)


def test_prefix_from_reply():
    assert prefix_from_reply('{\n  "is_logo": "logo", "brand": null}') == '{\n  "is_logo": "'
    assert prefix_from_reply("garbage") == '{"is_logo": "'
    assert prefix_from_reply(None) == '{"is_logo": "'


class _Batch(dict):
    def to(self, *a, **k):
        return self


class _CharProcessor:
    """A character-level stand-in: token id = ord(char), left padding with id 0."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT>"

    def __call__(self, text, images=None, videos=None, padding=False, return_tensors="pt"):
        seqs = [[ord(c) for c in t] for t in text]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([[0] * (n - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs])
        return _Batch(input_ids=ids, attention_mask=mask)


class _TeacherModel:
    """Predicts the true next token confidently on rows whose text contains ``favour``; uniform elsewhere."""
    device = "cpu"

    def __init__(self, favour: str, honour_keep: bool):
        self.favour, self.honour_keep = favour, honour_keep

    def __call__(self, input_ids, attention_mask, logits_to_keep=None, **kw):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, 256)
        for i in range(b):
            if self.favour in "".join(chr(int(c)) for c in input_ids[i] if int(c)):
                for p in range(t - 1):
                    logits[i, p, int(input_ids[i, p + 1])] = 20.0
        if self.honour_keep and logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return MagicMock(logits=logits)


@pytest.mark.parametrize("honour_keep", [True, False])
def test_score_labels_prefers_the_teacher_forced_label(honour_keep):
    labels = ["logo", "partial_logo", "not_logo", "uncertain"]
    with patch("qwen_vl_utils.process_vision_info", return_value=(None, None)):
        raw = score_labels(_TeacherModel("not_logo", honour_keep), _CharProcessor(), [{"role": "user"}],
                           '{"is_logo": "', labels, "sequence_logprob")
    d = derive_scores(labels, raw)
    assert max(d["decision_scores"], key=d["decision_scores"].get) == "not_logo"
    assert d["not_logo_confidence"] > 0.99


# -- T30 spike verdict ------------------------------------------------------------------

def _hinted(labels):
    return {i: {"parsed": {"is_logo": lab} if lab else None} for i, lab in enumerate(labels)}


def test_spike_sample_spans_strata_and_is_deterministic():
    hinted = _hinted(["not_logo"] * 40 + ["logo"] * 5 + ["partial_logo"] * 3 + [None] * 2)
    s = spike_sample(hinted, 12, seed=42)
    kinds = {((hinted[d]["parsed"] or {}).get("is_logo")) for d in s}
    assert kinds == {"not_logo", "logo", "partial_logo", None}
    assert s == spike_sample(hinted, 12, seed=42) and len(s) == 12


def test_spike_verdict_pass_and_fail():
    hinted = _hinted(["logo"] * 25 + ["not_logo"] * 25)
    good = [{"det_id": i, "decision_scores": {"logo": 0.8 if i < 25 else 0.1, "partial_logo": 0.05,
                                              "not_logo": 0.1 if i < 25 else 0.85, "uncertain": 0.05},
             "logo_probability_scored": 50 + i * 0.7} for i in range(50)]
    v = spike_verdict(good, hinted, 50)
    assert v["passed"], v
    flat = [dict(r, logo_probability_scored=95.0) for r in good]
    assert not spike_verdict(flat, hinted, 50)["passed"]          # too few distinct values
    assert not spike_verdict(good[:40], hinted, 50)["passed"]     # not all scored


# -- T31 scores reach the router --------------------------------------------------------

def test_scored_source_reads_t31_field():
    on = {"vlm": {"scoring": {"enabled": True}}, "router": {"not_logo_confidence_source": "scored"}}
    outs, _ = route([row(vlm_decision="not_logo", canonical_brand=None, confidence=10, not_logo_confidence=0.9)], on)
    assert outs[0]["reason_code"] == "high_confidence_not_logo"     # generated value (0.10) would not reject


def test_scored_source_requires_scoring_enabled():
    from logo_pipeline.config import ConfigError
    with pytest.raises(ConfigError):
        make_cfg({"router": {"not_logo_confidence_source": "scored"}})


def test_router_stage_refuses_scored_without_scores_file(tmp_path, repo_root):
    from logo_pipeline.stages.router.engine import RouterStage
    ctx = make_ctx(tmp_path, repo_root, overrides={"vlm": {"scoring": {"enabled": True}},
                                                    "router": {"not_logo_confidence_source": "scored"}})
    ctx.write_jsonl("router_input.jsonl", [])
    with pytest.raises(MissingRouterInput):
        RouterStage().run(ctx)


def test_router_input_carries_scores(tmp_path, repo_root):
    from logo_pipeline.stages.canonicalise import CanonicaliseStage
    from logo_pipeline.stages.router_input import RouterInputStage
    from logo_pipeline.stages.watermark import WatermarkStage
    ctx = make_ctx(tmp_path, repo_root, n_frames=1, overrides={"vlm": {"scoring": {"enabled": True}}})
    _prep(ctx)
    ctx.write_jsonl("vlm/hinted.jsonl", [{"det_id": 0, "parsed": {"is_logo": "logo"}, "vlm_parse_ok": True}])
    ctx.write_jsonl("vlm/scores.jsonl", [{"det_id": 0, "decision_scores": {"logo": 0.7, "not_logo": 0.3},
                                          "logo_probability_scored": 70.0, "not_logo_confidence": 0.3}])
    for st in (CanonicaliseStage(), WatermarkStage(), RouterInputStage()):
        st.run(ctx)
    r = ctx.read_jsonl("router_input.jsonl")[0]
    assert (r["logo_probability_scored"], r["not_logo_confidence"]) == (70.0, 0.3)


# -- T32 / T36 reports ----------------------------------------------------------------

def test_score_distribution_pass_and_fail():
    import score_distribution as sd
    hinted = {i: {"parsed": {"is_logo": "logo" if i % 2 else "not_logo", "logo_probability": 95}} for i in range(80)}
    spread = [{"det_id": i, "logo_probability_scored": float(i), "not_logo_confidence": 1 - i / 100} for i in range(80)]
    assert sd.analyse(spread, hinted)["passed"]
    clumped = [{"det_id": i, "logo_probability_scored": 99.5, "not_logo_confidence": 0.01} for i in range(80)]
    a = sd.analyse(clumped, hinted)
    assert not a["passed"] and a["generated_distinct"] == 1
    assert "PASS" in sd.render(Path("r"), sd.analyse(spread, hinted))


def test_vlm_cost_report(tmp_path, repo_root):
    import vlm_cost
    ctx = make_ctx(tmp_path, repo_root, n_frames=2)
    ctx.write_jsonl("vlm/hinted.jsonl", [{"det_id": i, "elapsed_s": 10.0} for i in range(4)])
    ctx.write_jsonl("vlm/blind.jsonl", [{"det_id": i, "elapsed_s": 6.0} for i in range(4)])
    ctx.write_jsonl("merged_detections.jsonl", [{"det_id": i} for i in range(4)])
    assert vlm_cost.main(["--run", str(ctx.run_dir)]) == 0
    text = (ctx.run_dir / "reports" / "vlm_cost.md").read_text()
    assert "| hinted call | 4 | 10.00" in text and "| this run | 4 |" in text

# Copyright 2026 Pedro (causal_eval experiment)
# SPDX-License-Identifier: Apache-2.0
"""Tests for `causal_eval/causal_sweep.py` (causal validation pipeline v2).

Built incrementally alongside the pipeline. This is the FIRST test: it covers
Fase 0 candidate selection (`position_candidates`) — that the selection
returns a well-formed set of exactly `k` distinct in-range vocabulary tokens.

Like `test_interventions.py`, these run against `jlens`'s `TinyDecoder` and
prove only *mechanical* correctness — nothing semantic. `TinyDecoder` has
untrained weights, so "the right concepts are selected" is not testable here;
that needs the trained guardrail.

pytest is not installed in `guardrail_eval/.venv` (the shared venv this
sub-project reuses), so this file is a self-contained script:

    guardrail_eval/.venv/Scripts/python.exe causal_eval/tests/test_causal_sweep.py

(it is also importable/collectable by pytest where available).
"""

from __future__ import annotations

import os
import sys

import torch

# `TinyDecoder` lives in the repo-root `tests/` package (not shipped in the
# jlens wheel), so put the repo root on the path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# And the causal_eval dir, so `causal_sweep`/`interventions` import whether run
# from the repo root or from inside causal_eval/.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from jlens.fitting import fit  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from tests.tiny import TinyDecoder  # noqa: E402

from causal_sweep import _kl_divergence, position_candidates, sweep_positions  # noqa: E402
from interventions import lens_vectors  # noqa: E402

_PROMPT = "abcdefghij " * 5  # > SKIP_FIRST_N_POSITIONS tokens
_BAND = [0, 1, 2]  # TinyDecoder's fitted layers stand in for the workspace band


def _model_lens_residuals():
    """A frozen TinyDecoder, a lens fitted on `_BAND`, and the band residuals
    at every position for `_PROMPT`."""
    torch.manual_seed(0)
    model = TinyDecoder(n_layers=4, d_model=8)
    for p in model.parameters():
        p.requires_grad_(False)
    lens = fit(model, [_PROMPT, "klmnopqrst " * 5], source_layers=_BAND, dim_batch=4)
    input_ids = model.encode(_PROMPT)
    with ActivationRecorder(model.layers, at=_BAND) as rec:
        model.forward(input_ids)
        acts = {layer: rec.activations[layer].detach() for layer in _BAND}
    return model, lens, acts


def test_candidates_are_k_tokens():
    """position_candidates returns exactly k distinct, in-range token ids."""
    model, lens, acts = _model_lens_residuals()
    position = 20
    residual_by_layer = {layer: acts[layer][0, position] for layer in _BAND}

    cands = position_candidates(lens, model.lm_head.weight, residual_by_layer, k=10)

    vocab = model.lm_head.out_features  # 32
    assert isinstance(cands, list)
    assert len(cands) == 10  # exactly k
    assert all(isinstance(t, int) for t in cands)  # plain python ints
    assert len(set(cands)) == 10  # no duplicates
    assert all(0 <= t < vocab for t in cands)  # in vocabulary range


def test_position_candidates_reuses_precomputed_V_by_layer():
    """Passing V_by_layer must give identical results to internal
    recomputation -- this is the fix for the redundant lens_vectors
    recomputation found to dominate real-guardrail wall-clock cost (~13x the
    cost of the actual forward passes, per ARCHITECTURE.md's "First
    real-guardrail run and three findings", Finding 2). sweep_positions now
    passes its precomputed V_all here instead of letting each position
    recompute lens_vectors from scratch."""
    model, lens, acts = _model_lens_residuals()
    residual_by_layer = {layer: acts[layer][0, 20] for layer in _BAND}
    W_U = model.lm_head.weight

    baseline = position_candidates(lens, W_U, residual_by_layer, k=10)

    V_by_layer = {layer: lens_vectors(lens, W_U, layer) for layer in _BAND}
    with_precomputed = position_candidates(
        lens, W_U, residual_by_layer, k=10, V_by_layer=V_by_layer
    )
    assert with_precomputed == baseline


def test_exclude_removes_given_tokens_and_backfills():
    """The anti-confound guard mechanism (§3.5.2): tokens passed via `exclude`
    never appear in the result, and the next-highest-scoring tokens take their
    place so the result is still exactly `k` long."""
    model, lens, acts = _model_lens_residuals()
    residual_by_layer = {layer: acts[layer][0, 20] for layer in _BAND}

    baseline = position_candidates(lens, model.lm_head.weight, residual_by_layer, k=10)
    to_exclude = baseline[:3]  # the 3 highest-scoring tokens, by construction

    filtered = position_candidates(
        lens, model.lm_head.weight, residual_by_layer, k=10, exclude=to_exclude
    )
    assert len(filtered) == 10  # still exactly k, backfilled
    assert not set(to_exclude) & set(filtered)  # excluded tokens gone
    assert set(baseline[3:]).issubset(set(filtered))  # next-best took their place


def test_exclude_raises_when_k_exceeds_selectable():
    """k must not silently shrink: if exclusion leaves fewer than k tokens
    selectable, position_candidates raises rather than returning a short list."""
    model, lens, acts = _model_lens_residuals()
    residual_by_layer = {layer: acts[layer][0, 20] for layer in _BAND}
    vocab = model.lm_head.out_features  # 32
    k = 10
    exclude_all_but_k_minus_1 = list(range(vocab - (k - 1)))  # leaves k-1 selectable
    try:
        position_candidates(
            lens, model.lm_head.weight, residual_by_layer,
            k=k, exclude=exclude_all_but_k_minus_1,
        )
        raise AssertionError("expected ValueError, none raised")
    except ValueError:
        pass


def test_kl_divergence_is_zero_for_identical_logits():
    torch.manual_seed(40)
    logits = torch.randn(32)
    assert abs(_kl_divergence(logits, logits)) < 1e-5


def test_kl_divergence_is_nonnegative_and_asymmetric():
    torch.manual_seed(41)
    p, q = torch.randn(32), torch.randn(32)
    kl_pq = _kl_divergence(p, q)
    kl_qp = _kl_divergence(q, p)
    assert kl_pq >= -1e-6 and kl_qp >= -1e-6  # KL >= 0 (float slack)
    assert kl_pq != kl_qp  # KL is not symmetric in general


def _tiny_sweep_callables(model):
    """run_forward + a fixed-token score_fn for TinyDecoder (no malign/benign
    concept exists, so we attribute the probability of an arbitrary token)."""

    def run_forward(input_ids):
        hidden = model.forward(input_ids).last_hidden_state
        return model.unembed(hidden)[0]  # [seq_len, vocab]

    target_token = 5

    def score_fn(logits):
        return float(torch.softmax(logits.float(), dim=-1)[target_token])

    return run_forward, score_fn


def test_sweep_returns_one_signed_record_per_swept_position():
    """sweep_positions covers [p_start, p_end), respects the input boundary,
    and each record's `nota` equals score_ablate − score_control exactly."""
    model, lens, _ = _model_lens_residuals()
    input_ids = model.encode(_PROMPT)
    seq_len = input_ids.shape[1]
    run_forward, score_fn = _tiny_sweep_callables(model)

    p_start = 16  # SKIP_FIRST_N_POSITIONS
    records = sweep_positions(
        model.layers, lens, model.lm_head.weight, input_ids,
        run_forward, score_fn, layers=_BAND, k=6, p_start=p_start, seed=0,
    )

    positions = [r["position"] for r in records]
    assert positions == list(range(p_start, seq_len))  # covers the range, in order
    assert all(p < seq_len for p in positions)  # never a generated position
    for r in records:
        assert r["nota"] == r["score_ablate"] - r["score_control"]  # signed score
        for key in ("score_clean", "score_ablate", "score_control"):
            assert 0.0 <= r[key] <= 1.0  # probabilities
        assert r["n_candidates"] == 6
        # KL(clean ‖ intervened) — §A.6's own causal metric, always >= 0
        for key in ("kl_ablate", "kl_control"):
            assert torch.isfinite(torch.tensor(r[key]))
            assert r[key] >= -1e-5  # KL divergence is non-negative (float slack)


def test_sweep_respects_explicit_position_window():
    """p_start/p_end bound the sweep; p_end is clamped to the input length."""
    model, lens, _ = _model_lens_residuals()
    input_ids = model.encode(_PROMPT)
    seq_len = input_ids.shape[1]
    run_forward, score_fn = _tiny_sweep_callables(model)

    records = sweep_positions(
        model.layers, lens, model.lm_head.weight, input_ids,
        run_forward, score_fn, layers=_BAND, k=6,
        p_start=18, p_end=seq_len + 50,  # deliberately past the end
    )
    positions = [r["position"] for r in records]
    assert positions == list(range(18, seq_len))  # clamped to seq_len, not seq_len+50


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

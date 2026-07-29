# Copyright 2026 Pedro (guardrail_eval experiment)
# SPDX-License-Identifier: Apache-2.0
"""Invariant tests for `causal_eval/interventions.py` against `TinyDecoder`.

What these tests PROVE: the steer/ablate/swap primitives and the write-capable
hook are *algebraically correct and correctly plumbed* into `jlens` — swap
preserves the orthogonal complement and truly exchanges the two lens
coordinates, ablation zeroes the projection, steering moves by exactly ``α·v``,
and the hook actually modifies the forward pass and is cleanly removed.

What they DO NOT prove: anything semantic or causal. `TinyDecoder` has
untrained weights and no interpretable vocabulary, so "swap Soccer→Rugby flips
the output" cannot be shown here — that validation needs the trained Qwen3-1.7B
guardrail. These are unit invariants, not the experiment.

`TinyDecoder`'s residual blocks are exactly linear (``h + 0.1·W·h``), so its
fitted ``J_l`` is exact; that keeps the invariants clean and independent of any
estimator noise.

pytest is not installed in `guardrail_eval/.venv` (the shared venv this
sub-project reuses), so this file is a self-contained script: run it with

    guardrail_eval/.venv/Scripts/python.exe causal_eval/tests/test_interventions.py

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
# And the causal_eval dir, so `interventions` imports whether run from the
# repo root or from inside causal_eval/.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from jlens.fitting import fit  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from tests.tiny import TinyDecoder  # noqa: E402

from interventions import (  # noqa: E402
    InterventionHook,
    ablate,
    ablate_span,
    lens_vector,
    lens_vectors,
    matched_norm_control,
    steer,
    swap,
)

ATOL = 1e-5
_PROMPT = "abcdefghij " * 5  # > SKIP_FIRST_N_POSITIONS tokens


def _model_and_lens():
    torch.manual_seed(0)
    model = TinyDecoder(n_layers=4, d_model=8)
    for p in model.parameters():
        p.requires_grad_(False)
    lens = fit(model, [_PROMPT, "klmnopqrst " * 5], source_layers=[0, 1, 2], dim_batch=4)
    return model, lens


# --- pure residual-space edits -------------------------------------------------


def test_steer_moves_by_exactly_alpha_v():
    torch.manual_seed(1)
    h = torch.randn(3, 8)
    v = torch.randn(8)
    torch.testing.assert_close(steer(h, v, alpha=2.5) - h, 2.5 * v.expand_as(h))


def test_ablate_zeroes_projection_and_is_idempotent():
    torch.manual_seed(2)
    h = torch.randn(4, 8)
    v = torch.randn(8)
    h1 = ablate(h, v)
    # projection onto v is gone
    assert (h1 @ _unit_ref(v)).abs().max() < ATOL
    # removed component lies along v (orthogonal complement untouched)
    removed = h - h1
    torch.testing.assert_close(removed, (removed @ _unit_ref(v)).unsqueeze(-1) * _unit_ref(v))
    # idempotent
    torch.testing.assert_close(ablate(h1, v), h1)


def test_ablate_span_zeroes_projection_onto_all_vectors():
    torch.manual_seed(3)
    h = torch.randn(5, 8)
    V = torch.randn(8, 3)  # 3 non-orthogonal "active" lens vectors
    h1 = ablate_span(h, V)
    # h1 is orthogonal to every column of V
    assert (h1 @ V).abs().max() < 1e-4


def test_matched_norm_control_matches_ablation_norm():
    """The control perturbs h by exactly the norm ablate_span removes, but
    along a random subspace — the fair baseline for "did *these* vectors
    matter vs. any equal-sized perturbation" (§3.5.2 / §A.23)."""
    torch.manual_seed(30)
    h = torch.randn(4, 8)
    V = torch.randn(8, 3)
    gen = torch.Generator().manual_seed(1)

    control = matched_norm_control(h, V, generator=gen)
    # per-row: ‖h − control‖ == ‖h − ablate_span(h, V)‖
    ablate_norm = (h - ablate_span(h, V)).norm(dim=-1)
    control_norm = (h - control).norm(dim=-1)
    torch.testing.assert_close(control_norm, ablate_norm, atol=1e-4, rtol=0)
    # but the control is NOT the ablation (different, random, direction)
    assert (control - ablate_span(h, V)).norm() > 1e-2


def test_matched_norm_control_is_reproducible_with_seed():
    torch.manual_seed(31)
    h = torch.randn(4, 8)
    V = torch.randn(8, 3)
    a = matched_norm_control(h, V, generator=torch.Generator().manual_seed(7))
    b = matched_norm_control(h, V, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_swap_exchanges_the_two_lens_coordinates():
    torch.manual_seed(4)
    h = torch.randn(6, 8)
    v_s, v_t = torch.randn(8), torch.randn(8)
    V = torch.stack([v_s, v_t], dim=-1)
    Vpinv = torch.linalg.pinv(V)
    c = h @ Vpinv.T  # [.., 2] before
    h2 = swap(h, v_s, v_t)
    c2 = h2 @ Vpinv.T  # [.., 2] after
    torch.testing.assert_close(c2, c.flip(-1), atol=1e-4, rtol=0)  # coordinates swapped


def test_swap_leaves_orthogonal_complement_unchanged():
    torch.manual_seed(5)
    h = torch.randn(6, 8)
    v_s, v_t = torch.randn(8), torch.randn(8)
    h2 = swap(h, v_s, v_t)
    delta = h2 - h
    # build a direction orthogonal to span{v_s, v_t} and check delta has no
    # component there (delta must live entirely in span{v_s, v_t}).
    V = torch.stack([v_s, v_t], dim=-1)
    w0 = torch.randn(8)
    w = w0 - V @ (torch.linalg.pinv(V) @ w0)  # Gram-Schmidt against span(V)
    assert w.norm() > 1e-3  # sanity: w is a real orthogonal direction
    assert (delta @ w).abs().max() < 1e-4


def test_swap_is_an_involution_at_alpha_one():
    torch.manual_seed(6)
    h = torch.randn(6, 8)
    v_s, v_t = torch.randn(8), torch.randn(8)
    torch.testing.assert_close(swap(swap(h, v_s, v_t), v_s, v_t), h, atol=1e-4, rtol=0)


def test_ablate_span_and_swap_run_in_bf16():
    """Regression test: torch.linalg.pinv has no bf16 CPU kernel, so both
    ablate_span and swap must upcast to float32 internally before calling it
    and cast back. This bug shipped past all-float32 TinyDecoder tests above
    and only surfaced when run against the real (bf16) Qwen3 guardrail —
    caught by causal_eval/run_plumbing_check.py, not by this suite, until now.
    """
    torch.manual_seed(7)
    h = torch.randn(6, 8, dtype=torch.bfloat16)
    V = torch.randn(8, 3, dtype=torch.bfloat16)
    v_s, v_t = torch.randn(8, dtype=torch.bfloat16), torch.randn(8, dtype=torch.bfloat16)

    h1 = ablate_span(h, V)
    assert h1.dtype == torch.bfloat16
    assert torch.isfinite(h1.float()).all()
    # bf16 has ~3 decimal digits of precision; the pinv is computed in fp32
    # internally, but h1 is cast back to bf16, so the residual is dominated by
    # that final rounding, not by the projection math itself.
    assert (h1.float() @ V.float()).abs().max() < 1e-1

    h2 = swap(h, v_s, v_t)
    assert h2.dtype == torch.bfloat16
    assert torch.isfinite(h2.float()).all()


# --- lens vectors --------------------------------------------------------------


def test_lens_vectors_shape_and_row_identity():
    model, lens = _model_and_lens()
    W_U = model.lm_head.weight  # [32, 8]
    V = lens_vectors(lens, W_U, layer=1)
    assert V.shape == (32, 8)
    # row t == single-vector helper == J.T @ W_U[t]
    for t in (0, 7, 31):
        torch.testing.assert_close(V[t], lens_vector(lens, W_U, 1, t))
        torch.testing.assert_close(V[t], lens.jacobians[1].T @ W_U[t], atol=1e-4, rtol=0)


# --- write-capable hook --------------------------------------------------------


def test_hook_modifies_residual_at_position_only():
    model, _ = _model_and_lens()
    input_ids = model.encode(_PROMPT)
    layer, pos = 1, 20
    v = torch.randn(8)

    # clean pass
    with ActivationRecorder(model.layers, at=[layer, 3]) as rec:
        model.forward(input_ids)
        clean_l = rec.activations[layer].detach().clone()
        clean_final = rec.activations[3].detach().clone()

    # steered pass: edit only `pos` at `layer`
    with InterventionHook(model.layers, {layer: lambda h: steer(h, v)}, positions=[pos]):
        with ActivationRecorder(model.layers, at=[layer, 3]) as rec:
            model.forward(input_ids)
            edited_l = rec.activations[layer].detach().clone()
            edited_final = rec.activations[3].detach().clone()

    # intervened position moved by exactly v; all other positions untouched
    torch.testing.assert_close(edited_l[0, pos] - clean_l[0, pos], v)
    mask = torch.ones(clean_l.shape[1], dtype=torch.bool)
    mask[pos] = False
    torch.testing.assert_close(edited_l[0, mask], clean_l[0, mask])
    # and the edit propagated downstream (final layer changed)
    assert (edited_final - clean_final).norm() > 1e-4


def test_hook_is_removed_after_context_exit():
    model, _ = _model_and_lens()
    input_ids = model.encode(_PROMPT)

    with ActivationRecorder(model.layers, at=[3]) as rec:
        model.forward(input_ids)
        before = rec.activations[3].detach().clone()

    with InterventionHook(model.layers, {1: lambda h: steer(h, torch.randn(8), 5.0)}):
        model.forward(input_ids)  # perturbs, but hook is scoped to this block

    # after exit, a clean pass must match the original exactly (no leftover hook)
    with ActivationRecorder(model.layers, at=[3]) as rec:
        model.forward(input_ids)
        after = rec.activations[3].detach().clone()
    torch.testing.assert_close(after, before)


def test_swap_via_hook_changes_final_logits():
    """End-to-end wiring check: swapping two real lens vectors at all positions
    changes the model's final logits (a non-trivial effect, not a semantic
    claim)."""
    model, lens = _model_and_lens()
    W_U = model.lm_head.weight
    v_s = lens_vector(lens, W_U, 1, token_id=3)
    v_t = lens_vector(lens, W_U, 1, token_id=17)
    input_ids = model.encode(_PROMPT)

    with ActivationRecorder(model.layers, at=[3]) as rec:
        model.forward(input_ids)
        clean = model.unembed(rec.activations[3].detach())

    with InterventionHook(model.layers, {1: lambda h: swap(h, v_s, v_t)}):
        with ActivationRecorder(model.layers, at=[3]) as rec:
            model.forward(input_ids)
            swapped = model.unembed(rec.activations[3].detach())

    assert (swapped - clean).norm() > 1e-4


def _unit_ref(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


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

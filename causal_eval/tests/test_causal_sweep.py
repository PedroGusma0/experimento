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

from causal_sweep import position_candidates  # noqa: E402

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

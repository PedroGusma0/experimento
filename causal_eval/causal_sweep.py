# Copyright 2026 Pedro (causal_eval experiment)
# SPDX-License-Identifier: Apache-2.0
"""Causal validation pipeline v2 — per-position causal attribution.

This is the orchestration layer for the v2 design documented in
`markdowns-de-referencia/ARCHITECTURE.md` ("Planned — causal validation
pipeline v2"). It builds on the low-level primitives in `interventions.py`
(`ablate_span`, `InterventionHook`, `lens_vectors`) to produce, for a single
prompt, a signed per-token importance score for the guardrail's `malign`
verdict.

Built incrementally, one tested unit at a time. So far:

- **Fase 0 — candidate selection** (:func:`position_candidates`): for one token
  position, the `k` J-lens vocabulary tokens most active there, aggregated
  across the workspace band.

Still to come (see ARCHITECTURE.md): the anti-confound guard, the matched-norm
random control, the per-position ablation sweep, and the signed score
`nota(p) = P_ablate(p) − P_controle(p)`.

Everything here is *mechanical*; semantic/causal validity is only established
by running it on the trained guardrail, never on the `TinyDecoder` the unit
tests use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from jlens.lens import JacobianLens

from interventions import lens_vectors

_AGGREGATES = ("max", "mean")


def position_candidates(
    lens: JacobianLens,
    unembed_weight: torch.Tensor,
    residual_by_layer: Mapping[int, torch.Tensor],
    *,
    k: int = 10,
    exclude: Iterable[int] | None = None,
    aggregate: str = "max",
) -> list[int]:
    """Top-``k`` most active J-lens vocabulary tokens at a single position.

    Implements the candidate-selection step (Fase 0) of the v2 pipeline. For
    one token position, every vocabulary token ``t`` is scored by its J-lens
    vector activation ``⟨v_t, h_l⟩`` at each band layer ``l`` (``v_t`` = row
    ``t`` of ``W_U · J_l``; this is the probe form of §2.5, dropping the final
    ``norm``). The per-layer scores are aggregated into one score per token,
    and the ``k`` highest are returned — the "``k`` most strongly activated
    J-lens vectors" of §3.5.2, selected *per position* (the paper selects at
    every position; v2 does the same rather than anchoring on the decision
    position).

    This is the top-``k``-by-inner-product notion of "active concepts". §2.5
    notes that sparse decomposition (gradient pursuit) yields a less-redundant
    set; that is the heavier, more faithful alternative, not used here.

    Args:
        lens: A fitted :class:`jlens.lens.JacobianLens`.
        unembed_weight: The model's unembedding matrix ``W_U``
            (``[vocab, d_model]``, e.g. ``model.lm_head.weight``).
        residual_by_layer: ``{band_layer: residual}`` where each ``residual``
            is the ``[d_model]`` residual-stream vector at THIS position, for
            one band layer. Aggregation is taken over these layers. Every key
            must be in ``lens.source_layers``.
        k: Number of candidate token ids to return.
        exclude: Token ids to drop before taking the top-``k`` — the
            anti-confound guard passes the clean pass's own next-token top-10
            here so a token the model was already about to emit is never
            ablated (§3.5.2).
        aggregate: ``"max"`` (most active anywhere in the band) or ``"mean"``.

    Returns:
        A list of ``k`` vocabulary token ids, highest aggregated activation
        first.

    Raises:
        ValueError: If ``residual_by_layer`` is empty, ``aggregate`` is
            unknown, or ``k`` exceeds the number of selectable tokens (vocab
            size minus excluded ids).
    """
    if aggregate not in _AGGREGATES:
        raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}")
    if not residual_by_layer:
        raise ValueError("residual_by_layer is empty; need at least one band layer")

    per_layer = []
    for layer, residual in residual_by_layer.items():
        V = lens_vectors(lens, unembed_weight, layer)  # [vocab, d_model]
        h = residual.to(V.dtype).to(V.device)
        per_layer.append(V @ h)  # [vocab] = ⟨v_t, h_l⟩ for every token t
    scores = torch.stack(per_layer, dim=0)  # [n_band_layers, vocab]
    scores = scores.amax(0) if aggregate == "max" else scores.mean(0)  # [vocab]

    excluded = set(int(t) for t in exclude) if exclude is not None else set()
    n_selectable = scores.numel() - len(excluded)
    if k > n_selectable:
        raise ValueError(
            f"k={k} exceeds selectable tokens ({n_selectable} = vocab "
            f"{scores.numel()} − {len(excluded)} excluded)"
        )
    if excluded:
        scores = scores.clone()
        scores[list(excluded)] = float("-inf")

    top = torch.topk(scores, k).indices
    return [int(t) for t in top]

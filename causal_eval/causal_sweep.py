# Copyright 2026 Pedro (causal_eval experiment)
# SPDX-License-Identifier: Apache-2.0
"""Causal validation pipeline v2 — per-position causal attribution.

This is the orchestration layer for the v2 design documented in
`markdowns-de-referencia/ARCHITECTURE.md` ("Planned — causal validation
pipeline v2"). It builds on the low-level primitives in `interventions.py`
(`ablate_span`, `InterventionHook`, `lens_vectors`) to produce, for a single
prompt, a signed per-token importance score for the guardrail's `malign`
verdict.

Built incrementally, one tested unit at a time:

- **Fase 0 — candidate selection** (:func:`position_candidates`): for one token
  position, the `k` J-lens vocabulary tokens most active there, aggregated
  across the workspace band, with the anti-confound guard exposed as `exclude`.
- **Fase 1 + 2 — the per-position ablation sweep** (:func:`sweep_positions`):
  for every input position, group-ablate its candidates (vs. a matched-norm
  random control), read the effect on a caller-supplied verdict score at the
  decision position, and return the signed per-position score
  `nota(p) = score_ablate(p) − score_control(p)`.

The sweep is model-agnostic (driven by two caller callables, `run_forward`
and `score_fn`) so it unit-tests against `TinyDecoder`. Everything here is
*mechanical*; semantic/causal validity is only established by running it on
the trained guardrail (see `run_causal_pipeline.py`), never on `TinyDecoder`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from jlens.fitting import SKIP_FIRST_N_POSITIONS
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from interventions import (
    InterventionHook,
    ablate_span,
    lens_vectors,
    matched_norm_control,
)

_AGGREGATES = ("max", "mean")


def position_candidates(
    lens: JacobianLens,
    unembed_weight: torch.Tensor,
    residual_by_layer: Mapping[int, torch.Tensor],
    *,
    k: int = 10,
    exclude: Iterable[int] | None = None,
    aggregate: str = "max",
    V_by_layer: Mapping[int, torch.Tensor] | None = None,
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
        V_by_layer: Optional precomputed ``{layer: lens_vectors(lens,
            unembed_weight, layer)}`` (each ``[vocab, d_model]``). Pass this
            when the caller already has it (e.g. :func:`sweep_positions`,
            once for the whole sweep) to skip recomputing the ``W_U · J_l``
            matmul on every call — that matmul (~1.3 TFLOP for Qwen3-1.7B's
            vocab) dominates wall-clock cost when called once per swept
            position instead of once per layer; see ARCHITECTURE.md, "First
            real-guardrail run and three findings", Finding 2. Falls back to
            computing it internally via ``lens``/``unembed_weight`` when
            omitted (unchanged behavior, still correct — just redundant at
            scale).

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
        V = (
            V_by_layer[layer]
            if V_by_layer is not None
            else lens_vectors(lens, unembed_weight, layer)
        )  # [vocab, d_model]
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


# ---------------------------------------------------------------------------
# Fase 1 + 2: the per-position ablation sweep with a signed score.
# ---------------------------------------------------------------------------


def sweep_positions(
    blocks: Sequence[nn.Module],
    lens: JacobianLens,
    unembed_weight: torch.Tensor,
    input_ids: torch.Tensor,
    run_forward: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor], float],
    *,
    layers: Sequence[int],
    k: int = 10,
    decision_position: int = -1,
    p_start: int | None = None,
    p_end: int | None = None,
    aggregate: str = "max",
    guard_top_n: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Per-position causal attribution sweep (v2 Fase 1 + 2).

    For every input position ``p`` in ``[p_start, p_end)``, group-ablate the
    ``k`` J-lens vectors most active at ``p`` (across ``layers``) and measure
    the effect on ``score_fn`` read at ``decision_position``, against a
    matched-norm random-direction control. Returns one record per position
    with the signed score ``nota(p) = score_ablate(p) − score_control(p)``.

    Model-agnostic by construction — the model shows up only through:

    Args:
        blocks: The residual blocks to hook (``model.layers``).
        lens: A fitted :class:`jlens.lens.JacobianLens`.
        unembed_weight: ``W_U``, ``[vocab, d_model]``.
        input_ids: ``[1, seq_len]`` — the rendered prompt only (no generated
            tokens; the sweep never crosses into positions the model would
            generate, so an input token is always attributable to the input).
        run_forward: ``input_ids -> logits`` of shape ``[seq_len, vocab]`` (all
            positions). Called clean and again under each `InterventionHook`;
            it must run the model's forward so the hook fires (e.g.
            ``lambda ids: gl.hf(ids).logits[0]``). No-grad is applied around it.
        score_fn: ``logits[vocab] -> float`` — the scalar whose change is
            attributed (e.g. ``P(malign)`` at the decision position). Kept
            caller-side because "malign" is guardrail-specific; `TinyDecoder`
            tests pass an arbitrary fixed-token probability.
        layers: Band layer indices to ablate across (all in one forward pass;
            band width costs no extra passes).
        k: Candidates per position.
        decision_position: Where ``score_fn`` is read (default ``-1``).
        p_start: First swept position. Defaults to ``SKIP_FIRST_N_POSITIONS``
            (the fitting.py attention-sink skip).
        p_end: One past the last swept position. Defaults to ``seq_len`` — the
            input/generation boundary (never a generated token).
        aggregate: Passed to :func:`position_candidates` (``"max"``/``"mean"``).
        guard_top_n: Anti-confound guard width — exclude the model's own clean
            next-token top-``guard_top_n`` at each position (§3.5.2).
        seed: Seeds the control's random directions for reproducibility.

    Returns:
        A list of dicts (ascending position), each with ``position``,
        ``n_candidates``, ``candidate_ids`` (the ``k`` vocabulary token ids
        actually ablated at this position — raw ids, not decoded strings,
        since this function is model/tokenizer-agnostic; the caller decodes
        them, same as it already decodes the swept position's own token),
        ``score_ablate``, ``score_control``, ``nota``, and
        ``kl_ablate``/``kl_control`` — ``KL(clean ‖ intervened)`` on the
        full next-token distribution at ``decision_position`` (§A.6's
        "ablation effect" metric, computed for both the real ablation and its
        matched-norm control, so a control-adjusted KL is available as
        ``kl_ablate - kl_control`` if wanted). ``score_clean`` (the
        un-intervened baseline) is attached to every row for reference.
    """
    seq_len = int(input_ids.shape[1])
    if p_start is None:
        p_start = SKIP_FIRST_N_POSITIONS
    if p_end is None:
        p_end = seq_len
    p_end = min(p_end, seq_len)  # never cross into generated positions

    generator = torch.Generator().manual_seed(seed)
    V_all = {layer: lens_vectors(lens, unembed_weight, layer) for layer in layers}

    with torch.no_grad():
        # One clean, recorded pass: band residuals at every position (for
        # candidate selection) + clean logits at every position (decision
        # readout + the per-position anti-confound guard).
        with ActivationRecorder(blocks, at=list(layers)) as rec:
            clean_logits = run_forward(input_ids)  # [seq_len, vocab]
            residuals = {
                layer: rec.activations[layer][0].detach() for layer in layers
            }
        clean_logits_dp = clean_logits[decision_position]
        score_clean = score_fn(clean_logits_dp)

        results: list[dict] = []
        for p in range(p_start, p_end):
            residual_by_layer = {layer: residuals[layer][p] for layer in layers}
            guard = clean_logits[p].topk(guard_top_n).indices.tolist()
            cands = position_candidates(
                lens, unembed_weight, residual_by_layer,
                k=k, exclude=guard, aggregate=aggregate, V_by_layer=V_all,
            )
            V = {layer: V_all[layer][cands].T for layer in layers}  # [d_model, k]

            ablate_edits = {
                layer: (lambda x, V=V[layer]: ablate_span(x, V)) for layer in layers
            }
            with InterventionHook(blocks, ablate_edits, positions=[p]):
                ablate_logits_dp = run_forward(input_ids)[decision_position]
            score_ablate = score_fn(ablate_logits_dp)
            kl_ablate = _kl_divergence(clean_logits_dp, ablate_logits_dp)

            control_edits = {
                layer: (
                    lambda x, V=V[layer]: matched_norm_control(x, V, generator=generator)
                )
                for layer in layers
            }
            with InterventionHook(blocks, control_edits, positions=[p]):
                control_logits_dp = run_forward(input_ids)[decision_position]
            score_control = score_fn(control_logits_dp)
            kl_control = _kl_divergence(clean_logits_dp, control_logits_dp)

            results.append(
                {
                    "position": p,
                    "n_candidates": len(cands),
                    "candidate_ids": cands,
                    "score_clean": score_clean,
                    "score_ablate": score_ablate,
                    "score_control": score_control,
                    "nota": score_ablate - score_control,
                    "kl_ablate": kl_ablate,
                    "kl_control": kl_control,
                }
            )
    return results


def _kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """``KL(P ‖ Q)`` between two next-token distributions given as raw
    (pre-softmax) logits — the paper's own "ablation effect" metric (§A.6):
    "we take its vector for the intermediate token... and record the KL
    divergence this induces on the model's output." Computed via
    ``log_softmax`` (not ``softmax`` then ``log``) for numerical stability.
    """
    log_p = torch.log_softmax(p_logits.float(), dim=-1)
    log_q = torch.log_softmax(q_logits.float(), dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum())

# Copyright 2026 Pedro (guardrail_eval experiment)
# SPDX-License-Identifier: Apache-2.0
"""Causal J-lens interventions (steer / ablate / swap) for the guardrail.

The `jlens` library ships fitting + reading only (`fit` / `apply` /
`transport`); the paper's two causal *writing* primitives are described but
not implemented there (see `markdowns-de-referencia/PAPER_SUMMARY.md`,
"The J-lens's read/write primitives", §2.5 / Figure 4C). Because neither
`causal_eval/` nor `guardrail_eval/` may modify `jlens/`, this module builds
them here, on top of the three ingredients `jlens` already exposes:

- ``J_l``          — ``JacobianLens.jacobians[layer]`` (a fitted lens),
- ``W_U``          — the model's unembedding weight (e.g. ``lm_head.weight``),
- a forward hook   — but a *write-capable* one, since
  :class:`jlens.hooks.ActivationRecorder` only reads (its hook stores the
  tensor, never returns a modified one).

The **J-lens vector** for vocabulary token ``t`` at layer ``l`` is the row
``t`` of ``W_U · J_l`` — a direction in residual-stream space (§2.1). Note it
drops the final ``norm``, so it is the linearised readout direction, not the
exact logit gradient (§2.5 caveat).

**Scope.** These are the *mechanical* primitives. They say nothing about
whether an intervention is semantically meaningful on a given model — that
requires a trained model (the Qwen3-1.7B guardrail), not the toy decoder the
unit tests run against. See ``tests/test_interventions.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from jlens.lens import JacobianLens

# ---------------------------------------------------------------------------
# J-lens vectors: v_t = row t of  W_U · J_l   (residual-stream direction).
# ---------------------------------------------------------------------------


def lens_vectors(
    lens: JacobianLens, unembed_weight: torch.Tensor, layer: int
) -> torch.Tensor:
    """All J-lens vectors at ``layer``: ``W_U · J_l``, shape ``[vocab, d_model]``.

    Row ``t`` is the residual-space direction for vocabulary token ``t``.

    Args:
        lens: A fitted :class:`jlens.lens.JacobianLens`.
        unembed_weight: The model's unembedding matrix ``W_U``
            (``[vocab, d_model]`` — e.g. ``model.lm_head.weight``). The final
            ``norm`` is intentionally *not* applied (matches the paper's
            "rows of ``W_U·J_l``" definition, §2.1).
        layer: Source layer index (must be in ``lens.source_layers``).
    """
    J = lens.jacobians[layer].to(unembed_weight.dtype).to(unembed_weight.device)
    return unembed_weight @ J  # [vocab, d] @ [d, d] -> [vocab, d]


def lens_vector(
    lens: JacobianLens, unembed_weight: torch.Tensor, layer: int, token_id: int
) -> torch.Tensor:
    """A single J-lens vector ``v_t`` at ``layer`` (row ``token_id``)."""
    J = lens.jacobians[layer].to(unembed_weight.dtype).to(unembed_weight.device)
    return unembed_weight[token_id] @ J  # [d] @ [d, d] -> [d]


# ---------------------------------------------------------------------------
# Residual-space edits. Each maps h of shape [..., d_model] -> [..., d_model],
# broadcasting over any leading (batch/position) dims.
# ---------------------------------------------------------------------------


def steer(h: torch.Tensor, v: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Steer along a J-lens vector: ``h + α·v`` (§2.5). Positive ``α`` injects
    the concept; negative ``α`` suppresses it (a coarser ablation than
    :func:`ablate`)."""
    return h + alpha * v.to(h.dtype).to(h.device)


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def ablate(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ablate a concept by projecting out the component of ``h`` along ``v``
    (§2.5 / §3.5.2). Afterwards ``⟨v, h⟩ ≈ 0``. Idempotent."""
    vhat = _unit(v.to(h.dtype).to(h.device))
    coeff = h @ vhat  # [...]
    return h - coeff.unsqueeze(-1) * vhat


def ablate_span(h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Zero ``h``'s projection onto the span of several J-lens vectors at once
    — the paper's **J-space ablation** with ``V`` the ``k`` most active
    vectors (§3.5.2, ``k=10`` across a layer band). ``V`` is ``[d_model, k]``.

    Uses the least-squares subspace projector ``P = V(VᵀV)⁺Vᵀ`` and returns
    ``h − hP``, so it is exact even when the columns of ``V`` are not
    orthonormal (J-lens vectors never are).

    ``torch.linalg.pinv`` (SVD-based) has no bf16/fp16 CPU kernel, so the
    pseudoinverse is always computed in float32 and cast back to ``h``'s
    dtype — needed for real (typically bf16) guardrail models; `TinyDecoder`'s
    tests default to float32, where this upcast is a no-op."""
    V = V.to(h.device)
    Vpinv = torch.linalg.pinv(V.float()).to(h.dtype)  # [k, d]
    coeffs = h @ Vpinv.transpose(-1, -2)  # [..., k] = V⁺ h
    return h - coeffs @ V.to(h.dtype).transpose(-1, -2)


def swap(
    h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Lens-coordinate swap (Figure 4C / §2.5): exchange concept ``s`` for
    concept ``t`` while leaving everything orthogonal to ``span{v_s, v_t}``
    unchanged.

    Forms ``V = [v_s  v_t]``, reads coordinates ``c = V⁺·h`` (pseudoinverse),
    and returns ``h + α·V(σ(c) − c)`` where ``σ`` swaps the two entries of
    ``c``. Default ``α=1``; the paper uses ``α=2`` when ``α=1`` moves the
    activation in the right direction but underdrives it (§A.13).

    Like :func:`ablate_span`, the pseudoinverse is computed in float32 and
    cast back (no bf16/fp16 CPU kernel for ``torch.linalg.pinv``).
    """
    v_s = v_s.to(h.dtype).to(h.device)
    v_t = v_t.to(h.dtype).to(h.device)
    V = torch.stack([v_s, v_t], dim=-1)  # [d, 2]
    Vpinv = torch.linalg.pinv(V.float()).to(h.dtype)  # [2, d]
    c = h @ Vpinv.transpose(-1, -2)  # [..., 2] = V⁺ h
    c_swapped = c.flip(-1)  # σ(c): swap the two entries
    delta = alpha * (c_swapped - c)  # [..., 2]
    return h + delta @ V.transpose(-1, -2)  # [..., d]


# ---------------------------------------------------------------------------
# Write-capable forward hook (the piece ActivationRecorder deliberately isn't).
# ---------------------------------------------------------------------------

Edit = Callable[[torch.Tensor], torch.Tensor]


class InterventionHook:
    """Context manager that **writes** an edit into the residual stream during
    the forward pass.

    Unlike :class:`jlens.hooks.ActivationRecorder` (whose hook only stores the
    block output), this hook *returns* a modified tensor, so PyTorch replaces
    the block's output with it and the change propagates through every
    downstream layer.

    Args:
        blocks: The model's residual blocks (``model.layers``).
        edits: ``{block_index: edit_fn}``. Each ``edit_fn`` maps the residual
            at the intervened positions (``[..., n_positions, d_model]``) to a
            tensor of the same shape — e.g. ``lambda h: swap(h, v_s, v_t)``.
        positions: Sequence positions to edit (indexing the ``dim=-2`` axis;
            negative indices allowed). ``None`` edits every position. The same
            positions apply to every layer in ``edits``.

    Registration order matters: enter this hook **before** an
    ``ActivationRecorder`` on the same blocks if the recorder must see the
    edited tensor.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        edits: Mapping[int, Edit],
        *,
        positions: Iterable[int] | None = None,
    ) -> None:
        self._blocks = blocks
        self._edits = dict(edits)
        self._positions = None if positions is None else list(positions)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, edit: Edit) -> Callable[..., object]:
        positions = self._positions

        def hook(module: nn.Module, inputs, output):
            is_tuple = not torch.is_tensor(output)
            tensor = output[0] if is_tuple else output
            new = tensor.clone()
            if positions is None:
                new = edit(new)
            else:
                seq_len = new.shape[-2]
                idx = torch.tensor(
                    [p % seq_len for p in positions], device=new.device
                )
                edited = edit(new.index_select(-2, idx))
                new = new.index_copy(-2, idx, edited.to(new.dtype))
            return (new, *output[1:]) if is_tuple else new

        return hook

    def __enter__(self) -> InterventionHook:
        try:
            for index, edit in self._edits.items():
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(edit))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

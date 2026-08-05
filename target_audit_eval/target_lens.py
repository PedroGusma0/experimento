#!/usr/bin/env python
"""J-lens-wrapped target model for pipeline v4 (see ARCHITECTURE.md,
"Planned -- pipeline v4"). No guardrail role here: this model receives the
PIArena-attacked prompt directly and the J-lens reads out ITS OWN
disposition, not a classifier's -- the reframing v4 records is "what is the
model that's actually under attack disposed to say/do", read right where it
is poised to act (the last prompt position, before generation starts).

Reuses guardrail_eval/jlens_readout.py's GuardrailLens (model+lens loading,
readout/readout_multi/token_span) via COMPOSITION, not duplication or
modification: TargetLens wraps one GuardrailLens instance and delegates to
it for everything generic (loading, readout), adding only what's new for
this role -- an open, undefended prompt render (no classifier system
prompt) and open-ended generation. Nothing under guardrail_eval/ is edited.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "guardrail_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jlens_readout import GuardrailLens, PRESETS  # noqa: E402


class TargetLens:
    """A target model + its fitted Jacobian lens, loaded once.

    Composes a :class:`GuardrailLens` for model/lens loading and readout
    (so any future preset/fix added there is picked up here for free), and
    adds only the target-specific prompt render + open-ended generation.

    Defaults to Qwen3-1.7B -- the only preset ever verified to load, encode,
    and lens-apply on a local CPU machine (~7.7GB RAM, no GPU); see
    ``guardrail_eval/jlens_readout.py``'s ``PRESETS`` for the
    ``google/gemma-3-4b-it`` alternative (RunPod-only, workspace band not
    yet established there -- not used by default here).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        *,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self._core = GuardrailLens(model_id=model_id, dtype=dtype, device=device)
        self._disable_thinking = (
            PRESETS[model_id].disable_thinking if model_id in PRESETS else False
        )

    # -- delegated straight to the wrapped GuardrailLens --
    @property
    def model(self):
        return self._core.model

    @property
    def hf(self):
        return self._core.hf

    @property
    def tok(self):
        return self._core.tok

    @property
    def lens(self):
        return self._core.lens

    def readout(self, prompt_str: str, **kwargs):
        return self._core.readout(prompt_str, **kwargs)

    def readout_multi(self, prompt_str: str, **kwargs):
        return self._core.readout_multi(prompt_str, **kwargs)

    def token_span(self, prompt_str: str, **kwargs):
        return self._core.token_span(prompt_str, **kwargs)

    # -- new for the target role --
    def render_prompt(self, target_inst: str, context: str) -> str:
        """Bare user turn, no system prompt -- mirrors
        ``target_model.py::TargetModel``'s design principle (isolate the
        attack's effect on the target from any confound of the target's own
        system-level framing), but renders to a STRING
        (``tokenize=False``) rather than a tensor, so
        ``self.model.encode(this same string)`` tokenizes identically for
        both :meth:`generate` and :meth:`readout` -- keeping the decision
        position (``-1``) aligned between the two.

        ``target_inst`` already ends in "...Context:" in the PIArena schema
        (e.g. "Complete the following task... Context:"), so a plain space
        join reads naturally as one instruction followed by its context.
        """
        messages = [{"role": "user", "content": f"{target_inst} {context}"}]
        kwargs = {}
        if self._disable_thinking:
            kwargs["enable_thinking"] = False
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    def generate(self, prompt_str: str, *, max_new_tokens: int = 200) -> str:
        """Open-ended greedy generation over the target's response.

        Same generation kwargs as ``TargetModel.generate`` /
        ``GuardrailLens.classify`` (``do_sample=False``, no
        temperature/top_p/top_k), but via ``self.model.encode(prompt_str)``
        (not ``apply_chat_template(return_tensors=...)``) so the exact same
        token ids :meth:`readout` reads are the ones generation continues
        from.
        """
        input_ids = self.model.encode(prompt_str)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output_ids = self.hf.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id
                if self.tok.pad_token_id is not None
                else self.tok.eos_token_id,
            )
        return self.tok.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        )

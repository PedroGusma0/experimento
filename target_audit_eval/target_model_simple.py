#!/usr/bin/env python
"""The downstream target model for the lens-free PIArena baseline experiment
(Ataque -> Alvo -> Judge, no guardrail, no J-lens at all -- see
ARCHITECTURE.md's PIArena section for why this superseded pipeline v4's
"target + J-lens" idea).

Mirrors `guardrail_eval/target_model.py`'s design principle exactly (bare
user turn, no system prompt, so an attack's effect on the target isn't
confounded by the target's own framing) but:
- takes PIArena's `(target_inst, context)` pair, not a single raw seed
  string (`guardrail_eval/target_model.py` predates the PIArena schema);
- loads via `AutoModelForImageTextToText` by default, since the default
  target (`google/gemma-3-4b-it`) is a multimodal checkpoint outside the
  CausalLM auto-map (same reason `GuardrailPreset` needs `loader=
  "image_text_to_text"` for it in `guardrail_eval/jlens_readout.py`) --
  pass `loader="causal_lm"` for a text-only model instead.
- never touches `jlens` -- no lens loaded, no readout, nothing recorded
  about internal activations. This is deliberate for this experiment: it's
  a faithful PIArena baseline replication, not an interpretability pass.
"""

from __future__ import annotations

import torch
import transformers

MODEL_ID = "google/gemma-3-4b-it"


class TargetModel:
    """A target model loaded once and reused across generate() calls."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        loader: str = "image_text_to_text",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ) -> None:
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
        if loader == "image_text_to_text":
            self.hf = transformers.AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=dtype
            )
        else:
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype
            )
        self.hf.to(device)
        self.hf.eval()
        self.device = device

    def __repr__(self) -> str:
        return f"TargetModel({type(self.hf).__name__})"

    def render_prompt(self, target_inst: str, context: str) -> str:
        """Bare user turn, no system prompt. `target_inst` already ends in
        "...Context:" in the PIArena schema, so a plain space join reads
        naturally as one instruction followed by its context (same
        convention as the superseded `target_lens.py::render_prompt`)."""
        messages = [{"role": "user", "content": f"{target_inst} {context}"}]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt_str: str, *, max_new_tokens: int = 200) -> str:
        """Greedy-generate the target's response to an already-rendered
        prompt string (from :meth:`render_prompt`)."""
        encoded = self.tok(prompt_str, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = self.hf.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id
                if self.tok.pad_token_id is not None
                else self.tok.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return self.tok.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        )

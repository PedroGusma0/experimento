#!/usr/bin/env python
"""The downstream target model: gemma-3-1b-it, plain open-ended generation.

No jlens involvement here -- the Jacobian lens is applied only to the
guardrail (see jlens_readout.py); the target is a bare, undefended assistant
that receives whatever prompt reaches it (raw seed or wrapped attack text)
and generates a normal response, with no system prompt. That isolates
whether an attack manipulates the target from any confound of the target's
own system-level framing.
"""

from __future__ import annotations

import torch
import transformers

MODEL_ID = "google/gemma-3-1b-it"


class TargetModel:
    """gemma-3-1b-it loaded once and reused across generate() calls."""

    def __init__(self, model_id: str = MODEL_ID, *, dtype: torch.dtype = torch.float32) -> None:
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.hf.eval()

    def __repr__(self) -> str:
        return f"TargetModel({type(self.hf).__name__})"

    def generate(self, prompt: str, *, max_new_tokens: int = 200) -> str:
        """Greedy-generate the target's response to a bare user turn."""
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        with torch.no_grad():
            output_ids = self.hf.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return self.tok.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        )

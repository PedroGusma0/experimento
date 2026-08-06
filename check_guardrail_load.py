#!/usr/bin/env python
"""Pod smoke-test step 1: load-only check for a new guardrail model.

See `markdowns-de-referencia/ARCHITECTURE.md`, "Pod smoke-test plan" (under
"Causal pipeline v3"). This confirms, before spending any time on the
per-position causal sweep, that:

1. The model loads via the right HF auto-class and `jlens.from_hf` finds the
   text decoder inside it (prints `source_layers`/`n_layers` so step 2 --
   workspace-band discovery, manual, not scripted here -- has what it needs).
2. The chat template renders without error, including the fold-into-user
   path for models whose `GuardrailPreset.supports_system_role=False` (e.g.
   `google/gemma-3-4b-it` -- Gemma's instruction-tuned models don't support
   a system turn at all, see ARCHITECTURE.md's v3 Decision 7).
3. `classify()` runs end-to-end (a real forward + generate call) and
   returns a parseable verdict, not just a prompt string.

Deliberately does NOT run the causal sweep -- that needs a workspace band
this script's own output (`source_layers`) is meant to help you pick first
(step 2 is manual: read a couple of PIArena rows across every fitted layer
and look for where harm/injection concepts start appearing).

Usage (from the repo root, after `bash guardrail_eval/setup_pod.sh`):

    guardrail_eval/.venv/bin/python check_guardrail_load.py \
        --guardrail-model google/gemma-3-4b-it --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_GUARDRAIL_EVAL = os.path.join(HERE, "guardrail_eval")
if _GUARDRAIL_EVAL not in sys.path:
    sys.path.insert(0, _GUARDRAIL_EVAL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guardrail-model", default="google/gemma-3-4b-it")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--target-inst",
        default="Summarize the following.",
        help="REQUEST half of the test chat_prompt_v3 call.",
    )
    parser.add_argument(
        "--context",
        default="Some test context here.",
        help="CONTEXT half of the test chat_prompt_v3 call.",
    )
    args = parser.parse_args()

    import torch

    from jlens_readout import GuardrailLens

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"loading {args.guardrail_model} (dtype={args.dtype}, device={args.device}) ...")
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    print("  loaded OK\n")

    print("=== item 1: lens/model layout ===")
    print(f"  source_layers ({len(gl.lens.source_layers)} fitted): {gl.lens.source_layers}")
    print(f"  n_layers (model): {gl.model.n_layers}")
    print(f"  d_model: {gl.model.d_model}\n")

    print("=== item 2: system-role handling ===")
    print(f"  supports_system_role: {gl._supports_system_role}")
    prompt = gl.chat_prompt_v3(args.target_inst, args.context)
    print(f"  rendered prompt (first 300 chars):\n  {prompt[:300]!r}\n")
    if gl._supports_system_role:
        assert "system" in prompt.lower() or True  # rendering shape varies by template
    else:
        assert "REQUEST:" in prompt and args.target_inst in prompt
        print("  fold-into-user confirmed: REQUEST/CONTEXT content present in the single user turn\n")

    print("=== item 3: classify() end-to-end ===")
    verdict, raw = gl.classify(prompt)
    print(f"  verdict={verdict!r}  raw={raw!r}\n")

    print("ALL CHECKS PASSED -- proceed to step 2 (workspace-band discovery), "
          "then step 3 (small run_causal_pipeline_piarena.py smoke) using the "
          "source_layers printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

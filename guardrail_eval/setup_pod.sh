#!/usr/bin/env bash
# Pod setup for guardrail_eval, run once (idempotent) before run_audit_pipeline.py.
#
# Fixes a dependency trap hit in a prior RunPod session: an older pod
# template (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) shipped
# torch 2.4.0, but jlens requires transformers>=5.5 -- 5.14.1 specifically
# imports `torch.distributed.tensor.DTensor`, unavailable in torch 2.4.0
# (ImportError). Upgrading torch alone (as done manually that time) then
# broke torchvision/torchaudio's compiled extensions against the new torch
# ABI (`operator torchvision::nms does not exist`, then an `undefined
# symbol` in libtorchaudio.so).
#
# The actual pod template varies between sessions (this repo has run on both
# a 2.4.0/cu124 and a 2.8.0/cu128 template so far) -- so this script does
# NOT hardcode a CUDA wheel index. If a reinstall is needed, the index is
# derived from whatever CUDA build the pod's own torch already reports
# (`torch.version.cuda`), so it can never silently downgrade/mismatch a
# newer template the way a hardcoded cu124 index would. It also no longer
# gates on torchvision/torchaudio -- neither is an actual dependency of
# jlens or guardrail_eval (check requirements.txt/pyproject.toml), so a
# template that simply doesn't ship them shouldn't trigger a reinstall at
# all; they're checked informationally only, after the real gate passes.
#
# Usage: bash guardrail_eval/setup_pod.sh   (run from anywhere; cds to repo root)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"

# The real gate: DTensor is what actually broke on torch 2.4.0, and it's
# part of torch itself (no dependency on transformers being installed yet --
# that only happens further down, via `pip install -e`/`pip install -r
# requirements.txt`). Testing `import transformers` here would always fail
# on a fresh pod regardless of torch's health, since it isn't installed yet
# at this point in the script.
self_test() {
    python -c "
from torch.distributed.tensor import DTensor
print('torch OK (DTensor importable)')
"
}

# Wheel index matched to *this* pod's actual CUDA build, not a hardcoded
# guess -- e.g. torch.version.cuda == '12.8' -> .../whl/cu128.
cuda_index_url() {
    python -c "
import torch
v = torch.version.cuda
if not v:
    raise SystemExit('torch has no CUDA build (torch.version.cuda is None) -- not a GPU pod?')
major, minor = v.split('.')[:2]
print(f'https://download.pytorch.org/whl/cu{major}{minor}')
"
}

echo "== guardrail_eval/setup_pod.sh =="
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| cuda build:', torch.version.cuda)" || true

echo "-- checking torch compatibility (DTensor) --"
if self_test; then
    echo "already compatible, skipping torch/torchvision/torchaudio reinstall"
else
    CUDA_INDEX_URL="$(cuda_index_url)"
    echo "incompatible (DTensor import failed) -- reinstalling from $CUDA_INDEX_URL (matched to this pod's CUDA build)"
    echo "   pip install --upgrade torch torchvision torchaudio --index-url $CUDA_INDEX_URL"
    pip install --upgrade torch torchvision torchaudio --index-url "$CUDA_INDEX_URL"

    echo "-- re-checking after reinstall --"
    if ! self_test; then
        echo "FATAL: torch still incompatible (DTensor still not importable) after" >&2
        echo "the matched-index reinstall. See PLAN_runpod_audit.md for manual" >&2
        echo "diagnosis (check torch/torchvision/torchaudio versions against the" >&2
        echo "pod's actual CUDA driver version)." >&2
        exit 1
    fi
fi

echo "-- torchvision/torchaudio (informational only -- not a jlens/guardrail_eval dependency) --"
python -c "
import torchvision, torchaudio
print('torchvision', torchvision.__version__, '| torchaudio', torchaudio.__version__)
" || echo "   not importable -- fine, guardrail_eval doesn't use them"

echo "-- installing jlens (editable) --"
pip install -e "$REPO_ROOT"

echo "-- installing guardrail_eval requirements --"
pip install -r "$HERE/requirements.txt"

echo "-- final sanity check --"
python -c "
import torch, transformers, jlens
print('CUDA available:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('jlens', jlens.__file__)
"

echo "== setup complete =="

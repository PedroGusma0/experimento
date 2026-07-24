#!/usr/bin/env bash
# Pod setup for guardrail_eval, run once (idempotent) before run_audit_pipeline.py.
#
# Fixes a dependency trap hit in a prior RunPod session: the pod's official
# PyTorch template (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
# ships torch 2.4.0, but jlens requires transformers>=5.5 -- 5.14.1
# specifically imports `torch.distributed.tensor.DTensor`, unavailable in
# torch 2.4.0 (ImportError). Upgrading torch alone (as done manually last
# time) then breaks torchvision/torchaudio's compiled extensions against the
# new torch ABI (`operator torchvision::nms does not exist`, then an
# `undefined symbol` in libtorchaudio.so) -- each isolated fix broke the next
# because torch/torchvision/torchaudio must be installed together, from the
# same CUDA-matched wheel index, not one at a time.
#
# Usage: bash guardrail_eval/setup_pod.sh   (run from anywhere; cds to repo root)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"

self_test() {
    python -c "
from torch.distributed.tensor import DTensor
import torchvision
import torchaudio
print('deps OK')
"
}

echo "== guardrail_eval/setup_pod.sh =="
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" || true

echo "-- checking torch/torchvision/torchaudio compatibility --"
if self_test; then
    echo "already compatible, skipping torch/torchvision/torchaudio reinstall"
else
    echo "incompatible (DTensor/torchvision/torchaudio import failed) -- reinstalling"
    echo "   pip install --upgrade torch torchvision torchaudio --index-url $CUDA_INDEX_URL"
    pip install --upgrade torch torchvision torchaudio --index-url "$CUDA_INDEX_URL"

    echo "-- re-checking after reinstall --"
    if ! self_test; then
        echo "FATAL: torch/torchvision/torchaudio still incompatible after the" >&2
        echo "matched-index reinstall. This script only handles the specific" >&2
        echo "DTensor/ABI-mismatch class seen before -- see PLAN_runpod_audit.md" >&2
        echo "for manual diagnosis (check torch/torchvision/torchaudio versions" >&2
        echo "against the pod's actual CUDA driver version)." >&2
        exit 1
    fi
fi

echo "-- installing jlens (editable) --"
pip install -e "$REPO_ROOT"

echo "-- installing guardrail_eval requirements --"
pip install -r "$HERE/requirements.txt"

echo "-- final sanity check --"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "== setup complete =="

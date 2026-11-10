#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
python3.11 -m venv --system-site-packages "$root/.venv"
python="$root/.venv/bin/python"
"$python" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY

flash=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
conv=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
"$python" -m pip install 'transformers==5.8.0' 'accelerate==1.15.0' \
  'flash-linear-attention[cuda]==0.4.2' 'nvidia-ml-py==13.615.71' "$flash" "$conv"
"$python" - <<'PY'
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
import pynvml, torch
print(torch.__version__, torch.version.cuda)
PY

#!/bin/bash
# Unattended JetPack 7.2 Orin compatibility check for slow/flaky links. Retries every download
# (uv and containerd keep finished pieces between attempts), then runs a GPU test and appends
# the outcome to ~/jp7test/RESULTS.txt. Order: smallest/most informative first.
#   1. pytorch.org torch 2.11+cu130 aarch64 wheel natively (built for sm_80..sm_120, no sm_87)
#   2. nvcr.io/nvidia/pytorch:26.05-py3   (base of Dockerfile.jp7-thor; NGC stopped -igpu after 26.01)
#   3. nvcr.io/nvidia/pytorch:25.08-py3-igpu (Dockerfile.jp7's Orin default)
cd ~/jp7test
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10
R=~/jp7test/RESULTS.txt
log() { echo "[$(date '+%F %T')] $*" | tee -a $R; }
TEST='import torch, torch.nn.functional as F
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); y = x @ x; torch.cuda.synchronize()
c = F.conv2d(torch.randn(1, 3, 224, 224, device="cuda"), torch.randn(16, 3, 3, 3, device="cuda"))
q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16); s = F.scaled_dot_product_attention(q, q, q)
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list(),
      "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
      "| bf16 matmul", bool(y.isfinite().all()), "| cudnn conv", tuple(c.shape), "| sdpa", tuple(s.shape))'
log "host: $(head -1 /etc/nv_tegra_release | cut -c1-40) | $(nvidia-smi | sed -n 3p | tr -s ' ')"

[ -x tvenv/bin/python ] || uv venv -q -p /usr/bin/python3 tvenv
for i in $(seq 1 60); do
  VIRTUAL_ENV=$HOME/jp7test/tvenv uv pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 >> pip_torch.log 2>&1 && break
  sleep 20
done
log "native pytorch.org cu130 wheel: $(tvenv/bin/python -c "$TEST" 2>&1 | tail -1)"

pull() { for i in $(seq 1 60); do docker pull "$1" >> pull.log 2>&1 && return 0; sleep 20; done; return 1; }
for img in nvcr.io/nvidia/pytorch:26.05-py3 nvcr.io/nvidia/pytorch:25.08-py3-igpu; do
  if pull $img; then
    log "$img: $(docker run --rm --runtime nvidia --entrypoint python $img -c "$TEST" 2>&1 | tail -1)"
  else
    log "$img: pull failed after 60 attempts"
  fi
done
log "DONE"

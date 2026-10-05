#!/bin/bash
# On a JetPack 7.2 Orin: pull cmpelkk/jetson-unified:jp7-orin with retries, then GPU-check it.
# Results are appended to ~/jp7test/RESULTS.txt.
cd ~/jp7test; R=~/jp7test/RESULTS.txt; IMG=cmpelkk/jetson-unified:jp7-orin
log() { echo "[$(date '+%F %T')] $*" | tee -a $R; }
ok=1; for i in $(seq 1 60); do docker pull $IMG >> pull_jp7orin.log 2>&1 && { ok=0; break; }; sleep 20; done
[ $ok = 0 ] || { log "$IMG: pull failed after 60 attempts"; exit 1; }
cat > /tmp/jp7orin_inner.sh <<'IN'
python - <<'PY'
import torch, numpy, cv2
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("torch", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "bf16 ok", bool((x @ x).isfinite().all()))
i = cv2.getBuildInformation(); arch = next(l.strip() for l in i.splitlines() if "GPU arch" in l)
g = cv2.cuda_GpuMat(); g.upload(numpy.zeros((480, 640, 3), numpy.uint8))
print("cv2", cv2.__version__, arch, "| cuda devices", cv2.cuda.getCudaEnabledDeviceCount(), "| cuda resize", cv2.cuda.resize(g, (320, 240)).download().shape)
import lerobot, transformers; print("lerobot", lerobot.__version__, "transformers", transformers.__version__)
PY
echo "gst: $(for p in nvv4l2decoder nvvidconv nvarguscamerasrc; do gst-inspect-1.0 $p >/dev/null 2>&1 && printf "$p=ok " || printf "$p=missing "; done)"
echo "llama: $(llama-cli --list-devices 2>&1 | grep -i -m1 'CUDA0')"
python -c "import llama_cpp; print('llama-cpp-python gpu offload', llama_cpp.llama_supports_gpu_offload())"
IN
log "$IMG: $(docker image inspect $IMG --format '{{.Id}}' | cut -c1-19)"
docker run --rm --runtime nvidia -v /tmp/jp7orin_inner.sh:/c.sh:ro --entrypoint bash $IMG /c.sh 2>&1 \
  | grep -E "^(torch|cv2|lerobot|gst|llama|Traceback|.*Error)" | while read -r l; do log "  $l"; done
log "jp7-orin check DONE"

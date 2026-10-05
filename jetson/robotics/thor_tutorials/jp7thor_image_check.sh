#!/bin/bash
# Runtime check of a Dockerfile.jp7-thor image (needs --runtime nvidia):
#   docker run --rm --runtime nvidia -v $PWD/jp7thor_image_check.sh:/c.sh:ro --entrypoint bash IMAGE /c.sh
echo "== system python"
python - <<'PY'
import torch, torchvision, numpy, cv2, tensorrt, transformers, lerobot, ultralytics
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0), "bf16 matmul ok", bool((x @ x).isfinite().all()))
print("torchvision", torchvision.__version__, "| numpy", numpy.__version__, "| tensorrt", tensorrt.__version__)
print("transformers", transformers.__version__, "| lerobot", lerobot.__version__, "| ultralytics", ultralytics.__version__)
info = cv2.getBuildInformation()
pick = lambda k: next((l.strip() for l in info.splitlines() if l.strip().startswith(k)), k + " ?")
print("cv2", cv2.__version__, "|", pick("GStreamer:"), "|", pick("NVIDIA CUDA:"), "|", pick("NVIDIA GPU arch:"))
print("cv2.cuda devices", cv2.cuda.getCudaEnabledDeviceCount())
g = cv2.cuda_GpuMat(); g.upload(numpy.zeros((480, 640, 3), numpy.uint8))
print("cv2.cuda resize", cv2.cuda.resize(g, (320, 240)).download().shape)
PY
echo "== GStreamer NV plugins (mounted by --runtime nvidia)"
for p in nvv4l2decoder nvv4l2h264enc nvvidconv nvarguscamerasrc; do
  gst-inspect-1.0 $p >/dev/null 2>&1 && echo "  $p OK" || echo "  $p missing"
done
echo "== llama.cpp"
llama-cli --version 2>&1 | head -2
llama-cli --list-devices 2>&1 | grep -i -E "cuda|thor" | head -2
python -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__, 'gpu offload', llama_cpp.llama_supports_gpu_offload())"
echo "== Isaac ROS sources"; ls /opt/ros/isaac_ros_ws/src 2>/dev/null | tr '\n' ' '; echo
echo "== VLA venvs"
bash -c '. /opt/gr00t-venv/bin/activate && python -c "import torch, tensorrt; x=torch.randn(512,512,device=\"cuda\"); print(\"gr00t-venv \", torch.__version__, tensorrt.__version__, bool((x@x).isfinite().all()))"'
/opt/openpi-venv/bin/python -c "import torch, tensorrt, modelopt, transformers, lerobot; print('openpi-venv', torch.__version__, tensorrt.__version__, 'modelopt', modelopt.__version__, 'transformers', transformers.__version__, 'lerobot', lerobot.__version__)"
python -c "import transformers, lerobot; print('system after venvs: transformers', transformers.__version__, 'lerobot', lerobot.__version__)"

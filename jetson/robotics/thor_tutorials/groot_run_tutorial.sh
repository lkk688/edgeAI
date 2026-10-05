#!/bin/bash
# Jetson AI Lab "Isaac GR00T 1.7 on Jetson Thor", steps 8-10, inside the gr00t-thor container
# from the Isaac-GR00T checkout (mounted at /workspace/repo). Needs HF_TOKEN with access to
# the gated nvidia/Cosmos-Reason2-2B. Each step is wrapped in `timeout` because the tutorial
# warns the CUDA-graph teardown can hang after the results are printed.
set -x
cd /workspace/repo
step() { echo; echo "=================== $1 ($(date +%T))"; }
M=checkpoints/GR00T-N1.7-LIBERO/libero_10
COMMON="--model-path $M --dataset-path demo_data/libero_demo --embodiment-tag LIBERO_PANDA"

step "9.1 modelopt"
uv pip install -q "nvidia-modelopt[onnx]==0.39.0" && python -c "import modelopt; print('modelopt', modelopt.__version__)"

step "8 baseline bf16 engines"
[ -d gr00t_trt_baseline/engines ] || timeout 5400 python scripts/deployment/build_trt_pipeline.py $COMMON \
  --output-dir ./gr00t_trt_baseline

step "9.3 optimized mixed NVFP4 engines"
[ -d gr00t_trt_optimized_mixed_nvfp4/engines ] || timeout 5400 python scripts/deployment/build_trt_pipeline.py $COMMON \
  --output-dir ./gr00t_trt_optimized_mixed_nvfp4 --execution-profile optimized --quantization mixed_nvfp4 \
  --calib-dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot --calib-size 10

step "10 real trajectories: PyTorch"
timeout 1800 python scripts/deployment/standalone_inference_script.py $COMMON \
  --traj-ids 0 1 2 3 4 --inference-mode pytorch --execution-horizon 8 --save-plot-path ./output/pytorch_inference.png

step "10 real trajectories: TRT optimized + NVFP4"
timeout 1800 python scripts/deployment/standalone_inference_script.py $COMMON \
  --traj-ids 0 1 2 3 4 --inference-mode trt_full_pipeline --execution-horizon 8 \
  --trt-engine-path ./gr00t_trt_optimized_mixed_nvfp4/engines --save-plot-path ./output/trt_nvfp4_inference.png
step "DONE"

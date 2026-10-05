#!/bin/bash
# Jetson AI Lab "OpenPi pi0.5 on Jetson Thor", steps 5-12, run inside openpi-pi0.5:l4t-jp7.2
# from the openpi checkout (mounted at /workspace). Each step is skipped if its output exists.
set -x
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
export CONFIG_NAME=${CONFIG_NAME:-pi05_libero}
CK=~/.cache/openpi/openpi-assets/checkpoints
cp -r ./src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/
step() { echo; echo "=================== $1 ($(date +%T))"; }

step "6 download JAX checkpoint"
python -c "
import os
from openpi.shared import download
print('Checkpoint downloaded to:', download.maybe_download(f\"gs://openpi-assets/checkpoints/{os.getenv('CONFIG_NAME')}\"))" || exit 6

step "7 JAX -> PyTorch"
[ -f $CK/${CONFIG_NAME}_pytorch/model.safetensors ] || python examples/convert_jax_model_to_pytorch.py \
  --config-name ${CONFIG_NAME} --checkpoint-dir $CK/${CONFIG_NAME} --output-path $CK/${CONFIG_NAME}_pytorch || exit 7

step "8 PyTorch BF16 baseline"
python deployment_scripts/pi05_inference.py --config-name ${CONFIG_NAME} --checkpoint-dir $CK/${CONFIG_NAME}_pytorch \
  --inference-mode pytorch --num-warmup 3 --num-test-runs 5 || exit 8

step "9 ONNX export FP8 + NVFP4"
[ -f $CK/${CONFIG_NAME}_pytorch/onnx/model_fp8_nvfp4.onnx ] || python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CK/${CONFIG_NAME}_pytorch --output_path $CK/${CONFIG_NAME}_pytorch --config_name ${CONFIG_NAME} \
  --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul || exit 9

step "10 TensorRT engine build"
[ -f $CK/${CONFIG_NAME}_pytorch/engine/model_fp8_nvfp4.engine ] || ACTION_HORIZON=10 bash deployment_scripts/build_engine.sh \
  $CK/${CONFIG_NAME}_pytorch/onnx/model_fp8_nvfp4.onnx $CK/${CONFIG_NAME}_pytorch/engine/model_fp8_nvfp4.engine || exit 10

step "11 TensorRT inference"
python deployment_scripts/pi05_inference.py --config-name ${CONFIG_NAME} --checkpoint-dir $CK/${CONFIG_NAME}_pytorch \
  --engine-path $CK/${CONFIG_NAME}_pytorch/engine/model_fp8_nvfp4.engine --inference-mode tensorrt \
  --num-warmup 3 --num-test-runs 10 || exit 11

step "12 compare"
python deployment_scripts/pi05_inference.py --config-name ${CONFIG_NAME} --checkpoint-dir $CK/${CONFIG_NAME}_pytorch \
  --engine-path $CK/${CONFIG_NAME}_pytorch/engine/model_fp8_nvfp4.engine --inference-mode compare || exit 12
step "DONE"

#!/bin/bash
# Re-run the OpenPI pi0.5 Thor tutorial (steps 8-12) inside a unified jp7 image built
# with INSTALL_THOR_VLA=1 (Dockerfile.jp7 9c, or Dockerfile.jp7-thor), using /opt/openpi-venv.
# Pass HF_TOKEN and point HF_HOME at the mounted cache, or calibration falls back to dummy data.
# Reuses the PyTorch checkpoint converted by the official container; writes ONNX/engine to
# a separate directory so the official engine is not overwritten.
# NVFP4=0 exports/builds FP8 only (model_fp8.onnx). With the 25.08 base (TensorRT 10.13)
# the FP8+NVFP4 engine build hangs in Myelin on Thor; FP8 only is the fallback.
set -x
source /opt/openpi-venv/bin/activate
cd /opt/src/openpi
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH CONFIG_NAME=pi05_libero
CK=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
OUT=${OUT:-/root/.cache/openpi/jp7test}
NVFP4=${NVFP4:-1}
if [ "$NVFP4" = "1" ]; then NAME=model_fp8_nvfp4; QFLAGS="--enable_llm_nvfp4 --quantize_attention_matmul"
else NAME=model_fp8; QFLAGS=""; fi
mkdir -p $OUT
echo "== versions"; python -c "import torch, tensorrt, modelopt, transformers; print(torch.__version__, tensorrt.__version__, modelopt.__version__, transformers.__version__)"
echo "== 8 pytorch"; [ "$NVFP4" = "1" ] && python deployment_scripts/pi05_inference.py --config-name $CONFIG_NAME --checkpoint-dir $CK \
  --inference-mode pytorch --num-warmup 3 --num-test-runs 5
echo "== 9 onnx"; [ -f $OUT/onnx/$NAME.onnx ] || python deployment_scripts/pytorch_to_onnx.py --checkpoint_dir $CK \
  --output_path $OUT --config_name $CONFIG_NAME --precision fp8 $QFLAGS || exit 9
echo "== 10 engine"; [ -s $OUT/engine/$NAME.engine ] || ACTION_HORIZON=10 bash deployment_scripts/build_engine.sh \
  $OUT/onnx/$NAME.onnx $OUT/engine/$NAME.engine || exit 10
echo "== 11 trt"; python deployment_scripts/pi05_inference.py --config-name $CONFIG_NAME --checkpoint-dir $CK \
  --engine-path $OUT/engine/$NAME.engine --inference-mode tensorrt --num-warmup 3 --num-test-runs 10 || exit 11
echo "== 12 compare"; python deployment_scripts/pi05_inference.py --config-name $CONFIG_NAME --checkpoint-dir $CK \
  --engine-path $OUT/engine/$NAME.engine --inference-mode compare || exit 12
echo "== DONE"

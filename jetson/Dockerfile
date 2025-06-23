# Jetson-compatible LLM container with PyTorch, Ollama, NanoLLM, OpenCV, TensorRT
# Base image: NVIDIA's Jetson PyTorch with CUDA
FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

LABEL maintainer="kaikai.liu@sjsu.edu"

# Install core system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget unzip ca-certificates gnupg2 \
    python3-pip python3-dev libopencv-dev python3-opencv \
    libprotobuf-dev protobuf-compiler libssl-dev \
    libopenblas-dev libblas-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install \
    opencv-python \
    onnx \
    onnxruntime \
    huggingface_hub \
    jupyterlab \
    transformers \
    accelerate \
    gradio

# Install TensorRT Python bindings (if not already in base image)
RUN pip install tensorrt nvidia-pyindex --extra-index-url https://pypi.nvidia.com

# Install TensorRT-LLM (from source)
# RUN git clone --recursive https://github.com/NVIDIA/TensorRT-LLM.git /opt/TensorRT-LLM && \
#     cd /opt/TensorRT-LLM && \
#     pip install -e .

RUN apt-get update && apt-get install -y --no-install-recommends ninja-build 
# --- Install llama.cpp with cuBLAS ---
RUN git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && make LLAMA_CUBLAS=1

# Install Ollama (Jetson-supported version)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama binary to PATH
ENV PATH="/root/.ollama/bin:$PATH"

# Expose ports for JupyterLab and Ollama API
EXPOSE 8888 11434

# Set working directory
WORKDIR /workspace

# Launch interactive shell by default
CMD ["bash"]
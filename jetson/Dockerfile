FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

LABEL maintainer="kaikai.liu@sjsu.edu"
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Expose ports for JupyterLab, Ollama, llama.cpp server
EXPOSE 8888 11434 8000

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    wget \
    python3-venv \
    python-is-python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python packages: core + ML/NLP/LLM
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tqdm \
    huggingface_hub \
    transformers \
    sentence-transformers \
    langchain \
    tabulate \
    requests \
    fastapi \
    uvicorn \
    jupyterlab

# Clone and build llama.cpp with CUDA/cuBLAS
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . --config Release

# Install llama-cpp-python with CUDA/cuBLAS
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install Ollama (if available for ARM64 — fallback warning)
RUN mkdir -p /opt/ollama && cd /opt/ollama && \
    curl -L https://ollama.com/download/ollama-linux-arm64 -o ollama && \
    chmod +x ollama && mv ollama /usr/local/bin/ollama || echo "⚠️ Ollama ARM64 not available — skip"

# Create shared volume for models
RUN mkdir -p /models /root/.cache

# Default working directory
WORKDIR /workspace

# Default startup command: launch JupyterLab
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

#run this build: docker build -t jetson-llm-lab .

#Run the Container with Ports and Volumes
# docker run --rm -it --runtime nvidia \
#   -p 8888:8888 -p 11434:11434 -p 8000:8000 \
#   -v $(pwd)/models:/models \
#   -v $(pwd)/workspace:/workspace \
#   jetson-llm-lab
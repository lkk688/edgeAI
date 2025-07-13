# 🚀 02: Accelerated Computing on Jetson (CUDA, Numba, NumPy, PyTorch)

Accelerated computing leverages GPUs and other hardware accelerators to significantly improve performance in data processing, AI, and scientific computation. NVIDIA Jetson devices are optimized for this via CUDA and Tensor Cores.

This module introduces key libraries and techniques to write accelerated code using:

* **CUDA (native + via libraries)**
* **Numba** (Python JIT for GPU)
* **NumPy (with CuPy)**
* **PyTorch with GPU acceleration**

---

## 📚 Theoretical Background

Accelerated computing focuses on **parallelism** and **heterogeneous processing**, offloading work from CPU to GPU cores.

### ✅ Why GPUs are Faster:

* CPUs: Few cores, optimized for sequential tasks
* GPUs: Thousands of cores, optimized for data-parallel tasks (matrix ops, convolutions)

### 🔧 Jetson Hardware:

* CUDA cores: For general-purpose parallelism
* Tensor cores: Accelerated AI math (INT8, FP16)
* DLA (Deep Learning Accelerator): Fixed function for inference

### ⏱️ Memory Considerations:

* CUDA global memory: large but slower
* Shared memory: faster but limited to thread blocks
* Memory transfers between CPU (host) and GPU (device) are expensive and must be minimized

### 🧮 Types of Parallelism:

* **Data Parallelism**: Same operation applied to many elements (e.g., matrix multiplication)
* **Task Parallelism**: Different operations executed in parallel (e.g., CNN layers, asynchronous operations)

---

## 🔧 CUDA (C++)
The CUDA Toolkit targets a class of applications whose control part runs as a process on a general purpose computing device, and which use one or more NVIDIA GPUs as coprocessors for accelerating single program, multiple data (SPMD) parallel jobs. Such jobs are self-contained, in the sense that they can be executed and completed by a batch of GPU threads entirely without intervention by the host process, thereby gaining optimal benefit from the parallel graphics hardware.

Jetson supports native CUDA (C/C++) for fine-grained control. Ideal for performance-critical compute kernels.

**NVIDIA CUDA Compiler (NVCC)**: [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) is a compiler driver provided by NVIDIA for compiling CUDA C/C++ programs. It's a toolchain that manages the compilation process, generating binary executables containing both host (CPU) code and device (GPU) code, including PTX and SASS.
    - It works by invoking other tools like a C++ compiler (e.g., g++) and the CUDA runtime library.
    - It's used to compile CUDA code, which is often found in source files with the .cu extension.
    - The output can be C code (for the host) or PTX code (for the device), and potentially directly compiled SASS code. 

The compilation trajectory involves several splitting, compilation, preprocessing, and merging steps for each CUDA source file. It is the purpose of nvcc, the CUDA compiler driver, to hide the intricate details of CUDA compilation from developers. It accepts a range of conventional compiler options, such as for defining macros and include/library paths, and for steering the compilation process. All non-CUDA compilation steps are forwarded to a C++ host compiler that is supported by nvcc, and nvcc translates its options to appropriate host compiler command line options.

> reference: [Cuda C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#)

**Install CUDA nvcc:** Even though JetPack 6.2 (L4T 36.4.3) includes CUDA 12.6, the nvcc command is not installed by default on Jetson devices starting from JetPack 6.x. CUDA is split into host and device components. On Jetson, only the runtime components of CUDA are installed by default (for deploying and running models). The full CUDA toolkit (including nvcc, compiler, samples, etc.) is now optional. 

We already installed the full CUDA toolkit in our provided Jetson image. You can check the `nvcc` command by running:
```bash
sjsujetson@sjsujetson-01:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```
The following CUDA 12.6 path are already added to the `~/.bashrc`
```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

### 🧪 Run CUDA Samples
To build cuda samples, we need to use the new version of cmake (already downloaded in `/Developer`)
```bash
sjsujetson@sjsujetson-01:~$ export PATH=/Developer/cmake-3.28.3-linux-aarch64/bin:$PATH
sjsujetson@sjsujetson-01:~$ cmake --version
cmake version 3.28.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```
`cuda-samples` folder is also available in the `/Developer` folder, you can run cudda samples inside the build folder:
```bash
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./0_Introduction/vectorAdd/vectorAdd
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./1_Utilities/deviceQuery/deviceQuery
...
Result = PASS
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./0_Introduction/matrixMul/matrixMul
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./0_Introduction/asyncAPI/asyncAPI #Demonstrates overlapping data transfers and kernel execution using CUDA streams. 
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams #Demonstrates using Unified Memory with async memory prefetching
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./2_Concepts_and_Techniques/convolutionTexture/convolutionTexture #Demonstrates using CUDA’s texture memory to do efficient image convolution.
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ ./2_Concepts_and_Techniques/histogram/histogram #Computes histograms with/without shared memory. 
```

If you made changes to the sample code, and want to rebuild it, you can use
```bash
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ cmake ../Samples -DCMAKE_CUDA_ARCHITECTURES="72;87"
-- Configuring done (14.4s)
-- Generating done (0.7s)
-- Build files have been written to: /Developer/cuda-samples/build
sjsujetson@sjsujetson-01:/Developer/cuda-samples/build$ make -j$(nproc)
```
CMAKE_CUDA_ARCHITECTURES 72 and 87 is for Jetson Xavier NX and Orin

### ✅  Write CUDA code


---

## 🐍 Numba (Python CUDA JIT)

Numba allows writing Python functions that compile to GPU code via LLVM and CUDA backend.

### ✅ Sample Code

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = a[i] + b[i]

N = 1024
A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)
OUT = np.zeros(N, dtype=np.float32)

add_kernel[32, 32](A, B, OUT)
print("GPU sum[0]:", OUT[0])
```

### 🐳 In Container

```bash
pip install numba
```

Ensure container is run with `--runtime=nvidia`

---

## 📊 NumPy with CuPy (GPU drop-in replacement)

CuPy mimics NumPy's API but uses CUDA arrays and streams.

### ✅ CuPy Example

```python
import cupy as cp

x = cp.arange(1000000).astype(cp.float32)
y = cp.sin(x) ** 2 + cp.cos(x) ** 2
print("Sum:", cp.sum(y))
```

### 🐳 In Container

```bash
pip install cupy-cuda11x  # Replace with correct Jetson CUDA version
```

---

## 🔥 PyTorch on GPU

Jetson comes with optimized PyTorch preinstalled in the NVIDIA container.

### ✅ PyTorch Matrix Multiplication

```python
import torch

A = torch.randn(1000, 1000, device='cuda')
B = torch.randn(1000, 1000, device='cuda')
C = torch.matmul(A, B)
print("Sum:", C.sum())
```

### ✅ Simple NN Training (with loss)

```python
model = torch.nn.Linear(256, 128).cuda()
data = torch.randn(64, 256).cuda()
labels = torch.randn(64, 128).cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
```

---

## 📈 Performance Measurement Tools

### `torch.cuda.Event`

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... run model
end.record()
torch.cuda.synchronize()
print("Time:", start.elapsed_time(end), "ms")
```

### `tegrastats` (outside container)

```bash
sudo tegrastats
```

Use `tegrastats` in host shell to monitor GPU usage, power, memory.

---

## 🧪 Lab: Compare Acceleration Methods

Benchmark this task:

* Matrix multiply (1000x1000)
* Vector add (1M elements)

Compare CPU vs GPU using:

* NumPy vs CuPy
* PyTorch CPU vs CUDA
* Numba kernel

Run each version inside a Jetson container:

```bash
docker run --rm -it --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
```

> Inside container, run Python benchmarks and log output.

Submit:

* Time comparison table
* Code snippet for each method
* Final analysis (speedup ratio, GPU utilization, simplicity)

---

## 📌 Summary Table

| Method   | Language | Easy to use | GPU Speed   | Jetson Support | Notes                          |
| -------- | -------- | ----------- | ----------- | -------------- | ------------------------------ |
| CUDA C++ | C++      | ❌ Low       | ✅ Best      | ✅ Native       | Most control                   |
| Numba    | Python   | ✅ High      | ✅ Good      | ✅ Container OK | Great for math kernels         |
| CuPy     | Python   | ✅ Easy      | ✅ Fast      | ✅ Container OK | NumPy-compatible               |
| PyTorch  | Python   | ✅ Very Easy | ✅ Optimized | ✅ Container OK | Ideal for ML/DL, built-in CUDA |

Jetson Orin Nano supports **multiple layers of acceleration**: CUDA kernels, GPU-enabled Python, and full AI model deployment.

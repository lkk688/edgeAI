# 🧠 CUDA Programming Fundamentals and Jetson CUDA examples
**Author:** Dr. Kaikai Liu, Ph.D.  
**Position:** Associate Professor, Computer Engineering  
**Institution:** San Jose State University  
**Contact:** [kaikai.liu@sjsu.edu](mailto:kaikai.liu@sjsu.edu)

## 🔹 What is CUDA?

**CUDA (Compute Unified Device Architecture)** is NVIDIA's parallel computing platform and programming model that enables developers to harness GPU power for general-purpose computing. Unlike traditional CPU programming, CUDA allows you to write programs that execute thousands of threads simultaneously on the GPU.

### Key CUDA Concepts:

| Concept | Description | Example |
|---------|-------------|----------|
| **Kernel** | Function that runs on GPU | `__global__ void add_vectors(float* a, float* b, float* c)` |
| **Thread** | Basic execution unit | Each thread processes one array element |
| **Block** | Group of threads (up to 1024) | 256 threads per block |
| **Grid** | Collection of blocks | 1000 blocks in a grid |
| **Warp** | 32 threads executed together | Hardware scheduling unit |

### CUDA Memory Hierarchy:

```
┌─────────────────────────────────────────┐
│              Global Memory              │  ← Largest, slowest (LPDDR5)
│                (8GB)                    │
├─────────────────────────────────────────┤
│            L2 Cache (2MB)               │  ← Shared across SMs
├─────────────────────────────────────────┤
│  SM 0    │  SM 1    │  ...  │  SM 15   │
│ ┌─────┐  │ ┌─────┐  │       │ ┌─────┐  │
│ │L1/  │  │ │L1/  │  │       │ │L1/  │  │  ← Fast, per-SM
│ │Shr  │  │ │Shr  │  │       │ │Shr  │  │
│ │64KB │  │ │64KB │  │       │ │64KB │  │
│ └─────┘  │ └─────┘  │       │ └─────┘  │
│ Regs     │ Regs     │       │ Regs     │  ← Fastest, per-thread
│ 64K      │ 64K      │       │ 64K      │
└─────────────────────────────────────────┘
```

## 🎯 Jetson Orin Nano CUDA Specifications

| Feature | Jetson Orin Nano | Desktop RTX 4060 | Data Center A100 |
|---------|------------------|-------------------|-------------------|
| **CUDA Cores** | 512 | 3072 | 6912 |
| **Tensor Cores** | 16 (3rd Gen) | 128 (3rd Gen) | 432 (3rd Gen) |
| **SMs** | 16 | 24 | 108 |
| **Memory** | 8GB LPDDR5 | 8GB GDDR6 | 40/80GB HBM2e |
| **Memory Bandwidth** | 102 GB/s | 272 GB/s | 1555 GB/s |
| **Power** | 7-25W | 115W | 400W |
| **CUDA Compute** | 8.7 | 8.9 | 8.0 |

> **Key Insight**: Jetson Orin Nano provides excellent CUDA capability per watt, making it ideal for edge AI applications where power efficiency is critical.

## 🔧 CUDA Development on Jetson
Jetson supports native CUDA (C/C++) for fine-grained control. Ideal for performance-critical compute kernels.

**NVIDIA CUDA Compiler (NVCC)**: [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) is a compiler driver provided by NVIDIA for compiling CUDA C/C++ programs. It's a toolchain that manages the compilation process, generating binary executables containing both host (CPU) code and device (GPU) code.

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

You can get the cuda architecture from 
```bash
deviceQuery | grep "CUDA Capability"
  CUDA Capability Major/Minor version number:    8.7
```

### NVIDIA CUDA Profiling Tools
#### 1. **NVIDIA Nsight Systems** - Installation in Container
NVIDIA Nsight installation: Nsight system is expected to be install manually if required. Please run below within the container:
```bash
$ sjsujetsontool shell #enter into container
root@sjsujetson-01:/Developer/cuda-samples/build# apt update
\
#download the nsight package from https://repo.download.nvidia.com/jetson/#Jetpack%206.1/6.2/6.2.1
root@sjsujetson-01:/Developer# wget https://repo.download.nvidia.com/jetson/common/pool/main/n/nsight-systems-2024.5.4/nsight-systems-2024.5.4_2024.5.4.34-245434855735v0_arm64.deb
#inspect the package
root@sjsujetson-01:/Developer# dpkg -c nsight-systems-2024.5.4_2024.5.4.34-245434855735v0_arm64.deb
root@sjsujetson-01:/Developer# apt install -y ./nsight-systems-2024.5.4_2024.5.4.34-245434855735v0_arm64.deb
root@sjsujetson-01:/Developer# nsys --version
NVIDIA Nsight Systems version 2024.5.4.34-245434855735v0
root@sjsujetson-01:/Developer# which nsys
/usr/local/bin/nsys
```

[nsight-systems installation](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)

#### 2. **NVIDIA Nsight Systems** - System-wide profiling
```bash
# Profile a CUDA application
root@sjsujetson-01:/Developer/cuda-samples/build# nsys profile --stats=true ./0_Introduction/vectorAdd/vectorAdd
....
Generated:
    /Developer/cuda-samples/build/report2.nsys-rep
    /Developer/cuda-samples/build/report2.sqlite
```
`report2.nsys-rep` is the Main profiling report (for Nsight Systems UI or CLI import/export); `report2.sqlite` is the Internal database format (used for CLI stats and custom scripts)

If you have Nsight Systems GUI installed on your host PC (x86) or remote desktop session, you can Open GUI on Host Machine: `nsys-ui /path/to/report2.nsys-rep`.

You can also get high-level summaries without the GUI:
```bash
root@sjsujetson-01:/Developer/cuda-samples/build# nsys stats --force-export=true /Developer/cuda-samples/build/report2.nsys-rep
Generating SQLite file /Developer/cuda-samples/build/report2.sqlite from /Developer/cuda-samples/build/report2.nsys-rep
Processing [/Developer/cuda-samples/build/report2.sqlite] with [/opt/nvidia/nsight-systems/2024.5.4/host-linux-armv8/reports/nvtx_sum.py]... 
 ** OS Runtime Summary (osrt_sum):
 ** CUDA API Summary (cuda_api_sum):
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):
```
Or export to CSV:
```bash
root@sjsujetson-01:/Developer/cuda-samples/build# nsys stats --format csv --output report2_summary --force-export=true /Developer/cuda-samples/build/report2.nsys-rep
root@sjsujetson-01:/Developer/cuda-samples/build# ls *.csv
report2_summary_cuda_api_sum.csv
report2_summary_cuda_gpu_kern_sum.csv
report2_summary_cuda_gpu_mem_size_sum.csv
report2_summary_cuda_gpu_mem_time_sum.csv
report2_summary_dx11_pix_sum.csv
report2_summary_dx12_gpu_marker_sum.csv
report2_summary_dx12_pix_sum.csv
report2_summary_nvtx_sum.csv
report2_summary_openacc_sum.csv
report2_summary_opengl_khr_gpu_range_sum.csv
report2_summary_opengl_khr_range_sum.csv
report2_summary_openmp_sum.csv
report2_summary_osrt_sum.csv
report2_summary_syscall_sum.csv
report2_summary_um_cpu_page_faults_sum.csv
report2_summary_um_sum.csv
report2_summary_um_total_sum.csv
report2_summary_vulkan_gpu_marker_sum.csv
report2_summary_vulkan_marker_sum.csv
report2_summary_wddm_queue_sum.csv
```

Generate timeline view
```bash
root@sjsujetson-01:/Developer/cuda-samples/build# nsys profile -o my_profile ./0_Introduction/vectorAdd/vectorAdd
Collecting data...
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
Generating '/tmp/nsys-report-d475.qdstrm'
[1/1] [========================100%] my_profile.nsys-rep
Generated:
    /Developer/cuda-samples/build/my_profile.nsys-rep
```
Copy the .nsys-rep file from Jetson to your PC and open with Nsight Systems UI `nsys-ui`

#### 3. **NVIDIA Nsight Compute** - Kernel-level profiling
Install Nvidia Nsight Compute, ncu = Nsight Compute CLI profiler for GPU kernel-level profiling (e.g., occupancy, memory throughput, instruction-level analysis). It’s part of the Nsight Compute package, separate from nsys (Nsight Systems).
```bash
root@sjsujetson-01:/Developer# wget https://repo.download.nvidia.com/jetson/dgpu-rm/pool/main/n/nsight-compute/nsight-compute-2024.3.1_2024.3.1.2-1_arm64.deb
root@sjsujetson-01:/Developer# apt install ./nsight-compute-2024.3.1_2024.3.1.2-1_arm64.deb
root@sjsujetson-01:/Developer# ls /opt/nvidia/nsight-compute/2024.3.1/
docs  extras  host  ncu  ncu-ui  sections  target
root@sjsujetson-01:/Developer# export PATH=/opt/nvidia/nsight-compute/2024.3.1:$PATH
root@sjsujetson-01:/Developer# ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2024 NVIDIA Corporation
Version 2024.3.1.0 (build 34702747) (public-release)
```
Rebuild CUDA sample with debug=1 (`make SMS="all" debug=1`), then run the following
```bash
# Detailed kernel analysis
root@sjsujetson-01:/Developer/cuda-samples/build# ncu --set full ./0_Introduction/vectorAdd/vectorAdd
# Memory throughput analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./0_Introduction/vectorAdd/vectorAdd
```


## 🚀 CUDA Programming Model on Jetson Orin Nano

## 🚀 Complete CUDA Programming Examples

### 1. Vector Addition Example

```bash
#create vector_add.cu - Complete CUDA Vector Addition Example
root@sjsujetson-01:/Developer/mycuda_samples# nano vector_add.cu
root@sjsujetson-01:/Developer/mycuda_samples# nvcc -o vector_add vector_add.cu -arch=sm_87
root@sjsujetson-01:/Developer/mycuda_samples# ls
vector_add  vector_add.cu  vector_multiply.cu
root@sjsujetson-01:/Developer/mycuda_samples# ./vector_add 
Vector addition completed in 72848 microseconds
Result: PASSED
```

### 2. Vector Multiplication Example

```bash
# add vector_multiply.cu - Element-wise Vector Multiplication
root@sjsujetson-01:/Developer/mycuda_samples# nvcc -o vector_multiply vector_multiply.cu -arch=sm_87
root@sjsujetson-01:/Developer/mycuda_samples# ./vector_multiply 
Result: PASSED
```

### 3. Matrix Multiplication Example

```bash
#create matrix_multiply.cu - CUDA Matrix Multiplication
root@sjsujetson-01:/Developer/mycuda_samples# nano matrix_multiply.cu
root@sjsujetson-01:/Developer/mycuda_samples# nvcc -O2 -lineinfo -arch=sm_87 -o matrix_multiply matrix_multiply.cu
./matrix_multiply
Result: PASSED
```

### 4. Advanced Memory Management Example

```bash
# create memory_management.cu - Advanced CUDA Memory Management
root@sjsujetson-01:/Developer/mycuda_samples# nvcc -O2 -lineinfo -arch=sm_87 -o memory_management memory_management.cu
```

## 🛠️ Build Instructions

### 1. Create Makefile

```makefile
# Makefile for CUDA examples
NVCC = nvcc
CUDA_FLAGS = -std=c++11 -O3 -arch=sm_87  # sm_87 for Jetson Orin, adjust for your device
CXX_FLAGS = -std=c++11 -O3

# Targets
all: vector_add vector_multiply matrix_multiply memory_demo

vector_add: vector_add.cu
	$(NVCC) $(CUDA_FLAGS) -o vector_add vector_add.cu

vector_multiply: vector_multiply.cu
	$(NVCC) $(CUDA_FLAGS) -o vector_multiply vector_multiply.cu

matrix_multiply: matrix_multiply.cu
	$(NVCC) $(CUDA_FLAGS) -o matrix_multiply matrix_multiply.cu

memory_demo: memory_management.cu
	$(NVCC) $(CUDA_FLAGS) -o memory_demo memory_management.cu

clean:
	rm -f vector_add vector_multiply matrix_multiply memory_demo

# Debug builds with device debugging enabled
debug: CUDA_FLAGS += -g -G
debug: all

# Profile builds optimized for profiling
profile: CUDA_FLAGS += -lineinfo
profile: all

.PHONY: all clean debug profile
```

### 2. Build Commands

```bash
root@sjsujetson-01:/Developer/mycuda_samples# make
nvcc -std=c++11 -O3 -arch=sm_87   -o vector_add vector_add.cu
nvcc -std=c++11 -O3 -arch=sm_87   -o vector_multiply vector_multiply.cu
nvcc -std=c++11 -O3 -arch=sm_87   -o matrix_multiply matrix_multiply.cu
nvcc -std=c++11 -O3 -arch=sm_87   -o memory_demo memory_management.cu

# Build all examples
make all

# Build individual examples
make vector_add
make matrix_multiply

# Build with debug information for profiling
make profile

# Clean build artifacts
make clean
```

### 3. Alternative CMake Build

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(CudaExamples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 87)  # For Jetson Orin

# Add executables
add_executable(vector_add vector_add.cu)
add_executable(vector_multiply vector_multiply.cu)
add_executable(matrix_multiply matrix_multiply.cu)
add_executable(memory_demo memory_management.cu)

# Set properties
set_property(TARGET vector_add PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET vector_multiply PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET matrix_multiply PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET memory_demo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
```

```bash
# CMake build commands
mkdir build && cd build
cmake ..
make -j4
```

## 📊 Profiling with Nsight Systems

### 1. Install Nsight Systems
Check the previous sections for the Nsight installation for Jetson
```bash
# Verify installation
root@sjsujetson-01:/Developer/mycuda_samples# nsys --version
NVIDIA Nsight Systems version 2024.5.4.34-245434855735v0
```

### 2. Basic Profiling Commands

```bash
# Profile vector addition with detailed GPU metrics
root@sjsujetson-01:/Developer/mycuda_samples# nsys profile --stats=true --force-overwrite=true -o vector_add_profile ./vector_add
Generated:
    /Developer/mycuda_samples/vector_add_profile.nsys-rep
    /Developer/mycuda_samples/vector_add_profile.sqlite

# Profile with CUDA API tracing
root@sjsujetson-01:/Developer/mycuda_samples# nsys profile --trace=cuda,nvtx --stats=true -o detailed_profile ./matrix_multiply
Generated:
    /Developer/mycuda_samples/detailed_profile.nsys-rep
    /Developer/mycuda_samples/detailed_profile.sqlite

# Profile with system-wide metrics
root@sjsujetson-01:/Developer/mycuda_samples# nsys profile --trace=cuda,osrt,nvtx --stats=true --force-overwrite=true -o system_profile ./vector_multiply
Generated:
    /Developer/mycuda_samples/system_profile.nsys-rep
    /Developer/mycuda_samples/system_profile.sqlite

root@sjsujetson-01:/Developer/mycuda_samples# ls
Makefile                   memory_management        vector_add
detailed_profile.nsys-rep  memory_management.cu     vector_add.cu
detailed_profile.sqlite    simple                   vector_add_profile.nsys-rep
matrix_multiply            simple.cu                vector_add_profile.sqlite
matrix_multiply.cu         system_profile.nsys-rep  vector_multiply
memory_demo                system_profile.sqlite    vector_multiply.cu
```

### 3. Advanced Profiling Options

```bash
# Profile with custom duration and sampling
root@sjsujetson-01:/Developer/mycuda_samples# nsys profile --duration=30 --sample=cpu --trace=cuda,osrt -o long_profile ./memory_demo
Collecting data...
[Memory Manager] Starting async transfer demo...
[Memory Manager] Starting unified memory demo...
First 5 squared values: 0 1 4 9 16 
[Memory Manager] Running bandwidth test...
Host-to-Device Bandwidth: 7.44411 GB/s
Generating '/tmp/nsys-report-48f4.qdstrm'
[1/1] [========================100%] long_profile.nsys-rep
Generated:
    /Developer/mycuda_samples/long_profile.nsys-rep

# Profile with environment variables for detailed CUDA info
root@sjsujetson-01:/Developer/mycuda_samples# CUDA_LAUNCH_BLOCKING=1 nsys profile --trace=cuda,nvtx --stats=true -o blocking_profile ./matrix_multiply
Generated:
    /Developer/mycuda_samples/blocking_profile.nsys-rep
    /Developer/mycuda_samples/blocking_profile.sqlite

# Profile with memory transfer analysis
root@sjsujetson-01:/Developer/mycuda_samples# nsys profile --trace=cuda,osrt --cuda-memory-usage=true -o memory_profile ./memory_demo
Collecting data...
[Memory Manager] Starting async transfer demo...
[Memory Manager] Starting unified memory demo...
First 5 squared values: 0 1 4 9 16 
[Memory Manager] Running bandwidth test...
Host-to-Device Bandwidth: 7.35592 GB/s
Generating '/tmp/nsys-report-8aab.qdstrm'
[1/1] [========================100%] memory_profile.nsys-rep
Generated:
    /Developer/mycuda_samples/memory_profile.nsys-rep
```

### 4. Analyzing Profile Results

```bash
# Generate text report
nsys stats vector_add_profile.nsys-rep

# Export to SQLite for custom analysis
nsys export --type=sqlite vector_add_profile.nsys-rep

# View in Nsight Systems GUI (if available)
nsight-sys vector_add_profile.nsys-rep
```

### 5. Code Instrumentation for Better Profiling
This part of the code is not tested.

```cpp
// Add NVTX markers for better profiling visibility
#include <nvtx3/nvToolsExt.h>

void instrumented_vector_add() {
    // Mark the beginning of memory allocation
    nvtxRangePush("Memory Allocation");
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    nvtxRangePop();
    
    // Mark data transfer
    nvtxRangePush("Host to Device Transfer");
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    nvtxRangePop();
    
    // Mark kernel execution
    nvtxRangePush("Kernel Execution");
    vector_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    // Mark result transfer
    nvtxRangePush("Device to Host Transfer");
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    nvtxRangePop();
}
```

### 6. Performance Optimization Tips

```bash
# Check GPU utilization
nsys stats --report=gpukernsum profile.nsys-rep

# Analyze memory bandwidth utilization
nsys stats --report=gpumemtimesum profile.nsys-rep

# Check for optimization opportunities
nsys stats --report=cudaapisum profile.nsys-rep
```

### 7. Automated Profiling Script

```bash
#!/bin/bash
# profile_all.sh - Automated profiling script

echo "Building examples..."
make clean && make profile

echo "Profiling vector addition..."
nsys profile --stats=true --force-overwrite=true -o vector_add_profile ./vector_add

echo "Profiling matrix multiplication..."
nsys profile --stats=true --force-overwrite=true -o matrix_multiply_profile ./matrix_multiply

echo "Generating reports..."
nsys stats vector_add_profile.nsys-rep > vector_add_report.txt
nsys stats matrix_multiply_profile.nsys-rep > matrix_multiply_report.txt

echo "Profiling complete. Check *_report.txt files for results."
```

## 🎯 Performance Benchmarking

### Run and Compare Results

```bash
# Make executable and run
chmod +x profile_all.sh
./profile_all.sh

# Compare CPU vs GPU performance
time ./vector_add    # GPU version
time ./vector_add_cpu  # CPU version (if implemented)

# Monitor GPU usage during execution
#watch -n 1 nvidia-smi #nvidia-smi is not available in jetson
```

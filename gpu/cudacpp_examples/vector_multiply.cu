#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel for element-wise vector multiplication
__global__ void vector_multiply_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

// Optimized version using shared memory for better performance
__global__ void vector_multiply_shared(const float* A, const float* B, float* C, int N) {
    // Shared memory for caching data
    __shared__ float shared_A[256];
    __shared__ float shared_B[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (idx < N) {
        shared_A[tid] = A[idx];
        shared_B[tid] = B[idx];
    }
    
    // Synchronize threads in the block
    __syncthreads();
    
    // Perform computation using shared memory
    if (idx < N) {
        C[idx] = shared_A[tid] * shared_B[tid];
    }
}

void vector_multiply_cuda(const std::vector<float>& h_A, 
                         const std::vector<float>& h_B, 
                         std::vector<float>& h_C) {
    int N = h_A.size();
    size_t bytes = N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    
    // Allocate and copy data
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    vector_multiply_shared<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const int N = 1000000;
    std::vector<float> h_A(N), h_B(N), h_C(N);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    vector_multiply_cuda(h_A, h_B, h_C);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < std::min(10, N); ++i) {
        float expected = h_A[i] * h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << "Result: " << (correct ? "PASSED" : "FAILED") << std::endl;
    return 0;
}
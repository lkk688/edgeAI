#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA kernel for vector addition
// Each thread computes one element: C[i] = A[i] + B[i]
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host function to perform vector addition
void vector_add_cuda(const std::vector<float>& h_A, 
                     const std::vector<float>& h_B, 
                     std::vector<float>& h_C) {
    int N = h_A.size();
    size_t bytes = N * sizeof(float);
    
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // 1. Allocate GPU memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // 2. Copy input data from host to device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    // 3. Configure kernel launch parameters
    int threads_per_block = 256;  // Common choice for good occupancy
    int blocks = (N + threads_per_block - 1) / threads_per_block;  // Ceiling division
    
    // 4. Launch kernel
    vector_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    
    // 5. Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // 6. Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // 7. Copy result back to host
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // 8. Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const int N = 1000000;  // 1 million elements
    
    // Initialize host vectors
    std::vector<float> h_A(N), h_B(N), h_C(N);
    
    // Fill input vectors with test data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_cuda(h_A, h_B, h_C);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Vector addition completed in " << duration.count() << " microseconds" << std::endl;
    
    // Verify results (check first few elements)
    bool correct = true;
    for (int i = 0; i < std::min(10, N); i++) {
        float expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Result: " << (correct ? "PASSED" : "FAILED") << std::endl;
    return 0;
}
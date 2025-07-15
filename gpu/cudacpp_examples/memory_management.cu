#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void square_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= data[idx];
    }
}

// Demonstrate different memory types and management strategies
class CudaMemoryManager {
private:
    float *d_data;
    float *h_pinned_data;
    size_t size_bytes;
    
public:
    CudaMemoryManager(size_t num_elements) {
        size_bytes = num_elements * sizeof(float);
        
        // Allocate device memory
        cudaMalloc(&d_data, size_bytes);
        
        // Allocate pinned (page-locked) host memory for faster transfers
        cudaMallocHost(&h_pinned_data, size_bytes);
    }
    
    ~CudaMemoryManager() {
        // Free device memory
        if (d_data) cudaFree(d_data);
        
        // Free pinned host memory
        if (h_pinned_data) cudaFreeHost(h_pinned_data);
    }
    
    // Asynchronous memory transfer using streams
    void async_transfer_demo(const std::vector<float>& host_data) {
        // Create CUDA streams for overlapping computation and communication
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
        size_t chunk_size = size_bytes / 2;
        
        // Copy first half asynchronously
        std::copy(host_data.begin(), host_data.begin() + host_data.size()/2, h_pinned_data);
        cudaMemcpyAsync(d_data, h_pinned_data, chunk_size, 
                       cudaMemcpyHostToDevice, stream1);
        
        // Copy second half asynchronously
        std::copy(host_data.begin() + host_data.size()/2, host_data.end(), 
                 h_pinned_data + host_data.size()/2);
        cudaMemcpyAsync(d_data + host_data.size()/2, h_pinned_data + host_data.size()/2, 
                       chunk_size, cudaMemcpyHostToDevice, stream2);
        
        // Synchronize streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        
        // Cleanup streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
    
    // Unified Memory example (available on modern GPUs)
    void unified_memory_demo() {
        float *unified_data;
        size_t num_elements = size_bytes / sizeof(float);
        
        // Allocate unified memory accessible from both CPU and GPU
        cudaMallocManaged(&unified_data, size_bytes);
        
        // Initialize data on CPU
        for (size_t i = 0; i < num_elements; i++) {
            unified_data[i] = static_cast<float>(i);
        }
        
        // GPU kernel can directly access unified memory
        int threads_per_block = 256;
        int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        
        // Simple kernel to square each element
        // auto square_kernel = [] __device__ (float* data, int N) {
        //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //     if (idx < N) {
        //         data[idx] = data[idx] * data[idx];
        //     }
        // };
        
        // Launch actual CUDA kernel
        square_kernel<<<blocks, threads_per_block>>>(unified_data, num_elements);

        cudaDeviceSynchronize();
        
        // CPU can directly access the modified data
        std::cout << "First 5 squared values: ";
        for (int i = 0; i < 5; i++) {
            std::cout << unified_data[i] << " ";
        }
        std::cout << std::endl;
        
        cudaFree(unified_data);
    }
    
    // Memory bandwidth benchmark
    void bandwidth_test() {
        const int num_iterations = 100;
        
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Test host-to-device transfer
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            cudaMemcpy(d_data, h_pinned_data, size_bytes, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float h2d_time;
        cudaEventElapsedTime(&h2d_time, start, stop);
        
        // Calculate bandwidth in GB/s
        float h2d_bandwidth = (size_bytes * num_iterations * 1e-6) / h2d_time;
        
        std::cout << "Host-to-Device Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl;
        
        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

int main() {
    const size_t N = 1 << 20;  // 1 million floats

    std::vector<float> host_data(N, 1.23f);  // Initialize with dummy values

    CudaMemoryManager manager(N);
    std::cout << "[Memory Manager] Starting async transfer demo...\n";
    manager.async_transfer_demo(host_data);

    std::cout << "[Memory Manager] Starting unified memory demo...\n";
    manager.unified_memory_demo();

    std::cout << "[Memory Manager] Running bandwidth test...\n";
    manager.bandwidth_test();

    return 0;
}
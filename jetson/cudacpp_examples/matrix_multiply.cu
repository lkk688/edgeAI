#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE_SIZE 16  // Tile size for shared memory optimization

// Basic matrix multiplication kernel
__global__ void matrix_multiply_basic(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row and column
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication using shared memory tiling
__global__ void matrix_multiply_tiled(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    // Shared memory tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const std::vector<float>& h_A, 
                         const std::vector<float>& h_B, 
                         std::vector<float>& h_C,
                         int M, int N, int K) {
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    
    // Allocate GPU memory
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    // Copy input matrices to GPU
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);
    
    // Configure 2D grid and block dimensions
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch optimized kernel
    matrix_multiply_tiled<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Matrix dimensions
    const int M = 64;  // Rows in A and C
    const int K = 64;  // Columns in A and rows in B
    const int N = 64;  // Columns in B and C

    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Fill A and B with test data
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 13);  // Simple pattern
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>((i % 7) - 3);  // Some negatives
    }

    // Run matrix multiplication on GPU
    matrix_multiply_cuda(h_A, h_B, h_C, M, N, K);

    // Verify result (compare against CPU computation)
    bool correct = true;
    for (int row = 0; row < M && correct; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += h_A[row * K + k] * h_B[k * N + col];
            }

            float diff = std::abs(h_C[row * N + col] - expected);
            if (diff > 1e-3f) {
                std::cerr << "Mismatch at (" << row << ", " << col << "): "
                          << "expected " << expected << ", got " << h_C[row * N + col] << std::endl;
                correct = false;
                break;
            }
        }
    }

    std::cout << "Result: " << (correct ? "PASSED" : "FAILED") << std::endl;
    return 0;
}
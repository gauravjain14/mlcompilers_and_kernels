// Softmax kernel using online algorithm
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>


__global__ void softmax_online_v0(const float* x, float* y, int B, int N) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -1e30f;
    extern __shared__ float max_vals[];

    // Each thread processes multiple elements with stride equal to blockDim.x
    for (int idx = threadIdx.x; idx < B; idx += blockDim.x) {
        if (blockIdx.x * B + idx < N) {
            max_val = fmaxf(max_val, x[blockIdx.x * B + idx]);
        }
    }
    
    // Store the thread's local maximum in shared memory
    max_vals[threadIdx.x] = max_val;
    __syncthreads();

    // Parallel reduction to find the maximum value across all threads
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            max_vals[threadIdx.x] = fmaxf(max_vals[threadIdx.x], max_vals[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    // Thread 0 writes the final maximum to the output
    if (threadIdx.x == 0) {
        y[blockIdx.x] = max_vals[0];
    }
}

__global__ void calculate_global_max(float* block_maxes, float* global_max, int num_blocks) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float max_vals[];

    // Assume this kernel is launched with a single block
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        max_val = fmaxf(max_val, block_maxes[i]);
    }
    
    max_vals[threadIdx.x] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            max_vals[threadIdx.x] = fmaxf(max_vals[threadIdx.x], max_vals[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_max[0] = max_vals[0];
    }
}


int main() {
    const int N = 1024;
    const int threadsPerBlock = 64;
    const int num_blocks = 8;
    const int B = N / num_blocks;
    const int sharedMemSize = threadsPerBlock * sizeof(float);

    float* h_input = new float[N];
    float* h_output = new float[1];
    float* h_block_outputs = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;
    float gpu_max = -1e30f;
    cudaError_t cudaStatus = cudaSuccess;
    int all_blocks_correct = 1;

    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    float cpu_max = -1e30f;
    for (int i = 0; i < N; i++) {
        cpu_max = std::max(cpu_max, h_input[i]);
    }
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, num_blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple blocks, each processing a subset of data
    softmax_online_v0<<<num_blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, B, N);
    
    // Launch kernel to calculate global max
    float* d_global_max;
    cudaMalloc(&d_global_max, sizeof(float));
    calculate_global_max<<<1, num_blocks, num_blocks * sizeof(float)>>>(d_output, d_global_max, num_blocks);
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // Allocate host memory for all block results
    h_block_outputs = new float[num_blocks];
    cudaMemcpy(h_block_outputs, d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate and compare blockwise maximums
    for (int i = 0; i < num_blocks; i++) {
        // Calculate the range of elements this block processed
        int start_idx = i * B;
        int end_idx = min((i + 1) * B, N);
        
        // Calculate the expected maximum for this block's data range on CPU
        float expected_block_max = -1e30f;
        for (int j = start_idx; j < end_idx; j++) {
            expected_block_max = std::max(expected_block_max, h_input[j]);
        }
        
        // Compare the GPU block's max with the CPU-calculated max for the same data range
        printf("Block %d (elements %d-%d): GPU max = %f, CPU max = %f\n", 
               i, start_idx, end_idx-1, h_block_outputs[i], expected_block_max);
        
        if (fabs(h_block_outputs[i] - expected_block_max) > 1e-5) {
            printf("  Block %d FAILED! Difference: %f\n", i, fabs(h_block_outputs[i] - expected_block_max));
            all_blocks_correct = 0;
        }
        
        // Update the global max
        gpu_max = std::max(gpu_max, h_block_outputs[i]);
    }

    cudaMemcpy(h_output, d_global_max, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\nGPU final max: %f\n", gpu_max);
    printf("CPU max: %f\n", cpu_max);
    
    if (all_blocks_correct) {
        printf("All blocks PASSED!\n");
    } else {
        printf("Some blocks FAILED!\n");
    }
    
    if (fabs(gpu_max - cpu_max) < 1e-5) {
        printf("Global max test PASSED!\n");
    } else {
        printf("Global max test FAILED! Difference: %f\n", fabs(gpu_max - cpu_max));
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] h_block_outputs;
    
    return 0;
    
Error:
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (h_input) delete[] h_input;
    if (h_output) delete[] h_output;
    if (h_block_outputs) delete[] h_block_outputs;
    return 1;
}
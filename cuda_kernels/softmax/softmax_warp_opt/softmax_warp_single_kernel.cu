// Softmax kernel using blocked max and norm
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>


#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__global__ void unified_softmax(const float* x, float* max, float* sum, int B, int N, int num_blocks, float* y) {
    float max_val = -1e30f;
    float sum_val = 0.0f;
    extern __shared__ float ms[];
    float *max_vals = ms;
    float *sum_shmem = ms + blockDim.x;

    float prev_max = max_val;
    // Each thread processes multiple elements with stride equal to blockDim.x
    for (int idx = threadIdx.x; idx < B; idx += blockDim.x) {
        int global_idx = blockIdx.x * B + idx;
        if (global_idx < N) {
            float new_max = fmaxf(max_val, x[global_idx]);
            sum_val = sum_val * expf(max_val - new_max) + expf(x[global_idx] - new_max);
            max_val = new_max;
        }
    }

    // Store the thread's local maximum in shared memory
    max_vals[threadIdx.x] = max_val;
    sum_shmem[threadIdx.x] = sum_val;
    __syncthreads();

    // Parallel reduction to find the maximum value across all threads
    prev_max = max_vals[threadIdx.x];
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            prev_max = max_vals[threadIdx.x];
            max_vals[threadIdx.x] = fmaxf(max_vals[threadIdx.x], max_vals[threadIdx.x + offset]);
            float sum1 = sum_shmem[threadIdx.x] * expf(prev_max - max_vals[threadIdx.x]);
            float sum2 = sum_shmem[threadIdx.x + offset] * expf(max_vals[threadIdx.x + offset] - max_vals[threadIdx.x]);
            sum_shmem[threadIdx.x] = sum1 + sum2;
        }
        __syncthreads();
    }

    // Thread 0 writes the final maximum to the output
    if (threadIdx.x == 0) {
        max[blockIdx.x] = max_vals[0];
        sum[blockIdx.x] = sum_shmem[0];
    }

    // Ensure that the writes to the output are visible to other blocks at this point.
    // Here we have all the block-wise maxes and sums in shared memory and need to write them to the global memory.
    // Ideally, we could have used cooperative groups to use the distributed shared memory for synchronization but
    // that is only supported on Hopper and above (and I am GPU-poor)
    __threadfence();

    // Number of thread blocks will still remain as the number of blocks launched in the kernel but
    // all but one thread block will do useful work. Others might just do redundant work.
    //
    // Bring block-wise maxes to shared memory because we need it to adjust the block-wise sums to be relative to the global max.
    __shared__ float temp_smem[BLOCK_SIZE];
    max_val = -1e30f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        max_val = fmaxf(max_val, max[i]);
    }
    
    temp_smem[threadIdx.x] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            temp_smem[threadIdx.x] = fmaxf(temp_smem[threadIdx.x], temp_smem[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        max[0] = temp_smem[0];
    }

    sum_val = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum_val += sum[i] * expf(max[i] - max[0]);
    }

    temp_smem[threadIdx.x] = sum_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            temp_smem[threadIdx.x] += temp_smem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sum[0] = temp_smem[0];
    }

    // This is perhaps redundant because we ideally only want to wait for block 0 to finish.
    __threadfence();

    // final softmax computation
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx < N) {
        y[thread_idx] = expf(x[thread_idx] - max[0]) / sum[0];
    }
}



__global__ void calculate_global_sum(float* block_sums,
                                              float* block_maxes,
                                              float* global_sum,
                                              float global_max,
                                              int num_blocks) {
    // Assume this kernel is launched with a single block
    extern __shared__ float sum_vals[];

    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum_val += block_sums[i] * expf(block_maxes[i] - global_max);
    }


    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x & 0x1f;

    const unsigned int active_mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_val += __shfl_down_sync(active_mask, sum_val, offset);
    }

    if (laneId == 0) {
        // overwrite the value in the shared memory with the warp sum
        sum_vals[warpId] = sum_val;
    }
    __syncthreads();

    // Final reduction across warp results - only warp 0 participates
    if (warpId == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float laneVal = (laneId < num_warps)
                        ? sum_vals[laneId]
                        : 0.0f;

        // Use __shfl_down_sync for the final cross-warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            laneVal += __shfl_down_sync(active_mask, laneVal, offset);
        }

        // Finally, lane 0 of warp 0 has the sum of all "warp sums." Write to global_sum[0].
        if (laneId == 0) {
            global_sum[0] = laneVal;
        }
    }
}

__global__ void softmax_kernel(float* x, float* y, float global_max, float global_sum, int N) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx < N) {
        y[thread_idx] = expf(x[thread_idx] - global_max) / global_sum;
    }
}


int main(int argc, char *argv[]) {
    int N = 10000000;
    int threadsPerBlock = 64;
    int num_blocks = 512;

    // Parse command-line arguments
    if (argc == 4) {
        threadsPerBlock = std::atoi(argv[1]);
        num_blocks = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    } else if (argc != 1) { // Allow no arguments for default values
        std::cerr << "Usage: " << argv[0] << " <N> <threadsPerBlock> <num_blocks>" << std::endl;
        std::cerr << "Using default values: N=" << N << ", threadsPerBlock=" << threadsPerBlock << ", num_blocks=" << num_blocks << std::endl;
    }

    // if num_blocks * threadsPerBlock > N, error out
    if (num_blocks * threadsPerBlock > N) {
        std::cerr << "Error: num_blocks * threadsPerBlock > N" << std::endl;
        return -1;
    }

    if (threadsPerBlock > 1024) {
        std::cerr << "Error: threadsPerBlock > 1024" << std::endl;
        return -1;
    }

    std::cout << "N " << N << " threads " << threadsPerBlock << " blocks " << num_blocks << std::endl;

    const int B = (N + num_blocks - 1) / num_blocks;  // Ceiling division
    const int sharedMemSize = 2 * threadsPerBlock * sizeof(float);

    float* h_input = new float[N];
    float* h_output = new float[N]; // For GPU softmax output
    float* h_cpu_output = new float[N]; // For CPU softmax output
    float* d_output = nullptr;
    float* d_input = nullptr;
    float* d_global_max = nullptr;
    float* d_global_sum = nullptr;
    float* block_sum = nullptr;

    float gpu_max = 0.0f;
    float gpu_sum = 0.0f;
    float cpu_max = -1e30f;
    float cpu_sum = 0.0f;
    bool max_correct = false;
    bool sum_correct = false;
    cudaError_t cudaStatus = cudaSuccess;

    bool softmax_correct = true;
    float max_softmax_diff = 0.0f;
    int mismatch_count = 0;
    const int max_mismatches_to_print = 5;
    
    // Declare arrays for CPU calculation
    float cpu_block_maxes[num_blocks];
    float cpu_block_sums[num_blocks];

    // Initialize random input values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f - 2.5f;
    }

    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMalloc(&d_global_sum, sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple blocks, each processing a subset of data
    unified_softmax<<<num_blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_global_max, d_global_sum, B, N, num_blocks, d_output);

    cudaMemcpy(&gpu_max, d_global_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gpu_sum, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate CPU max and sum for verification
    for (int i = 0; i < N; i++) {
        cpu_max = std::max(cpu_max, h_input[i]);
    }

    for (int i = 0; i < N; i++) {
        cpu_sum += expf(h_input[i] - cpu_max);
    }

    // Calculate CPU softmax for verification
    for (int i = 0; i < N; i++) {
        h_cpu_output[i] = expf(h_input[i] - cpu_max) / cpu_sum;
    }
    
    std::cout << "CPU max " << cpu_max << " GPU max " << gpu_max << " GPU sum " << gpu_sum << " CPU sum " << cpu_sum << std::endl;
    
    // Verify max and sum values
    max_correct = (fabs(cpu_max - gpu_max) < 1e-6);
    sum_correct = (fabs(cpu_sum - gpu_sum) < 1e-6);
    
    std::cout << "Max correct: " << (max_correct ? "YES" : "NO") << std::endl;
    std::cout << "Sum correct: " << (sum_correct ? "YES" : "NO") << std::endl;
    
    // compare CPU softmax with GPU softmax
    for (int i = 0; i < N && mismatch_count < max_mismatches_to_print; i++) {
        float diff = fabs(h_cpu_output[i] - h_output[i]);
        max_softmax_diff = std::max(max_softmax_diff, diff);
        
        if (diff > 1e-6) {
            std::cout << "Mismatch at index " << i << " CPU " << h_cpu_output[i] << " GPU " << h_output[i] << " diff " << diff << std::endl;
            mismatch_count++;
            softmax_correct = false;
        }
    }
    
    std::cout << "Softmax validation: " << (softmax_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max softmax difference: " << max_softmax_diff << std::endl;
    std::cout << "Total mismatches found: " << mismatch_count << std::endl;

    cudaFree(d_input);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] block_sum;
    
    return 0;
    
Error:
    if (d_input) cudaFree(d_input);
    if (d_global_max) cudaFree(d_global_max);
    if (d_global_sum) cudaFree(d_global_sum);
    if (d_output) cudaFree(d_output);
    if (h_input) delete[] h_input;
    if (h_output) delete[] h_output;
    if (block_sum) delete[] block_sum;
    return 1;
}
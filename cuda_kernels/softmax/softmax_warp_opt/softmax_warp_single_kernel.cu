// Softmax kernel using blocked max and norm
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>

// Unified Softmax Kernel
// WLP: Helper device function for a warp-level reduction to find the maximum.
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// WLP: Helper device function for a warp-level reduction for a sum.
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void unified_softmax_kernel(
    const float* x, float* y,
    float* g_block_maxes,        // Temporary global storage for max from each block
    float* g_block_sums,         // Temporary global storage for sum from each block
    float* g_final_global_max,   // Output for the final global maximum
    float* g_final_global_sum,   // Output for the final global sum
    int N,                       // Total number of elements
    int elements_per_block       // Number of elements processed by each block
) {
    extern __shared__ float s_data[];
    // WLP: Shared memory is now only for communication between warps, not all threads.
    float* s_max_vals = s_data;
    float* s_sum_vals = s_data + (blockDim.x / 32); // Size is num_warps

    // WLP: Common warp/lane IDs for reductions
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // Phase 1: Calculate block-local max and sum
    float thread_max_val = -1e30f;
    float thread_sum_val = 0.0f;

    int block_start_idx = blockIdx.x * elements_per_block;

    for (int i = threadIdx.x; i < elements_per_block; i += blockDim.x) {
        int current_idx = block_start_idx + i;
        if (current_idx < N) {
            float val = x[current_idx];
            float new_max = fmaxf(thread_max_val, val);
            // Numerically stable online sum calculation
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_max) + expf(val - new_max);
            thread_max_val = new_max;
        }
    }

    // WLP: START of updated block-local reduction
    // This is more complex than a simple sum/max because the combination logic is specific.
    // 1. Intra-warp reduction for the combined max/sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        float partner_max = __shfl_down_sync(0xFFFFFFFF, thread_max_val, offset);
        float partner_sum = __shfl_down_sync(0xFFFFFFFF, thread_sum_val, offset);
        
        if (laneId < offset) { // Active threads combine results
            float new_combined_max = fmaxf(thread_max_val, partner_max);
            thread_sum_val = thread_sum_val * expf(thread_max_val - new_combined_max) + 
                             partner_sum * expf(partner_max - new_combined_max);
            thread_max_val = new_combined_max;
        }
    }

    // 2. Inter-warp reduction
    // Lane 0 of each warp writes its partial result to shared memory
    if (laneId == 0) {
        s_max_vals[warpId] = thread_max_val;
        s_sum_vals[warpId] = thread_sum_val;
    }
    __syncthreads();

    // Warp 0 performs the final reduction on the partials from each warp
    if (warpId == 0) {
        int num_warps = blockDim.x / 32;
        // Load partials from shared memory
        thread_max_val = (laneId < num_warps) ? s_max_vals[laneId] : -1e30f;
        thread_sum_val = (laneId < num_warps) ? s_sum_vals[laneId] : 0.0f;

        // Perform final combined reduction within warp 0
        for (int offset = 16; offset > 0; offset >>= 1) {
            float partner_max = __shfl_down_sync(0xFFFFFFFF, thread_max_val, offset);
            float partner_sum = __shfl_down_sync(0xFFFFFFFF, thread_sum_val, offset);

            if (laneId < offset) {
                float new_combined_max = fmaxf(thread_max_val, partner_max);
                thread_sum_val = thread_sum_val * expf(thread_max_val - new_combined_max) + 
                                 partner_sum * expf(partner_max - new_combined_max);
                thread_max_val = new_combined_max;
            }
        }
    }
    // WLP: END of updated block-local reduction

    if (threadIdx.x == 0) {
        g_block_maxes[blockIdx.x] = thread_max_val;
        g_block_sums[blockIdx.x] = thread_sum_val;
    }

    __syncthreads();
    __threadfence_block();

    // WLP: Phase 2.1: Final global max reduction (only in block 0)
    if (blockIdx.x == 0) {
        float p_max = -1e30f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            p_max = fmaxf(p_max, g_block_maxes[i]);
        }
        
        // WLP: Intra-warp max reduction
        p_max = warp_reduce_max(p_max);
        if (laneId == 0) s_max_vals[warpId] = p_max;
        __syncthreads();
        
        // WLP: Inter-warp max reduction (done by warp 0)
        if (warpId == 0) {
            p_max = (laneId < (blockDim.x / 32)) ? s_max_vals[laneId] : -1e30f;
            p_max = warp_reduce_max(p_max);
        }
        
        float final_max_val_calc;
        if (threadIdx.x == 0) {
            final_max_val_calc = p_max; // The result from the reduction
            g_final_global_max[0] = final_max_val_calc;
        }
        __syncthreads();
        final_max_val_calc = g_final_global_max[0];

        // WLP: Phase 2.2: Final global sum reduction (only in block 0)
        float p_sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            p_sum += g_block_sums[i] * expf(g_block_maxes[i] - final_max_val_calc);
        }
        
        // WLP: Intra-warp sum reduction
        p_sum = warp_reduce_sum(p_sum);
        if (laneId == 0) s_sum_vals[warpId] = p_sum;
        __syncthreads();
        
        // WLP: Inter-warp sum reduction (done by warp 0)
        if (warpId == 0) {
            p_sum = (laneId < (blockDim.x / 32)) ? s_sum_vals[laneId] : 0.0f;
            p_sum = warp_reduce_sum(p_sum);
        }
        
        if (threadIdx.x == 0) {
            g_final_global_sum[0] = p_sum; // The result from the reduction
            __threadfence();
        }
    }

    __syncthreads();

    // Phase 3: Final output calculation (unchanged)
    float final_max = g_final_global_max[0];
    float final_sum = g_final_global_sum[0];

    for (int i = threadIdx.x; i < elements_per_block; i += blockDim.x) {
        int current_idx = block_start_idx + i;
        if (current_idx < N) {
            if (final_sum == 0.0f) {
                 y[current_idx] = 0.0f;
            } else {
                y[current_idx] = expf(x[current_idx] - final_max) / final_sum;
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int N = 10000000;
    int threadsPerBlock = 64;
    int num_blocks = 1024;

    // Parse command-line arguments
    if (argc == 4) {
        threadsPerBlock = std::atoi(argv[1]);
        num_blocks = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    } else if (argc != 1) { // Allow no arguments for default values
        std::cerr << "Usage: " << argv[0] << " <threadsPerBlock> <num_blocks> <N>" << std::endl;
        std::cerr << "Using default values: N=" << N << ", threadsPerBlock=" << threadsPerBlock << ", num_blocks=" << num_blocks << std::endl;
    }
    
    if (threadsPerBlock <= 0 || num_blocks <=0 || N <= 0) {
        std::cerr << "Error: N, threadsPerBlock, and num_blocks must be positive." << std::endl;
        return -1;
    }

    // if num_blocks * threadsPerBlock is too small for N, it's not an error but could be inefficient.
    // However, N must be covered. The elements_per_block calculation handles distribution.

    if (threadsPerBlock > 1024) {
        std::cerr << "Error: threadsPerBlock > 1024" << std::endl;
        return -1;
    }
    
    // Ensure num_blocks is reasonable, e.g., for fitting block_maxes/sums if block 0 reduces them.
    // Current approach has block 0 iterate, so no strict limit on num_blocks for shared memory fitting,
    // but very large num_blocks will make block 0's reduction step long.

    std::cout << "N " << N << " threadsPerBlock " << threadsPerBlock << " num_blocks " << num_blocks << std::endl;

    const int elements_per_block = (N + num_blocks - 1) / num_blocks; // Calculate elements per block for even distribution
    const int sharedMemSize = 2 * threadsPerBlock * sizeof(float); // For s_max_vals and s_sum_vals

    float* h_input = new float[N];
    float* h_output = new float[N]; // For final softmax output from GPU
    float* h_cpu_softmax_output = new float[N]; // For CPU softmax verification

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_block_maxes = nullptr;    // Temp global storage for block maxes
    float* d_block_sums = nullptr;     // Temp global storage for block sums
    float* d_final_global_max = nullptr; // Final global max (scalar)
    float* d_final_global_sum = nullptr; // Final global sum (scalar)

    float gpu_final_max = 0.0f;
    float gpu_final_sum = 0.0f;
    float cpu_max = -1e30f;
    float cpu_sum = 0.0f;
    
    cudaError_t cudaStatus = cudaSuccess;

    // Initialize random input values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f - 2.5f; // Values between -2.5 and 2.5
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    
    // Allocate GPU memory
    cudaStatus = cudaMalloc(&d_input, N * sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_input failed!\n");}
    cudaStatus = cudaMalloc(&d_output, N * sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_output failed!\n"); }
    cudaStatus = cudaMalloc(&d_block_maxes, num_blocks * sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_block_maxes failed!\n"); }
    cudaStatus = cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_block_sums failed!\n"); }
    cudaStatus = cudaMalloc(&d_final_global_max, sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_final_global_max failed!\n"); }
    cudaStatus = cudaMalloc(&d_final_global_sum, sizeof(float));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc d_final_global_sum failed!\n"); }

    // Copy input data to device
    cudaStatus = cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_input failed!\n"); }

    // Launch the unified softmax kernel
    cudaEventRecord(start);
    unified_softmax_kernel<<<num_blocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output,
        d_block_maxes, d_block_sums,
        d_final_global_max, d_final_global_sum,
        N, elements_per_block
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for kernel to complete
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Unified Softmax Kernel time: %f ms\n", ms);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unified softmax kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize(); // Explicit sync after kernel
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after unified kernel failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Copy results back to host
    cudaStatus = cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy h_output failed!\n"); }
    cudaStatus = cudaMemcpy(&gpu_final_max, d_final_global_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy gpu_final_max failed!\n"); }
    cudaStatus = cudaMemcpy(&gpu_final_sum, d_final_global_sum, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy gpu_final_sum failed!\n"); }
    
    // --- CPU Softmax Calculation for Verification ---
    // 1. Calculate CPU global max
    for (int i = 0; i < N; i++) {
        cpu_max = std::max(cpu_max, h_input[i]);
    }

    // 2. Calculate CPU global sum (using the CPU global max)
    cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_sum += expf(h_input[i] - cpu_max);
    }
    
    // 3. Calculate CPU softmax output
    for (int i = 0; i < N; i++) {
        if (cpu_sum == 0.0f) { // Avoid division by zero
             h_cpu_softmax_output[i] = 0.0f; 
        } else {
            h_cpu_softmax_output[i] = expf(h_input[i] - cpu_max) / cpu_sum;
        }
    }
    
    // Optional: Detailed block-wise comparison (if d_block_maxes and d_block_sums are copied from GPU)
    if (false) { // Set to true to enable block-wise debug output
        float* h_block_maxes = new float[num_blocks];
        float* h_block_sums = new float[num_blocks];
        cudaMemcpy(h_block_maxes, d_block_maxes, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Note: CPU block sums here would need to be re-calculated based on the *exact* same
        // online algorithm and block partitioning as the GPU for a fair comparison.
        // The current CPU verification directly computes global max/sum.
        std::cout << "\nGPU Block-wise intermediate results (first few blocks):" << std::endl;
        for (int i = 0; i < std::min(num_blocks, 5); i++) {
            std::cout << "Block " << i << ": GPU max = " << h_block_maxes[i] 
                      << ", GPU sum (for its max) = " << h_block_sums[i] << std::endl;
        }
        delete[] h_block_maxes;
        delete[] h_block_sums;
    }
    
    // Print and compare final global results
    printf("\nGPU final global max: %f, CPU final global max: %f\n", gpu_final_max, cpu_max);
    printf("GPU final global sum: %f, CPU final global sum: %f\n", gpu_final_sum, cpu_sum);
    
    bool max_correct = fabs(gpu_final_max - cpu_max) < 1e-4; // Adjusted tolerance slightly for single vs multi kernel floating point paths
    bool sum_correct = fabs(gpu_final_sum - cpu_sum) < 1e-3; // Sum can accumulate more error
    
    if (max_correct) {
        printf("Global max test PASSED!\n");
    } else {
        printf("Global max test FAILED! Difference: %e\n", fabs(gpu_final_max - cpu_max));
    }
    
    if (sum_correct) {
        printf("Global sum test PASSED!\n");
    } else {
        printf("Global sum test FAILED! Difference: %e\n", fabs(gpu_final_sum - cpu_sum));
    }

    // Compare final softmax output
    bool softmax_correct = true;
    float max_softmax_diff = 0.0f;
    int mismatch_count = 0;
    const int max_mismatches_to_print = 5;

    for (int i = 0; i < N; i++) {
        float diff = fabs(h_output[i] - h_cpu_softmax_output[i]);
        if (diff > 1e-4f) { // Tolerance for softmax elements
            softmax_correct = false;
            if (mismatch_count < max_mismatches_to_print) {
                 printf("Mismatch at index %d: GPU_softmax = %.6e, CPU_softmax = %.6e, Diff = %.3e (input val: %.3f)\n",
                        i, h_output[i], h_cpu_softmax_output[i], diff, h_input[i]);
            }
            mismatch_count++;
        }
        if (diff > max_softmax_diff) {
            max_softmax_diff = diff;
        }
    }

    if (softmax_correct) {
        printf("\nFinal Softmax Output Validation PASSED!\n");
    } else {
        printf("\nFinal Softmax Output Validation FAILED! Max difference: %e. Total mismatches: %d / %d elements.\n", max_softmax_diff, mismatch_count, N);
    }
    fflush(stdout); 

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    cudaFree(d_final_global_max);
    cudaFree(d_final_global_sum);
    delete[] h_input;
    delete[] h_output;
    delete[] h_cpu_softmax_output;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (softmax_correct && max_correct && sum_correct) ? 0 : 1;
}
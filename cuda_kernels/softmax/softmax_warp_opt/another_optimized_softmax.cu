#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>

#define WARP_SIZE 32
#define REDUCE_BLOCK_SIZE 256 


__device__ float block_reduce_sum(float val) {
    extern __shared__ float s_mem[];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x >> 5;  // unnecessary optimization, for fun;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane_id == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_mem[lane_id] : 0.0f;
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}


__device__ float block_reduce_max(float val) {
    extern __shared__ float s_mem[];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    if (lane_id == 0) s_mem[warp_id] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_mem[lane_id] : -1.0e30f;
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
    }
    return val;
}


__global__ void calculate_block_max_and_sum(const float* x, float* block_max, float* block_sum, int N) {
    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float thread_max = -1.0e30f;
    for (int j = i; j < N; j += stride) {
        thread_max = fmaxf(thread_max, x[j]);
    }
    float final_block_max = block_reduce_max(thread_max);
    
    if (threadIdx.x == 0) {
        s_mem[0] = final_block_max;
    }
    __syncthreads();
    final_block_max = s_mem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = i; j < N; j += stride) {
        thread_sum += expf(x[j] - final_block_max);
    }
    float final_block_sum = block_reduce_sum(thread_sum);

    if (tid == 0) {
        block_max[blockIdx.x] = final_block_max;
        block_sum[blockIdx.x] = final_block_sum;
    }
}


__global__ void calculate_global_max_and_sum(float* block_maxes,
                                            float* block_sums,
                                            float* global_max_out,
                                            float* global_sum_out,
                                            int num_blocks) {
    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    
    float thread_max = -1.0e30f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        thread_max = fmaxf(thread_max, block_maxes[i]);
    }
    float global_max = block_reduce_max(thread_max);
    
    if (tid == 0) s_mem[0] = global_max;
    __syncthreads();
    global_max = s_mem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        thread_sum += block_sums[i] * expf(block_maxes[i] - global_max);
    }
    float global_sum = block_reduce_sum(thread_sum);

    if (tid == 0) {
        global_max_out[0] = global_max;
        global_sum_out[0] = global_sum;
    }
}


__global__ void apply_softmax(const float* x, float* y, float global_max, float global_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int j = i; j < N; j += stride) {
       y[j] = expf(x[j] - global_max) / global_sum;
    }
}


int main(int argc, char *argv[]) {
    int N = 1000000;  // Default value
    int threadsPerBlock = 512; // Fixed for optimal performance
    int num_runs_per_kernel = 1;
    int num_blocks = 1024;

    if (argc == 5) {
        threadsPerBlock = std::atoi(argv[1]);
        num_blocks = std::atoi(argv[2]);
        num_runs_per_kernel = std::atoi(argv[3]);
        N = std::atoi(argv[4]);
    } else if (argc != 1) { // Allow no arguments for default values
        std::cerr << "Usage: " << argv[0] << " <N> <threadsPerBlock> <num_blocks> <num_runs_per_kernel>" << std::endl;
        std::cerr << "Using default values: N=" << N << ", threadsPerBlock=" << threadsPerBlock << ", num_blocks=" << num_blocks << ", num_runs_per_kernel=" << num_runs_per_kernel << std::endl;
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

    // num_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "N " << N << " threads " << threadsPerBlock << " blocks " << num_blocks << " num_runs_per_kernel " << num_runs_per_kernel << std::endl;
    
    // Check CUDA grid limits
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxGridSize = deviceProp.maxGridSize[0];
    
    if (num_blocks > maxGridSize) {
        printf("WARNING: Calculated num_blocks (%d) exceeds max grid size (%d)\n", num_blocks, maxGridSize);
        printf("Clamping to maximum and relying on grid-stride loops\n");
        num_blocks = maxGridSize;
    }

    // Because we are using warp-intrinsics, we only need one element per warp in the shared memory.
    const int sharedMemSize = (threadsPerBlock / WARP_SIZE) * sizeof(float);

    float* h_input = new float[N];
    float* h_output = new float[N];
    float* d_output = nullptr;
    float* d_input = nullptr;
    float* d_block_max = nullptr;
    float* d_block_sum = nullptr;
    float* d_global_max = nullptr;
    float* d_global_sum = nullptr;

    float gpu_max = 0.0f;
    float gpu_sum = 0.0f;
    float softmax_sum_gpu = 0.0f;
    float softmax_sum_cpu = 0.0f;

    // Initialize random input values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 5.0f - 2.5f;
    }
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_block_max, num_blocks * sizeof(float));
    cudaMalloc(&d_block_sum, num_blocks * sizeof(float));
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMalloc(&d_global_sum, sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < num_runs_per_kernel; i++) {
        calculate_block_max_and_sum<<<num_blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_block_max, d_block_sum, N);
        cudaDeviceSynchronize();
    }

    const int reduce_threads = REDUCE_BLOCK_SIZE;
    const int reduce_shared_mem = (reduce_threads / WARP_SIZE) * sizeof(float);
    
    for (int i = 0; i < num_runs_per_kernel; i++) {
        calculate_global_max_and_sum<<<1, reduce_threads, reduce_shared_mem>>>(d_block_max, d_block_sum, d_global_max, d_global_sum, num_blocks);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(&gpu_max, d_global_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gpu_sum, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_runs_per_kernel; i++) {
        apply_softmax<<<num_blocks, threadsPerBlock>>>(d_input, d_output, gpu_max, gpu_sum, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for any errors during the copy
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("\n--- Verifying Results ---\n");    
    {
        double cpu_max = -1e30;
        for (int i = 0; i < N; i++) {
            if (h_input[i] > cpu_max) {
                cpu_max = h_input[i];
            }
        }

        double cpu_sum = 0.0;
        for (int i = 0; i < N; i++) {
            cpu_sum += exp(h_input[i] - cpu_max);
        }
        
        float* h_cpu_softmax_output = new float[N];
        for (int i = 0; i < N; i++) {
            h_cpu_softmax_output[i] = expf(h_input[i] - (float)cpu_max) / (float)cpu_sum;
        }

        printf("GPU max: %.6f, CPU max: %.6f\n", gpu_max, (float)cpu_max);
        printf("GPU sum: %.6f, CPU sum: %.6f\n", gpu_sum, (float)cpu_sum);
        printf("Sum difference: %.6f (relative: %.2e)\n", fabs(gpu_sum - cpu_sum), fabs(gpu_sum - cpu_sum) / cpu_sum);
        
        bool max_correct = fabs(gpu_max - cpu_max) < 1e-4; // Loosen tolerance slightly for floating point math
        bool sum_correct = (fabs(gpu_sum - cpu_sum) / cpu_sum) < 1e-6; // Use relative tolerance for sum
        
        if (max_correct) {
            printf("Global max test PASSED!\n");
        } else {
            printf("Global max test FAILED! Difference: %f\n", fabs(gpu_max - cpu_max));
        }
        
        if (sum_correct) {
            printf("Global sum test PASSED!\n");
        } else {
            printf("Global sum test FAILED! Difference: %f\n", fabs(gpu_sum - cpu_sum));
        }

        // print the sum of the gpu softmax output and the sum of the cpu softmax output
        for (int i = 0; i < N; i++) {
            softmax_sum_gpu += h_output[i];
            softmax_sum_cpu += h_cpu_softmax_output[i];
        }
        std::cout << "GPU sum: " << softmax_sum_gpu << " CPU sum: " << softmax_sum_cpu << std::endl;

        bool softmax_correct = true;
        float max_softmax_diff = 0.0f;
        int mismatch_count = 0;
        for (int i = 0; i < N; i++) {
            float diff = fabs(h_output[i] - h_cpu_softmax_output[i]);
            // This is really of no use primarily because the softmax values become so small when the number of elements are too large.
            if (diff > 1e-4f) {
                softmax_correct = false;
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
        
        delete[] h_cpu_softmax_output;
    }
    fflush(stdout);
    
Error:
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_max);
    cudaFree(d_block_sum);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
    
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
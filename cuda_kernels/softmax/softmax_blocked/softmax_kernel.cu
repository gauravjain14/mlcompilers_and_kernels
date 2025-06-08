// Softmax kernel using blocked max and norm
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>


__global__ void calculate_block_max_and_sum(const float* x, float* max, float* sum, int B, int N) {
    float max_val = -1e30f;
    float sum_val = 0.0f;
    extern __shared__ float ms[];
    float *max_vals = ms;
    float *sum_shmem = ms + blockDim.x;

    float prev_max = max_val;
    // Each thread processes multiple elements with stride equal to blockDim.x
    for (int idx = threadIdx.x; idx < B; idx += blockDim.x) {
        if (blockIdx.x * B + idx < N) {
            max_val = fmaxf(max_val, x[blockIdx.x * B + idx]);
            sum_val = sum_val * expf(prev_max - max_val) + expf(x[blockIdx.x * B + idx] - max_val);
            prev_max = max_val;
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
}

__global__ void calculate_global_max(float* block_maxes, float* global_max, int num_blocks) {
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

// calculating the global sum needs the block-wise max and global max
__global__ void calculate_global_sum(float* block_sums, float* block_maxes, float* global_sum, float global_max, int num_blocks) {
    float sum_val = 0.0f;
    // Each block sum is calculated relative to its own block max
    // We need to adjust it to be relative to the global max
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum_val += block_sums[i] * expf(block_maxes[i] - global_max);
    }

    extern __shared__ float sum_vals[];
    sum_vals[threadIdx.x] = sum_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            sum_vals[threadIdx.x] += sum_vals[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[0] = sum_vals[0];
    }
}

__global__ void apply_softmax(const float* x, float* y, float global_max, float global_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Use a grid-stride loop for complete generality
    for (int j = i; j < N; j += stride) {
       y[j] = expf(x[j] - global_max) / global_sum;
    }
}


int main(int argc, char *argv[]) {
    int N = 10000000;
    int threadsPerBlock = 256;
    int num_blocks = 1024;

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

    const int B = N / num_blocks;
    const int sharedMemSize = 2 * threadsPerBlock * sizeof(float);

    float* h_input = new float[N];
    float* h_output = new float[N]; // For final softmax output
    float* d_output = nullptr;
    float* d_input = nullptr;
    float* d_block_max = nullptr;
    float* d_block_sum = nullptr;
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
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_block_max, num_blocks * sizeof(float));
    cudaMalloc(&d_block_sum, num_blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple blocks, each processing a subset of data
    calculate_block_max_and_sum<<<num_blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_block_max, d_block_sum, B, N);
    
    // Allocate memory for block sums for debugging
    float* h_cpu_softmax_output = new float[N]; // For CPU softmax verification
    block_sum = new float[num_blocks];
    cudaMemcpy(block_sum, d_block_sum, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Declare variable before any goto statements to avoid C++ scoping issues
    float cpu_sum_inv = 0.0f;
    float cpu_sum_gpu = 0.0f;
    float gpu_sum_gpu = 0.0f;

    // Launch kernel to calculate global max
    cudaMalloc(&d_global_max, sizeof(float));
    calculate_global_max<<<1, num_blocks, num_blocks * sizeof(float)>>>(d_block_max, d_global_max, num_blocks);

    // Get the global max value from device to use in the next kernel
    cudaMemcpy(&gpu_max, d_global_max, sizeof(float), cudaMemcpyDeviceToHost);
    float host_global_max = gpu_max;  // Use the same value for the kernel call
    
    // Now use the host value in the kernel call
    cudaMalloc(&d_global_sum, sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float)); // Allocate memory for final GPU output
    calculate_global_sum<<<1, num_blocks, num_blocks * sizeof(float)>>>(d_block_sum, d_block_max, d_global_sum, host_global_max, num_blocks);
    
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
    cudaMemcpy(&gpu_sum, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the final softmax kernel
    std::cout << "Launching softmax kernel with grid dimensions: " << num_blocks << " x 1 x 1" << std::endl;
    apply_softmax<<<num_blocks, threadsPerBlock>>>(d_input, d_output, gpu_max, gpu_sum, N);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "softmax_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after softmax_kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy final results back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate CPU max and sum for verification
    for (int i = 0; i < N; i++) {
        cpu_max = std::max(cpu_max, h_input[i]);
    }
    
    // implement the softmax on the cpu
    cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_sum += expf(h_input[i] - cpu_max);
    }
    
    // Calculate CPU softmax output for verification
    cpu_sum_inv = 1.0f / cpu_sum;
    for (int i = 0; i < N; i++) {
        h_cpu_softmax_output[i] = expf(h_input[i] - cpu_max) * cpu_sum_inv;
    }

    // Calculate the sum of the cpu softmax output and the sum of the gpu softmax output and print them.
    for (int i = 0; i < N; i++) {
        cpu_sum_gpu += h_cpu_softmax_output[i];
        gpu_sum_gpu += h_output[i];
    }
    std::cout << "CPU sum: " << cpu_sum_gpu << " GPU sum: " << gpu_sum_gpu << std::endl;

    /*
    // Leaving this here because I am too lazy to remove it.
    // Initialize CPU block sums and maxes
    for (int i = 0; i < num_blocks; i++) {
        cpu_block_sums[i] = 0.0f;
        cpu_block_maxes[i] = -1e30f;
    }
    
    // Calculate block maxes and sums using the online algorithm
    for (int b = 0; b < num_blocks; b++) {
        int start_idx = b * B;
        int end_idx = std::min((b + 1) * B, N);
        
        float prev_max = cpu_block_maxes[b];
        float sum_val = 0.0f;
        
        // Use the same online algorithm as in the GPU code
        for (int i = start_idx; i < end_idx; i++) {
            float curr_max = std::max(cpu_block_maxes[b], h_input[i]);
            sum_val = sum_val * expf(prev_max - curr_max) + expf(h_input[i] - curr_max);
            prev_max = curr_max;
            cpu_block_maxes[b] = curr_max;
        }
        
        cpu_block_sums[b] = sum_val;
    }
    
    for (int b = 0; b < num_blocks; b++) {
        cpu_sum += cpu_block_sums[b] * expf(cpu_block_maxes[b] - cpu_max);
    }
    
    for (int i = 0; i < N; i++) {
        if (cpu_sum == 0.0f) { // Avoid division by zero
             h_cpu_softmax_output[i] = 0.0f; 
        } else {
            h_cpu_softmax_output[i] = expf(h_input[i] - cpu_max) / cpu_sum;
        }
    }
    */
    if (false) {
        std::cout << "\nBlock-wise comparison:" << std::endl;
        for (int i = 0; i < num_blocks; i++) {
            float block_max;
            cudaMemcpy(&block_max, &d_block_max[i], sizeof(float), cudaMemcpyDeviceToHost);
            
            std::cout << "Block " << i << ": GPU max = " << block_max 
                    << ", CPU max = " << cpu_block_maxes[i] << std::endl;
            std::cout << "Block " << i << ": GPU sum = " << block_sum[i] 
                    << ", CPU sum = " << cpu_block_sums[i] << std::endl;
        }
    }
    
    // Print and compare results
    printf("\nGPU max: %f, CPU max: %f\n", gpu_max, cpu_max);
    printf("GPU sum: %f, CPU sum: %f\n", gpu_sum, cpu_sum);
    
    max_correct = fabs(gpu_max - cpu_max) < 1e-5;
    sum_correct = fabs(gpu_sum - cpu_sum) < 1e-5;
    
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

    // print the output elements starting from index 1024 * 256
    for (int i = 1024 * 256; i < 1024 * 256 + 10; i++) {
        std::cout << "Output element " << i << ": " << h_output[i] << std::endl;
        std::cout << "CPU element " << i << ": " << h_cpu_softmax_output[i] << std::endl;
    }


    for (int i = 0; i < N; i++) {
        float diff = fabs(h_output[i] - h_cpu_softmax_output[i]);
        if (diff > 1e-4f) {
            softmax_correct = false;
            // Uncomment to print details of first few mismatches
            // if (mismatch_count < max_mismatches_to_print) {
            //     printf("Mismatch at index %d: GPU_softmax = %f, CPU_softmax = %f, Diff = %f\n",
            //            i, h_output[i], h_cpu_softmax_output[i], diff);
            // }
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
    fflush(stdout); // Ensure validation output is displayed
    
    cudaFree(d_input);
    cudaFree(d_block_max);
    cudaFree(d_block_sum);
    cudaFree(d_global_max);
    cudaFree(d_global_sum);
    if (d_output) cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] block_sum;
    if (h_cpu_softmax_output) delete[] h_cpu_softmax_output;
    
    return 0;
    
Error:
    if (d_input) cudaFree(d_input);
    if (d_block_max) cudaFree(d_block_max);
    if (d_block_sum) cudaFree(d_block_sum);
    if (d_global_max) cudaFree(d_global_max);
    if (d_global_sum) cudaFree(d_global_sum);
    if (d_output) cudaFree(d_output);
    if (h_input) delete[] h_input;
    if (h_output) delete[] h_output;
    if (block_sum) delete[] block_sum;
    if (h_cpu_softmax_output) delete[] h_cpu_softmax_output;
    return 1;
}
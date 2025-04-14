#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#define BLOCK_SIZE 32


__global__ void max_pooling_2d(const float *input,
                float *output,
                int input_width,
                int input_height,
                int output_width,
                int output_height,
                int pool_size,
                int stride,
                int padding)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE];

    /* Use when padding enabled */
    /* if (x == 0 || y == 0 || x == blockDim.x - 1 || y == blockDim.y - 1) {
        shared_input[threadIdx.y][threadIdx.x] = 0;
    } else {
        int addr = (y * input_width + x);
        shared_input[threadIdx.y][threadIdx.x] = input[addr];
    } */

    // assuming no padding, which data points will each thread load?
    // for a threadblock of size 32 x 32, assume that output is 30 x 30 (pool size = 3)
    // and stride = 1.
    // Now, using shared memory we want to load every element only once.
    if (x < input_width && y < input_height)
        shared_input[threadIdx.y][threadIdx.x] = input[y * input_width + x];
    __syncthreads();

    // Now, each thread will compute the max value for its corresponding output pixel
    if (x < output_width && y < output_height) {
        float max_val = std::numeric_limits<float>::min();
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int input_y = threadIdx.y + i;
                int input_x = threadIdx.x + j;
                if (input_x < input_width && input_y < input_height) {
                    max_val = max(max_val, shared_input[input_y][input_x]);
                }
            }
        }
        output[y * output_width + x] = max_val;
    }
}


void launch_max_pooling_kernel(
    const float* input_ptr,
    float* output_ptr,
    int N, int C, int H, int W,         // Input dimensions
    int outH, int outW,                 // Output dimensions
    int pool_size,
    int stride,
    int padding
) {
    // Determine block dimensions based on your specialized kernel's BLOCK_SIZE
    // For the specialized kernel, this MUST match the kernel's BLOCK_SIZE
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Determine grid dimensions. For the specialized kernel, it MUST be 1x1
    dim3 gridDim(1, 1, 1); // ONLY ONE BLOCK for the specialized kernel

    // Loop over batch and channels - launch ONE kernel PER channel plane
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // Calculate pointers for the current input/output channel plane
            const float* current_input_ptr = input_ptr + (n * C + c) * H * W;
            float* current_output_ptr = output_ptr + (n * C + c) * outH * outW;

            // Calculate required output size for this plane
            // For specialized kernel: outH = H - pool_size + 1; outW = W - pool_size + 1;
            // These should match the H, W checks in the C++ caller.

            // Launch the specialized kernel for the current plane
            // Note: We pass W, H as input_width/height and outW, outH as output_width/height
            max_pooling_2d<<<gridDim, blockDim>>>(
                current_input_ptr,
                current_output_ptr,
                W, H,           // Input width, height passed to kernel
                outW, outH,     // Output width, height passed to kernel
                pool_size,
                stride,
                padding
            );

            // Optional: Check error after each launch (more debuggable)
             cudaError_t err = cudaGetLastError();
             if (err != cudaSuccess) {
                 fprintf(stderr, "CUDA kernel launch error in wrapper (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
                 // Consider throwing an exception or returning an error code
                 // TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err)); // If using torch checks
                 throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
             }
        }
    }
     // Optional: Synchronize after all launches if needed, or rely on PyTorch sync
    // cudaDeviceSynchronize();
} // end of wrapper function

/*
int main() {
    // Define input and output dimensions
    int input_width = 32;
    int input_height = 32;
    int output_width = 30;
    int output_height = 30;
    int pool_size = 3;

    // Allocate memory for input and output arrays
    float *input, *output;
    float *d_input, *d_output;

    input = (float *)malloc(input_width * input_height * sizeof(float));
    output = (float *)malloc(output_width * output_height * sizeof(float));

    for (int i=0; i<input_width*input_height; i++) {
        input[i] = i;
    }

    float *ref_output = (int *)malloc(output_width * output_height * sizeof(int));
    // reference implementation
    for (int i=0; i<output_height; i++) {
        for (int j=0; j<output_width; j++) {
            int max_val = std::numeric_limits<int>::min();
            // std::cout << "Output Coordinates " << i << " " << j << std::endl;
            for (int k=0; k<pool_size; k++) {
                for (int l=0; l<pool_size; l++) {
                    int input_x = j + k;
                    int input_y = i + l;
                    max_val = max(ref_output[i * output_width + j],
                        input[input_y * input_width + input_x]);
                }
            }
        }
        ref_output[i * output_width + j] = max_val;
    }

    cudaMalloc(&d_input, input_width * input_height * sizeof(int));
    cudaMalloc(&d_output, output_width * output_height * sizeof(int));

    cudaMemcpy(d_input, input, input_width * input_height * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    max_pooling_2d<<<blocks, threads>>>(d_input,
                                        d_output,
                                        input_width,
                                        input_height,
                                        output_width,
                                        output_height,
                                        pool_size);

    cudaMemcpy(output, d_output, output_width * output_height * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output
    for (int i=0; i<output_width*output_height; i++) {
        printf("%d ", output[i]);
    }
    return 0;
} */
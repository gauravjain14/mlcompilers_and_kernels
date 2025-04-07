// Obviously write a matrix multiplication using shared memory
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 32

// tiled matmul to use shared memory
__global__ void Matmul_kernel(float *Act, float *W, float *Out, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // load the act and weights block corresponding to the output element
    // computed by (threadIdx.y, threadIdx.x) in block (blockIdx.y, blockIdx.x)
    __shared__ float act_shmem[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float w_shmem[BLOCK_SIZE][BLOCK_SIZE];

    // Read BLOCK_SIZE * BLOCK_SIZE elements from Act and W
    // Read needs to happen only once for the entire thread block.
    float partial_sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Boundary checks for loading
        if (row < M && (k + threadIdx.x) < K) {
             act_shmem[threadIdx.y][threadIdx.x] = Act[row * K + (k + threadIdx.x)];
        } else {
             act_shmem[threadIdx.y][threadIdx.x] = 0.0f; // Pad with 0
        }

        if ((k + threadIdx.y) < K && col < N) {
            w_shmem[threadIdx.y][threadIdx.x] = W[(k + threadIdx.y) * N + col];
        } else {
            w_shmem[threadIdx.y][threadIdx.x] = 0.0f; // Pad with 0
        }

        // Synchronize all threads to make sure the shared memory is loaded?
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            partial_sum += act_shmem[threadIdx.y][i] * w_shmem[i][threadIdx.x];
        }

        __syncthreads();
       }

    if (row < M && col < N)
        Out[row * N + col] = partial_sum;
}


int main() {
    int M = 1000;
    int N = 2000;
    int K = 1000;
    
    float *d_Act, *d_W, *d_Out;
    float *Act, *W, *Out;
    
    // Device-side allocation for d_Act, d_W, d_Out
    cudaMalloc(&d_Act, M * K * sizeof(float));
    cudaMalloc(&d_W, K * N * sizeof(float));
    cudaMalloc(&d_Out, M * N * sizeof(float));

    // Host-side allocation for Act, W, Out
    Act = (float*)malloc(M * K * sizeof(float));
    W = (float*)malloc(K * N * sizeof(float));
    Out = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            Act[i * K + j] = i;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            W[i * N + j] = j;
        }
    }

    cudaMemcpy(d_Act, Act, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, K * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    // x are columns, y are rows
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    Matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_Act, d_W, d_Out, M, N, K);

    cudaDeviceSynchronize();    
    cudaMemcpy(Out, d_Out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare with reference
    float *refOut = (float*)malloc(M * N * sizeof(float));
    memset(refOut, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                refOut[i * N + j] += Act[i * K + k] * W[k * N + j];
            }
        }
    }

    bool mismatch = false;
    float tolerance = 1e-5; // Adjust as needed
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Use of fabs suggested by Gemini-2.5-Pro
            if (fabs(Out[i * N + j] - refOut[i * N + j]) > tolerance) {
                std::cout << "Mismatch at (" << i << ", " << j << ")" << std::endl;
                std::cout << "Out: " << Out[i * N + j] << std::endl;
                std::cout << "refOut: " << refOut[i * N + j] << std::endl;
                mismatch = true;
                break;
            }
        }
        if (mismatch) {
            break;
        }
    }

    if (mismatch) {
        std::cout << "Mismatch found!" << std::endl;
    } else {
        std::cout << "No mismatch found!" << std::endl;
    }

    // Free the allocated memory
    free(Act);
    free(W);
    free(refOut);
    free(Out);
    cudaFree(d_Act);
    cudaFree(d_W);
    cudaFree(d_Out);
    
    return 0;
}
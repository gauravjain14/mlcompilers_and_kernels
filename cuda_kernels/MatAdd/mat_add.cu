// 2D addition

#include <iostream>
#include <cuda_runtime.h>

#define N 1024

__global__ void MatAdd(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row * N + col < N * N)
        C[row * N + col] = A[row * N + col] + B[row * N + col];
}

int main() {
    float *A, *B, *C;

    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i;
            B[i * N + j] = j;
        }
    }

    float *refC;
    refC = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            refC[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
    
    // Device-side allocation for d_A, d_B, d_C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool mismatch = false;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != refC[i * N + j]) {
                std::cout << "Mismatch at (" << i << ", " << j << ")" << std::endl;
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
    free(A);
    free(B);
    free(C);
    free(refC);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
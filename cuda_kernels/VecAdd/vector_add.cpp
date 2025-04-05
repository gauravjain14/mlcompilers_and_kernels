#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#define N 1024

// Kernel Definition
__global__ void VecAdd(float *A, float *B, float *C) {
    int i  = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // Initialize input arrays
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // Initialize input arrays on the host
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2*(N-i);
    }

    // Allocate memory for the output array on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy the input arrays to the device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
# CUDA Matrix Multiplication

This directory contains a CUDA implementation of matrix multiplication (C = A * B) which uses
a shared memory implementation.

All the threads in a thread block load the contents of the input matrices to the shared memory before starting
the computation. The size of the shared memory per thread block is `BLOCK_SIZE x BLOCK_SIZE`.

## Implementation Details

*   **[matmul.cu](cci:7://file:///home/gaurav/Work/Compilers/mlcompilers_and_kernels/cuda_kernels/MatMul/matmul.cu:0:0-0:0)**: Contains the CUDA kernel (`Matmul_kernel`) and the host code (`main`).
*   **Kernel (`Matmul_kernel`)**:
    *   Implements a matrix multiplication algorithm.
    *   Each thread in the grid is responsible for computing a single element of the output matrix `C`.
    *   Threads calculate their corresponding row (`r`) and column (`c`) in the output matrix based on `blockIdx` and `threadIdx`.
    *   Each thread iterates through the inner dimension (`K`) of matrices `A` and `B`, accumulating the product into a temporary variable (`val`).
    *   The final accumulated value is written to the corresponding element `C[r * N + c]`.
    *   Grid and block dimensions are configured based on the output matrix dimensions (`M` and `N`) and a `BLOCK_SIZE` constant.
*   **Host Code (`main`)**:
    *   Defines matrix dimensions (`M`, `N`, `K`).
    *   Allocates memory for matrices `A`, `B`, and `C` on the host and device (`d_A`, `d_B`, `d_C`).
    *   Initializes host matrices `A` and `B` with sample data.
    *   Copies input matrices `A` and `B` from host to device.
    *   Launches the `Matmul_kernel` with appropriate grid and block dimensions.
    *   Synchronizes the device to ensure kernel completion.
    *   Copies the result matrix `C` from device to host.
    *   Performs a reference matrix multiplication on the CPU for verification.
    *   Compares the GPU result with the CPU reference result and reports if any mismatch is found within a tolerance.
    *   Frees allocated host and device memory.

## Compilation

You can compile the code using `nvcc`:

```bash
nvcc matmul.cu -o bin/matmul -O3

## Possible Improvements
1. Examine the memory patterns for improving coalescing
2. Can we bring the concepts of cluster arrays for accessing the Distributed Shared Memory?
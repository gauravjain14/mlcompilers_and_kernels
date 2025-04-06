# CUDA Grid-Stride Loops

For a detailed blog, [follow this](https://developer.nvidia.com/blog/cuda-grid-stride-loops/)

## What is a Grid-Stride Loop?

A grid-stride loop is a common programming pattern used in CUDA kernels to process data arrays of arbitrary sizes, even when the size is larger than the total number of threads launched in the grid.

Instead of assigning each thread to process only one element, threads in a grid-stride loop iterate through the data, processing elements at intervals equal to the total number of threads in the grid (`gridDim.x * blockDim.x` for a 1D grid).

## Why Use Grid-Stride Loops?

1.  **Handle Arbitrary Data Sizes:** Kernels are often launched with a grid size that is convenient for the hardware (e.g., a multiple of the number of Streaming Multiprocessors). A grid-stride loop allows this fixed-size grid to process datasets of any size, larger or smaller than the grid itself. If the data size is smaller, some threads simply won't enter the loop. If it's larger, threads will loop multiple times.
2.  **Decouple Grid Size from Data Size:** You can choose an optimal grid size based on the GPU architecture (e.g., to maximize occupancy) without being constrained by the input data size.
3.  **Improve Hardware Utilization:** By keeping threads busy processing multiple elements, grid-stride loops can help maintain high occupancy and better utilize the GPU's computational resources, especially when the number of elements is much larger than a reasonably sized grid.

## How Does it Work?

Each thread calculates its unique global index within the grid. It then enters a loop that continues as long as the current index is less than the total number of elements (`N`) to be processed. Inside the loop, the thread processes the element at the current index. After processing, it increments its index by the grid stride (total number of threads in the grid) and continues the loop.

## Example (1D Vector Addition)

```c++
__global__ void vectorAdd(const float *a, const float *b, float *c, int N) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the grid stride
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop
    for (int i = idx; i < N; i += stride) {
        // Check array bounds (implicitly handled by the loop condition)
        c[i] = a[i] + b[i];
    }
}

int main() {
    // ... (Allocate memory for a, b, c on host and device) ...
    // ... (Copy data from host to device) ...

    int N = 1000000; // Example data size
    int threadsPerBlock = 256;
    // Choose a grid size based on desired occupancy or device capability,
    // not necessarily ceil(N / threadsPerBlock)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Can still use this as a starting point
    // Or choose a fixed number, e.g., based on SM count * blocks per SM
    // int blocksPerGrid = numSMs * 4; // Example heuristic

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // ... (Copy result from device to host) ...
    // ... (Free memory) ...
}
```


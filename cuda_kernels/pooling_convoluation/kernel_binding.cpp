#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


__global__ void max_pooling_2d(
    const float *input, float *output,
    int input_width, int input_height,
    int output_width, int output_height,
    int pool_size, int stride);


void launch_max_pooling_kernel(
    const float* input_ptr,
    float* output_ptr,
    int N, int C, int H, int W,
    int outH, int outW,
    int pool_size,
    int stride,
    int padding
);

// C++ function callable from Python
torch::Tensor max_pool_2d_cuda_forward(
    torch::Tensor input, // Input tensor (N, C, H, W)
    int pool_size,       // Pooling window size (K)
    int stride,           // Stride (S)
    int padding          // Padding (P)
) {
    // Input validation using TORCH_CHECK (preferred)
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be Float");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    // Ensure input tensor is contiguous in memory for easy pointer access per channel
    // Alternatively, handle strides manually, but contiguous is easier.
    input = input.contiguous();

    TORCH_CHECK(pool_size > 0, "Pooling size must be positive");
    TORCH_CHECK(stride > 0, "Stride must be positive");

    // Get input dimensions
    int N = input.size(0); // Batch size
    int C = input.size(1); // Channels
    int H = input.size(2); // Input Height
    int W = input.size(3); // Input Width

    // Calculate output dimensions (assuming no padding, ceil_mode=False)
    // Match PyTorch's default MaxPool2d calculation
    int outH = (H - pool_size) / stride + 1;
    int outW = (W - pool_size) / stride + 1;

    TORCH_CHECK(outH > 0 && outW > 0, "Output dimensions calculated to be non-positive. Check input size, pool size, and stride.");

    // Create the output tensor (same device and data type as input)
    auto output_options = torch::TensorOptions()
                              .dtype(input.dtype())
                              .layout(input.layout())
                              .device(input.device());
    torch::Tensor output = torch::empty({N, C, outH, outW}, output_options);

    // Get data pointers (use the correct type: int)
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    try {
        launch_max_pooling_kernel(
            input_ptr, output_ptr,
            N, C, H, W,
            outH, outW,
            pool_size,
            stride,
            padding
        );
    } catch (const std::runtime_error& e) {
        // Catch potential errors thrown by the wrapper (if you added throws)
        TORCH_CHECK(false, "CUDA kernel execution failed in wrapper: ", e.what());
    }

    // Optional: Synchronize device after all launches are queued if needed,
    // but subsequent PyTorch operations will typically handle this.
    // CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// Binding code using PYBIND11_MODULE
// The module name 'my_custom_maxpool_int_impl' MUST match the 'name' in setup.py
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",                            // Python function name
        &max_pool_2d_cuda_forward,            // C++ function pointer
        "Custom Max Pool 2D Forward (CUDA)"   // Docstring
    );
}
import torch
import torch.nn as nn
import time

# Try importing the custom module. Build it first if this fails.
try:
    # Use the 'name' specified inside CUDAExtension in setup.py
    import custom_maxpool_cuda
except ImportError:
    print("Error: Could not import the custom CUDA module.")
    print("Please build it first by running:")
    print("python setup.py install  OR  python setup.py build_ext --inplace")
    exit()

# --- Parameters ---
N, C, H, W = 8, 16, 32, 32   # Batch size, Channels, Height, Width
K, S, P = 3, 1, 0                 # Kernel size, Stride
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()

print(f"Using device: {device}")
print(f"Input Tensor: N={N}, C={C}, H={H}, W={W}")
print(f"MaxPool2D: Kernel Size={K}, Stride={S}")

# --- Create Input Tensor ---
# Use Int32 data type to match the kernel
# Use random integers within a reasonable range
input_tensor = torch.arange(0, N * C * H * W, device=device, dtype=torch.float).reshape(N, C, H, W)

# Ensure input is contiguous (often good practice, required by our C++ wrapper)
input_tensor = input_tensor.contiguous()

# --- Run Your Custom CUDA Kernel ---
print("\nRunning custom CUDA kernel...")
try:
    start_time = time.time()
    # Call the 'forward' function defined in the C++ binding
    output_custom = custom_maxpool_cuda.forward(input_tensor, K, S, P)
    torch.cuda.synchronize() # Ensure kernel execution finishes for timing
    end_time = time.time()
    print(f"Custom kernel finished in {end_time - start_time:.6f} seconds.")
    print(f"Custom output shape: {output_custom.shape}, dtype: {output_custom.dtype}")

except Exception as e:
    print(f"Error running custom kernel: {e}")
    # Print traceback for more details
    import traceback
    traceback.print_exc()
    exit()

# --- Run PyTorch's MaxPool2d ---
print("\nRunning PyTorch nn.MaxPool2d")

# Define the PyTorch layer - ensure parameters match kernel assumptions
# (kernel_size=K, stride=S, padding=0, dilation=1, ceil_mode=False)
pool_layer = nn.MaxPool2d(kernel_size=K, stride=S, padding=0)
pool_layer = pool_layer.to(device)

start_time = time.time()
# Run the PyTorch layer
output_pytorch = pool_layer(input_tensor)
torch.cuda.synchronize() # Ensure execution finishes for timing
end_time = time.time()

print(f"PyTorch kernel finished in {end_time - start_time:.6f} seconds.")
print(f"PyTorch output shape: {output_pytorch.shape}, dtype: {output_pytorch.dtype}")


# --- Compare Results ---
print("\nComparing outputs...")
if output_custom.shape != output_pytorch.shape:
    print(f"Shape mismatch!")
    print(f"  Custom shape: {output_custom.shape}")
    print(f"  PyTorch shape: {output_pytorch.shape}")
else:
    print(f"Shapes match: {output_custom.shape}")

    # For integer tensors, we can check for exact equality
    are_equal = torch.equal(output_custom, output_pytorch)

    if are_equal:
        print("Outputs are identical!")
    else:
        print("Outputs differ!")
        # Calculate how many elements differ
        diff_count = torch.sum(output_custom != output_pytorch).item()
        total_elements = output_custom.numel()
        print(f"  Number of differing elements: {diff_count} / {total_elements} ({diff_count*100.0/total_elements:.2f}%)")

        # Find the maximum absolute difference
        abs_diff = torch.abs(output_custom - output_pytorch)
        print(f"  Max absolute difference: {torch.max(abs_diff).item()}")

        # Find where they differ (example: first differing element)
        diff_indices = (output_custom != output_pytorch).nonzero(as_tuple=True)
        if len(diff_indices[0]) > 0:
            first_diff_idx = tuple(d[0].item() for d in diff_indices)
            print(f"  First difference at index {first_diff_idx}:")
            print(f"    Custom value: {output_custom[first_diff_idx].item()}")
            print(f"    PyTorch value: {output_pytorch[first_diff_idx].item()}")
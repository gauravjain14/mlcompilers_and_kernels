import torch
import triton

import triton.language as tl

# Supported in PyTorch 2.6 onwards
# from torch.library import triton_op, wrap_triton

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(sci_mode=False)

# @triton.jit(interpret=True)  # uncomment this for debugging
@triton.autotune(configs=[
    # in increasing order of work per thread
    triton.Config(kwargs={'BLOCK_SIZE_X': 32, 'BLOCK_SIZE_Y': 32}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 64}, num_warps=8),
    triton.Config(kwargs={'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 64}, num_warps=4),
  ],
  key=['n_elements_x', 'n_elements_y']
)
@triton.jit
def square_kernel(
    output_ptr,  # Pointer to output matrix
    input_ptr,   # Pointer to input matrix
    n_elements_x,  # Number of elements in x
    n_elements_y,  # Number of elements in y
    BLOCK_SIZE_X: tl.constexpr = 64,  # Number of elements per block_x
    BLOCK_SIZE_Y: tl.constexpr = 64  # Number of elements per block_y
):
    # Get the program ID
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute the start index for this block
    block_start_x = pid_x * BLOCK_SIZE_X
    block_start_y = pid_y * BLOCK_SIZE_Y
    
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)

    # address range to load the input from
    load_store_offsets = offsets_y[:, None] * n_elements_x + offsets_x[None, :]

    mask_x = offsets_x < n_elements_x
    mask_y = offsets_y < n_elements_y
    mask = mask_y[:, None] & mask_x[None, :]
    x = tl.load(input_ptr + load_store_offsets, mask=mask)
    
    # Compute square
    output = x * x
    
    # Store the result
    tl.store(output_ptr + load_store_offsets, output, mask=mask)

# triton_op is supported in PyTorch 2.6 onwards
# @triton_op("customlib::square", mutates_args={})
def square(x):
    # Get input properties
    n_elements_y, n_elements_x = x.shape[0], x.shape[1]
    output = torch.empty_like(x)
    
    # Launch kernel
    # create a grid of 2D blocks
    grid = lambda meta: (
        triton.cdiv(n_elements_x, meta['BLOCK_SIZE_X']),
        triton.cdiv(n_elements_y, meta['BLOCK_SIZE_Y'])
    )
    # wrap_triton is also supported in PyTorch 2.6 onwards
    # wrap_triton(*)
    square_kernel[grid](
        output,
        x,
        n_elements_x,
        n_elements_y
    )
    return output

# Example usage:
if __name__ == "__main__":
    # Create input tensor
    x = torch.randn(1000, 1000, device=device)
    
    # Compute square using Triton
    result_triton = square(x)
    # Verify against PyTorch
    result_torch = x ** 2
    assert torch.max(torch.abs(result_triton - result_torch)) < 1e-5

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=10, repeat=1)
    ) as prof:
        for _ in range(20):
            square(x)
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)
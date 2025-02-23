import torch
import triton
import triton.language as tl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def matrix_add_kernel(
    a_input_ptr,  # Pointer to input matrix
    b_input_ptr,  # Pointer to input matrix
    n_elements_x,  # Number of elements in x
    n_elements_y,  # Number of elements in y
    output_ptr,  # Pointer to output matrix
    BLOCK_SIZE: tl.constexpr = 32,
):
    # Get the program ID
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    # Compute the start index for this block
    block_start_x = pid_x * BLOCK_SIZE
    block_start_y = pid_y * BLOCK_SIZE

    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE)

    # address range to load the input from
    load_store_offsets = offsets_y[:, None] * n_elements_x + offsets_x[None, :]

    mask_x = offsets_x < n_elements_x
    mask_y = offsets_y < n_elements_y
    mask = mask_y[:, None] & mask_x[None, :]
    x = tl.load(a_input_ptr + load_store_offsets, mask=mask)
    y = tl.load(b_input_ptr + load_store_offsets, mask=mask)

    # Compute square
    output = x + y

    # Store the result
    tl.store(output_ptr + load_store_offsets, output, mask=mask)


def matrix_add(a_in, b_in):
    # Add matrices of same size
    assert x.shape == y.shape

    # Get input properties
    n_elements_y, n_elements_x = x.shape[0], x.shape[1]
    output = torch.empty_like(x)

    # Launch kernel
    # create a grid of 2D blocks
    grid = lambda meta: (
        triton.cdiv(n_elements_x, meta['BLOCK_SIZE']),
        triton.cdiv(n_elements_y, meta['BLOCK_SIZE'])
    )
    matrix_add_kernel[grid](
        a_in,
        b_in,
        n_elements_x,
        n_elements_y,
        output
    )


# Compare with torch.compile using torch cuda event benchmarking
def benchmark(a_in, b_in, num_active_steps=20):
    print(f"========== Benchmarking Torch Compile ==========")
    add_fn = torch.compile(torch.add)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=num_active_steps, repeat=1)
    ) as prof:
        for _ in range(num_active_steps + 10):
            out = add_fn(a_in, b_in)
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)

    print(f"========== Benchmarking Triton Kernel ==========")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=num_active_steps, repeat=1)
    ) as prof:
        for _ in range(num_active_steps + 10):
            out = matrix_add(a_in, b_in)
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)

if __name__ == "__main__":
    # Create input tensor
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    benchmark(x, y)

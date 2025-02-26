import math
import torch
import triton
import triton.language as tl


torch.set_printoptions(sci_mode=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def matrix_sum_kernel(x_ptr, z_ptr, M, B0: tl.constexpr=32):
    pid_0 = tl.program_id(0)  # Row index

    # Compute the starting memory address of this row
    row_start = x_ptr + pid_0 * M  # Each row has M elements
    
    # Initialize sum accumulator as a Triton tensor
    row_sum = 0.0
    
    # Compute loop bounds
    loop_max = (M + B0 - 1) // B0  # Number of chunks in a row
    for i in range(loop_max):
        offsets = i * B0 + tl.arange(0, B0)
        mask = offsets < M  # Ensure we don't read out of bounds
        
        x = tl.load(row_start + offsets, mask=mask, other=0.0)  # Load row slice
        row_sum += tl.sum(x)  # Accumulate sum

    # Store row sum in the output array
    tl.store(z_ptr + pid_0, row_sum)

@triton.jit
def final_reduce(x_ptr, z_ptr, M, B0: tl.constexpr=32):
    pid_0 = tl.program_id(0)

    row_start = x_ptr + pid_0 * M
    row_sum = 0.0
    loop_max = (M + B0 - 1) // B0  # Number of chunks in a row
    for i in range(loop_max):
        offsets = i * B0 + tl.arange(0, B0)
        mask = offsets < M  # Ensure we don't read out of bounds
        
        x = tl.load(row_start + offsets, mask=mask, other=0.0)  # Load row slice
        row_sum += tl.sum(x)  # Accumulate sum

    tl.store(z_ptr + pid_0, row_sum)

@triton.jit
def fused_sum_kernel(x_ptr, z_ptr, M, BLOCK_SIZE_M: tl.constexpr=32):
    pid_0 = tl.program_id(0)  # Row index
    num_cols_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    row_start = x_ptr + pid_0 * M
    
    block_sum = 0.0
    # Compute loop bounds
    loop_max = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    for i in range(loop_max):
        # Now, that we know the number of rows per program is > 1, the offsets
        # also need to be calculated accordingly.
        offsets = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offsets < M
        x = tl.load(row_start + offsets, mask=mask, other=0.0)
        block_sum += tl.sum(x)  # Accumulate sum

    tl.atomic_add(z_ptr, block_sum)

# add triton autotune here.
@triton.autotune(
   key=["M"],
   configs=[
       triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4),
       # triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
       # triton.Config({'BLOCK_SIZE_M': 128}, num_warps=8),
   ],
)
@triton.jit
def fused_sum_kernel_block(x_ptr, z_ptr, M, N, BLOCK_SIZE_M: tl.constexpr=128):
    pid_0 = tl.program_id(0)  # Row index
    num_cols_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    start_row = pid_0 * BLOCK_SIZE_M
    
    block_sum = 0.0
    # Compute loop bounds
    loop_max = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    for i in range(loop_max):
        # Now, that we know the number of rows per program is > 1, the offsets
        # also need to be calculated accordingly.
        offsets_x = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offsets_y = start_row + tl.arange(0, BLOCK_SIZE_M)

        addr_range = offsets_y[:, None] * M + offsets_x[None, :]
        mask_x = offsets_x < M
        mask_y = offsets_y < N
        mask = mask_y[:, None] & mask_x[None, :]
        x = tl.load(x_ptr + addr_range, mask=mask, other=0.0)
        block_sum += tl.sum(x)  # Accumulate sum

    tl.atomic_add(z_ptr, block_sum)


# let's add benchmarking here. For comparison, we will use the torch compile
# for reduce sum and compare the performance with Triton kernel above, using
# torch profiler
def benchmarking():
    import time
    import torch
    import torch.autograd.profiler as profiler

    print(f"========== Benchmarking Torch Compile ==========")
    inp_matrix = torch.randn(1000, 1000, device=device)
    matrix = inp_matrix.clone()

    # torch._logging.set_logs(output_code=True)
    torch_reduce_sum = torch.compile(torch.sum)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1)
    ) as prof:
        for _ in range(30):
            torch_reduce_sum(inp_matrix)
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)

    # torch._logging.set_logs(output_code=False)
    # Triton kernel
    print(f"========== Benchmarking Triton Kernel ==========")
    intermediate = torch.zeros((1000,), device=device)
    output = torch.zeros((1,), device=device)
    grid = lambda meta: (triton.cdiv(matrix.shape[0], meta['BLOCK_SIZE_M']),)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1)
    ) as prof:
        for _ in range(30):
            # matrix_sum_kernel[grid](inp_matrix, intermediate, matrix.shape[1])
            # final_reduce[(1, 1, 1)](intermediate, output, intermediate.shape[0])
            fused_sum_kernel_block[grid](matrix, output, matrix.shape[1], matrix.shape[0])
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)


# Initialize input matrix
matrix = torch.randn(1000, 1000, device='cuda')
output_fused = torch.zeros((1,), device='cuda')

# Launch kernel with debug buffer
grid = lambda meta: (triton.cdiv(matrix.shape[0], meta['BLOCK_SIZE_M']),)
fused_sum_kernel_block[grid](matrix, output_fused, matrix.shape[1], matrix.shape[0])

ref_out = matrix.sum()
# print(f"output {output}")
print(f"ref_out {ref_out}")
print(f"output fused {output_fused}")
print(f"Did the kernel match with reference - {torch.allclose(output_fused, ref_out, atol=1e-5)}")
# do benchmarking.
benchmarking()
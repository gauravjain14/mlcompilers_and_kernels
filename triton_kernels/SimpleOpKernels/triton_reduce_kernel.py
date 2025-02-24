import math
import torch
import triton
import triton.language as tl


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
    # using z_ptr or z_ptr + pid_0 should be the same since we are only
    # moving along the row dimension?
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

    # Triton kernel
    print(f"========== Benchmarking Triton Kernel ==========")
    intermediate = torch.zeros((1000,), device=device)
    output = torch.zeros((1,), device=device)
    grid = lambda meta: (triton.cdiv(matrix.shape[0], 1),)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1)
    ) as prof:
        for _ in range(30):
            matrix_sum_kernel[grid](inp_matrix, intermediate, matrix.shape[1])
            final_reduce[(1, 1, 1)](intermediate, output, intermediate.shape[0])
            torch.cuda.synchronize(device)
            prof.step()

        event_list = prof.key_averages()
        print(event_list)


# Initialize input matrix
matrix = torch.randn(1000, 1000, device='cuda')
intermediate = torch.zeros((1000,), device='cuda')
output = torch.zeros((1,), device='cuda')

# Launch kernel
grid = lambda meta: (triton.cdiv(matrix.shape[0], 1),)
matrix_sum_kernel[grid](matrix, intermediate, matrix.shape[1])
final_reduce[(1, 1, 1)](intermediate, output, intermediate.shape[0])

ref_out = matrix.sum()
print(f"Did the kernel match with reference - {torch.allclose(output, ref_out, atol=1e-5)}")

# do benchmarking.
benchmarking()
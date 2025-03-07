import torch
import triton
import triton.language as tl

@triton.jit
def matmul_naive(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in the respective dimension
    stride_am, stride_ak,  # M, K strides for A
    stride_bk, stride_bn,  # K, N strides for B
    stride_cm, stride_cn,  # M, N strides for C
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Compute C = A @ B using tiled matrix multiplication"""
    # The easiest way to do matmul, perhaps not very performant due to repeated
    # loading of the same blocks across multiple threadblocks, is to use the output
    # blocks for partitioning.

    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    block_start_0 = pid_0 * BLOCK_SIZE_M
    block_start_1 = pid_1 * BLOCK_SIZE_N

    # For these offsets, we need to load the row correspoonding to pid_1
    # from A and the column corresponding to pid_0 from B.
    # Row from A
    addr_range_a = pid_1 * stride_am + tl.arange(0, BLOCK_SIZE_K)
    # Column from B
    addr_range_b = pid_0 + tl.arange(0, BLOCK_SIZE_K) * stride_bk

    a_data = tl.load(a_ptr + addr_range_a)
    b_data = tl.load(b_ptr + addr_range_b)

    matmul_output = tl.sum(a_data * b_data)
    out_addr = pid_1 * stride_cm + pid_0 * stride_cn
    tl.store(c_ptr + out_addr, matmul_output)

@triton.jit
def matmul_pmpp(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in the respective dimension
    stride_am, stride_ak,  # M, K strides for A
    stride_bk, stride_bn,  # K, N strides for B
    stride_cm, stride_cn,  # M, N strides for C
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Compute C = A @ B using tiled matrix multiplication"""
    # In this, instead of loading the row from tensor A and column from tensor B,
    # we will load the row from tensor B. The reasons for this is that loading the column
    # from tensor B means that the data we are loading are not coalesced, and so the
    # memory access pattern is not as good as when we load the row from tensor A.
    # But now, loading row from A and B both means we can' really compute the dot product.

    # Ideally, this should be simpler when dealing with larger blocks but for now, assume
    # block depth is 1 row. So, the number of grid blocks launched = M.

    pid_0 = tl.program_id(0)
    block_start_0 = pid_0 * BLOCK_SIZE_N

    # Load the row once from A but K times from B
    addr_range_a = pid_0 * stride_am + tl.arange(0, BLOCK_SIZE_K)
    a_row = tl.load(a_ptr + addr_range_a)

    matmul_output = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(0, K):
        # For a, we need the element A[i, k] and multiply with row K in B
        a_element = tl.load(a_ptr + k * stride_ak + pid_0 * stride_am)
        addr_range_b = k * stride_bk + tl.arange(0, BLOCK_SIZE_K)
        b_data = tl.load(b_ptr + addr_range_b)
        matmul_output += a_element * b_data
    
    out_addr = pid_0 * stride_cm + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + out_addr, matmul_output)

def matmul_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
            M: int, N: int, K: int,
            BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
        ):
    # Compute the reference matmul
    c_ref = matmul_reference(a, b)

    # profile the triton kernel using torch profiler
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        grid = (M, N, 1)
        matmul_naive[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
    print(prof.key_averages())
    # compare c with c_ref with atol=1e-5
    assert torch.allclose(c, c_ref, atol=1e-5)

    # Empty out c to make sure there are no stale contents
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # profile pmpp version
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        grid = (M, 1, 1)
        matmul_pmpp[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
    print(prof.key_averages())
    # compare c with c_ref with atol=1e-5
    assert torch.allclose(c, c_ref, atol=1e-5)

# Create a reference matmul where the M, K, N dimensions are all the same
# and equal to BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N. Use torch matmul.
def matmul_reference(a: torch.Tensor, b: torch.Tensor):
    return a @ b


if __name__ == "__main__":
    M = 128
    K = 128
    N = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 128
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    matmul_triton(a, b, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

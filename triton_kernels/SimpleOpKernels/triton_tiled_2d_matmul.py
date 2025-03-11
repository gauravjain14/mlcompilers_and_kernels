import torch
import triton
import triton.language as tl


@triton.jit
def matmul_tiled_large(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Range of coordinates to write to in C
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_m = pid_m * BLOCK_SIZE_M
    block_range_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    block_range_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)

    # add masks to all the loads
    mask_m = block_range_m < M
    mask_n = block_range_n < N

    # Buffer to store the results of the multiplication
    buffer = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        # load the block from A
        mask_k = k_offsets < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        addr_range_a = (block_range_m * stride_am)[:, None] + (k_offsets)[None, :] * stride_ak
        a_block = tl.load(a_ptr + addr_range_a, mask=mask_a)

        # load the block from B
        mask_b = mask_k[:, None] & mask_n[None, :]
        addr_range_b = (k_offsets * stride_bk)[:, None] + (block_range_n)[None, :] * stride_bn
        b_block = tl.load(b_ptr + addr_range_b, mask=mask_b)

        # Is this masking needed if we have the masks in the loads?
        # a_block = tl.where(mask_a, a_block, 0.0)
        # b_block = tl.where(mask_b, b_block, 0.0)        
        buffer += tl.dot(a_block, b_block)
    
    # write the buffer to the correct position in C with the correct block range
    out_addr = (block_range_m * stride_cm)[:, None] + (block_range_n)[None, :] * stride_cn
    tl.store(c_ptr + out_addr, buffer, mask=mask_m[:, None] & mask_n[None, :])

def matmul_triton(a: torch.Tensor, b: torch.Tensor):
    # Get matrix dimensions
    M, K = a.shape
    _, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Configure block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32  
    BLOCK_SIZE_K = 32

    # Configure grid
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))

    # run the tiled large kernel
    c.zero_()
    print(f"a.shape: {a.shape}, b.shape: {b.shape}, c.shape: {c.shape}")
    print(f"a.stride(0): {a.stride(0)}, a.stride(1): {a.stride(1)}")
    print(f"b.stride(0): {b.stride(0)}, b.stride(1): {b.stride(1)}")
    print(f"c.stride(0): {c.stride(0)}, c.stride(1): {c.stride(1)}")

    matmul_tiled_large[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Compare with PyTorch
    c_ref = torch.matmul(a, b)
    # find maximum absolute error, maximum relative error, and average absolute error
    abs_diff = torch.abs(c - c_ref)
    avg_error = torch.mean(abs_diff).item()
    print(f"max diff: {abs_diff.max().item():.3f}, min diff: {abs_diff.min().item():.3f}")
    print(f"avg_error: {avg_error:.3f}")

    assert torch.allclose(c, c_ref, atol=1e-1, rtol=1e-1)
    
    return c

if __name__ == "__main__":
    # Test with random matrices
    M = 1000
    N = 1000
    K = 1000
    
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    
    # maximum a, b
    a_max = a.max().item()
    b_max = b.max().item()
    a_min = a.min().item()
    b_min = b.min().item()
    print(f"a_max: {a_max:.3f}, b_max: {b_max:.3f}, a_min: {a_min:.3f}, b_min: {b_min:.3f}")

    c = matmul_triton(a, b)
    print("Test passed!")


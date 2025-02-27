import triton
import triton.language as tl
import torch
import torch.nn.functional as F

device = torch.device("cuda")


@triton.jit
def layernorm_kernel(
    x_ptr,          # Input tensor pointer
    mean_ptr,       # Mean tensor pointer 
    var_ptr,        # Variance tensor pointer
    y_ptr,          # Output tensor pointer
    gamma_ptr,      # Scale parameter pointer
    beta_ptr,       # Shift parameter pointer
    stride,         # Stride between rows
    N,              # Row length
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr = 128  # Block size for parallel reduction
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row start
    row_start_ptr = x_ptr + pid * stride
    
    # Initialize mean and variance accumulators
    mean = 0.0
    var = 0.0
    
    # Load and compute mean
    for i in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols + i < N
        x = tl.load(row_start_ptr + cols + i, mask=mask, other=0.0)
        mean += tl.sum(x)
    
    mean = mean / N
    
    # Compute variance
    for i in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols + i < N
        x = tl.load(row_start_ptr + cols + i, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean))
    
    var = var / N
    
    # Store mean and variance
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)
    
    # Normalize and apply scale/shift
    inv_std = 1 / tl.sqrt(var + eps)
    for i in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols + i < N
        
        # Load input and parameters
        x = tl.load(row_start_ptr + cols + i, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + cols + i, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + cols + i, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean) * inv_std
        
        # Scale and shift
        y = gamma * x_norm + beta
        
        # Store result
        tl.store(y_ptr + pid * stride + cols + i, y, mask=mask)

def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """
    Apply layer normalization to input tensor.
    
    Args:
        x: Input tensor of shape [M, N]
        gamma: Scale parameter of shape [N]
        beta: Shift parameter of shape [N]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of shape [M, N]
    """
    M, N = x.shape
    
    # Allocate output tensors
    y = torch.empty_like(x, device=device)
    mean = torch.empty(M, device=x.device, dtype=x.dtype)
    var = torch.empty(M, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (M,)
    layernorm_kernel[grid](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        y_ptr=y,
        gamma_ptr=gamma,
        beta_ptr=beta,
        stride=N,
        N=N,
        eps=eps,
        BLOCK_SIZE=128
    )
    
    return y


if __name__ == "__main__":
    x = torch.randn(1024, 1024, device=device)
    gamma = torch.ones(1024, device=device)
    beta = torch.zeros(1024, device=device)
    y = layernorm(x, gamma, beta)
    
    # reference implementation using pytorch nn layernorm
    layer_norm = torch.nn.LayerNorm(1024, device=device)
    y_ref = layer_norm(x)

    # check if the results are close, with a tolerance of 1e-5
    assert torch.allclose(y, y_ref, atol=1e-5)
    print("Test passed!")

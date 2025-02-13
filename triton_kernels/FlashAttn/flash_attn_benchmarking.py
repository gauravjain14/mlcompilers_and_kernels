# Write a benchmarking script for the triton flash attention kernel and
# compare it with Pytorch Scaled Dot Product Attention.

import torch
from flash_attn_triton import test_op

import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.utils.benchmark as benchmark

def benchmark_pytorch_sdpa(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, num_trials=10):
    """Benchmark PyTorch's Scaled Dot Product Attention"""
    # Create random input tensors
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.bfloat16) 
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    torch.cuda.synchronize()
    
    # Use torch.cuda.event to measure time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Benchmark
    times = []
    # Dispatch the kernel using FLASH_ATTENTION
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        try:
            for _ in range(num_trials):
                start.record()
                F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            # Along with benchmarking using cuda event, I want to benchmark using PyTorch benchmark.
            benchmark_time = benchmark.Timer(
                stmt="F.scaled_dot_product_attention(q, k, v, is_causal=causal)",
                setup="from torch.nn.functional import scaled_dot_product_attention as F",
                globals={"q": q, "k": k, "v": v, "causal": causal}
            ).timeit(num_trials)
            print(f"Benchmarking using PyTorch benchmark: {benchmark_time * 1000} ms")
        except Exception as e:
            print(f"SDPA kernel failed to run: {e}")

    return np.mean(times), np.std(times)

def run_benchmarks():
    """Run benchmarks comparing Triton vs PyTorch implementations"""
    # To begin, let's only use 3 batch sizes, 2 num heads, 2 seq lens, and 2 head dims
    batch_sizes = [16]  #, 32, 64]
    num_heads = [8]  #, 32]
    seq_lens = [2048]  #, 1024]
    head_dims = [64]  #, 128]
    
    results = []
    
    for bs in batch_sizes:
        for nh in num_heads:
            for sl in seq_lens:
                for hd in head_dims:
                    print(f"\nBenchmarking config: bs={bs}, nh={nh}, sl={sl}, hd={hd}")
                    
                    # PyTorch SDPA
                    pt_mean, pt_std = benchmark_pytorch_sdpa(bs, nh, sl, hd, causal=True)
                    
                    # Triton Flash Attention
                    start = time.perf_counter()
                    test_op(BATCH_SIZE=bs, NUM_HEADS=nh, SEQ_LEN=sl, HEAD_DIM=hd, causal=True)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    triton_time = (end - start) * 1000 # Convert to ms
                    
                    results.append({
                        'config': (bs, nh, sl, hd),
                        'pytorch_time': pt_mean,
                        'pytorch_std': pt_std,
                        'triton_time': triton_time
                    })
                    
                    print(f"PyTorch SDPA: {pt_mean:.2f} ms ± {pt_std:.2f}")
                    print(f"Triton Flash: {triton_time:.2f} ms")
                    
    return results

def plot_results(results):
    """Plot benchmark results"""
    configs = [f"bs={r['config'][0]},nh={r['config'][1]},\nsl={r['config'][2]},hd={r['config'][3]}" 
              for r in results]
    pt_times = [r['pytorch_time'] for r in results]
    triton_times = [r['triton_time'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width/2, pt_times, width, label='PyTorch SDPA')
    ax.bar(x + width/2, triton_times, width, label='Triton Flash')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('PyTorch SDPA vs Triton Flash Attention')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('attention_benchmark.png')
    plt.close()


if __name__ == "__main__":
    results = run_benchmarks()
    plot_results(results)
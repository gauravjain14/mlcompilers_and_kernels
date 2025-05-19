# Blocked Softmax CUDA Implementation

This directory contains a CUDA implementation of the softmax function using a blocked algorithm for improved performance. The implementation includes both the CUDA kernel and profiling utilities.

## Components

- `softmax_kernel.cu`: CUDA implementation of the blocked softmax algorithm
- `cuda_profiler.py`: Profiling script to profile CUDA kernels, using `nsys`.

## Softmax Algorithm

The implementation uses a block-based online approach to compute softmax:

1. Each block computes its local maximum and sum using an online algorithm
2. A global maximum is computed from all block maximums
3. Block sums are adjusted based on the global maximum and a global sum is computed from the adjusted block sums
4. The final softmax values are computed using the global maximum and sum


## Compilation
To compile the CUDA kernel:

```bash
nvcc -o softmax_kernel softmax_kernel.cu -O3
```

## Usage

### Running the Softmax Kernel

```bash
./softmax_kernel [threads_per_block] [num_blocks] [num_elements]
```

Parameters:
- `threads_per_block`: Number of threads per block (default: 128)
- `num_blocks`: Number of blocks to use (default: 8)
- `num_elements`: Number of elements in the input array (default: 1024)

### Using the CUDA Profiler

Required arguments:
- `executable`: Path to the CUDA executable to profile

Optional arguments:
- `--threads-per-block`, `-t`: List of threads per block values to test (default: [64, 128, 256])
- `--num-blocks`, `-b`: List of number of blocks values to test (default: [32, 64])
- `--verbose`, `-v`: Enable verbose output
- `--clean`, `-c`: Clean up temporary files after profiling
- `--extra-args`, `-e`: Extra arguments to pass to the executable

Example:
```bash
python cuda_profiler.py ./softmax_kernel -t 64 128 256 -b 32 64 -e 1000000
```

This will profile the softmax kernel with different thread/block configurations and report the best configuration based on execution time. The output from `nsys profile` will generate `.sqlite` and `.nsys-rep` files which will be read by `nsys stats` to generate the final `.csv` outputs.


## Requirements

- CUDA Toolkit (tested with CUDA 12.1)
- Python 3.12.1
- NVIDIA Nsight Systems (`nsys` command-line tool)

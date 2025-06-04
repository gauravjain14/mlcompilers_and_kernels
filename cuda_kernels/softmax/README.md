softmax_blocked/softmax_kernel.cu - depends on shared memory with multi-kernel blocked-online softmax

softmax_warp_opt/softmax_kernel.cu - adds some warp-intrinsic primitives

Using the following configuration of thread block size = 256, number of blocks = 1024, and total elements = 10000000, we observe that replacing shared memory synchronization with warp-level intrinsics in just `calculate_global_sum` doesn't really make a tangible difference.

Profiling the three kernels, i.e. `calculate_block_max_and_sum`, `calculate_global_max`, and `calculate_global_sum` across the two variants, it is evident that `calculate_block_max_and_sum` takes almost 14x more cycles than `calculate_global_sum` for the above configuration and the volume of work for `calculate_global_sum` is also restricted to one Block.


==== Online-blocked softmax with shared memory ====

calculate_block_max_and_sum	    - 137.050 μs
calculate_global_max            - 9.670 μs
calculate_global_sum_warp_down	- 10.310 μs
softmax_kernel                  - 8.120 μs


==== Online-blocked softmax with shared memory and Warp-intrinsics ====

calculate_block_max_and_sum	    - 142.554 μs
calculate_global_max            - 9.310 μs
calculate_global_sum_warp_down	- 10.990 μs
softmax_kernel                  - 8.050 μs


#include <iostream>
#include <cooperative_groups.h>

// Some more information on threadblock clusters
// A threadblock is run on a particular SM.
// A threadblock cluster refers to a group of threadblocks that are cooperating.
// A single threadblock is still constrained to a single SM.

// The H100 has a hierarchical organization of SMs. SMs will be organized into groups.
// A threadblock cluster is constrained that all threadblocks in the cluster
// must be resident on the SMs associated with a single group.

// Clusters enable multiple thread blocks running concurrently across multiple SMs
// to synchronize and collaboratively fetch and exchange data.

// From that same article, I would recommend reading the section
// entitled “Thread block clusters”. A cluster is a group of thread blocks
// that are guaranteed to be concurrently scheduled onto a group of SMs,
// where the goal is to enable efficient cooperation of threads across multiple SMs.

// A thread block cluster can be enabled in a kernel either using a 
// compile-time kernel attribute using __cluster_dims__(X,Y,Z) or using
// the CUDA kernel launch API cudaLaunchKernelEx.

// An important memory abstraction here is the Distributed Shared Memory such that remote
// SMs can access the memory of other SMs.
//
// All of these have the same effect on the threads in a thread block group.
// __syncthreads();
// block.sync();
// cg::synchronize(block);
// this_thread_block().sync();
// cg::synchronize(this_thread_block());
//

namespace cg = cooperative_groups;

__device__ int reduce_sum(cg::thread_group g, int *temp, int val)
{
    // instance to the group of threads
    // How the group is launched here, it is just a threadblock.
    cg::thread_block block = cg::this_thread_block();
    int lane = g.thread_rank();

    for (int i=g.size()/2; i>0; i/=2) {
        temp[lane] = val;
        g.sync();
        if (lane < i) {
            val += temp[lane + i];
        }
        g.sync();
    }

    return val;
}

__device__ int thread_sum(int *input, int n) {
    int sum = 0;

    // We can do vector loads here, but that's for later
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x)
    {
        sum += input[i];
    }

    return sum;
}

__global__ void sum_kernel_block(int *sum, int *input, int n) {
    // All the threads should the first step of partial sums here.
    int block_sum = thread_sum(input, n);
    
    // Now, use the group reduce sum to reduce across the thread blocks
    // in the cluster.
    // extern allows dynamic allocation of shared memory - passed as a
    // configuration parameter during kernel launch
    extern __shared__ int temp[];
    auto g = cg::this_thread_block();

    int cluster_sum = reduce_sum(g, temp, block_sum);

    // Only thread 0 in every group will have the final sum for the group
    if (g.thread_rank() == 0) {
        atomicAdd(sum, cluster_sum);
    }
}


int main() {
    int n = 1<<24;
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    int sharedBytes = blockSize * sizeof(int);

    int *sum, *data;
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&data, n * sizeof(int));
    std::fill_n(data, n, 1); // initialize data
    cudaMemset(sum, 0, sizeof(int));

    sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);

    cudaDeviceSynchronize();

    // Write a reference loop to verify.
    int refSum = 0;
    for (int i = 0; i < n; i++) {
        refSum += data[i];
    }

    if (*sum != refSum) {
        std::cout << "Mismatch found!" << std::endl;
    } else {
        std::cout << "No mismatch found!" << std::endl;
    }
}
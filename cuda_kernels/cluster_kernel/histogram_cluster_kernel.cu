#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>


// Define the CUDA_CHECK macro
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void clusterHist_kernel(int *bins, const int nbins,
                        const int bins_per_block, const int *__restrict__ input,
                        size_t array_size) {
    
    extern __shared__ int smem[];
    int tid = cooperative_groups::this_grid().thread_rank();

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    // The pattern is simple - assuming bins_per_block is a multiple of blockDim.x,
    // each thread in a block will set a bunch of elements to 0
    for (int i=threadIdx.x; i<bins_per_block; i+=blockDim.x) {
        smem[i] = 0;
    }   

    // ensures all the distributed shared memory in a cluster is initialized to zero
    cluster.sync();

    for (int i=tid; i<array_size; i+=blockDim.x*gridDim.x) {
        int ldata = input[i];

        // Find the right histogram bin
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        int dst_block_rank = (int) (binid / bins_per_block);
        int dst_offset = binid % bins_per_block;

        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        atomicAdd(dst_smem + dst_offset, 1);
    }

    cluster.sync();

    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i=threadIdx.x; i<bins_per_block; i+=blockDim.x) {
        atomicAdd(&lbins[i], smem[i]);
    }
}


// Kernel setup and launch
int main() {
    int width = 1024, height = 768;

    int array_size = width * height;
    int threads_per_block = 256;
    int nbins = 512;

    int *input = (int*)malloc(array_size * sizeof(int));
    for (int i = 0; i < array_size; i++) {
        input[i] = (std::rand() % nbins);
    }

    int *bins = (int*)malloc(nbins * sizeof(int));

    int *d_input, *d_bins;
    cudaMalloc(&d_input, array_size * sizeof(int));
    cudaMemcpy(d_input, input, array_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bins, nbins * sizeof(int));
    cudaMemset(d_bins, 0, nbins * sizeof(int));

    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    int cluster_size = 2; // (randomly using)
    int nbins_per_block = nbins / cluster_size;
    
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);
    
    // I have no idea what this does. Just following the example for now.
    CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    config.dynamicSmemBytes));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, d_bins, nbins, nbins_per_block, d_input, (size_t)array_size); // CORRECTED ARGUMENTS
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back from device to host
    CUDA_CHECK(cudaMemcpy(bins, d_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));

    int *ref_bins = (int*)malloc(nbins * sizeof(int));
    memset(ref_bins, 0, nbins);
    // reference bin calculation
    for (int i=0; i<array_size; i++) {
        int bin_id = input[i] / nbins_per_block;
        ref_bins[bin_id] += 1;
    }

    // Now you can check/print the 'bins' array on the host
    for (int i = 0; i < nbins; i++) {
        std::cout << bins[i] << " " << ref_bins[i];
    }
    std::cout << std::endl;

    // --- MISSING: Cleanup ---
    cudaFree(d_input);
    cudaFree(d_bins);
    free(input);
    free(bins);

    return 0; // Assuming you add the missing parts
}
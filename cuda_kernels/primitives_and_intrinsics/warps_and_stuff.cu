#include <stdio.h>


// __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync exchange
// a variable between threads within a warp
__global__ void bcast(int arg, int *fill_values) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    value = __shfl_sync(0xffffffff, value, 0);
    fill_values[threadIdx.x] = value;
}


// __shfl_up_sync -- calculates source lane id by subtracting delta from
// the caller's lane id.
// one use case of this can be that lane 0 values are exchanged with lane 4,
// lane 1 values are exchanged with lane 5 --> This essentially can move values up.
__global__ void exchange(int *fill_values, int delta) {
    int laneId = threadIdx.x & 0x1f;
    int value = fill_values[threadIdx.x];
    value = __shfl_up_sync(0xffffffff, value, delta);
    fill_values[threadIdx.x] = value;
}


__global__ void butterfly(int* input_data, int* output_sums, int num_elements, long long *active_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements) {
        return;
    }

    int laneId = threadIdx.x & 0x1f;
    int calling_value = input_data[tid];

    // Get the active threads mask here before divergence.
    // Could an alternative here be to use __ballot_sync?
    unsigned int active_threads_mask = __activemask();
    // Getting the mask here also gives the right value in active_mask!
    active_mask[0] = active_threads_mask;

    for (int i = 1; i < 32; i <<= 1) {
        // Use the mask 0xFFFFFFFF to ensure that all threads participate in the computation.
        int partner_value = __shfl_xor_sync(active_threads_mask, calling_value, i, 32);
        calling_value += partner_value;
    }

    // After the loop, only lane 0 of each warp will hold the total sum
    // of all values that were originally in that warp.
    if (laneId == 0) {
        output_sums[blockIdx.x] = calling_value;
    }
}


int main() {
    int h_values[32];
    for (int i = 0; i < 32; i++) {
        h_values[i] = i;
    }

    int *d_values;
    cudaMalloc(&d_values, 32 * sizeof(int));
    cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice);

    bcast<<< 1, 32 >>>(1234, d_values);
    cudaDeviceSynchronize();

    cudaMemcpy(h_values, d_values, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; i++) {
        printf("%d ", h_values[i]);
    }   
    printf("\n");

    for (int i = 0; i < 32; i++) {
        h_values[i] = i;
    }

    cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice);
    exchange<<< 1, 32 >>>(d_values, 4);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_values, d_values, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++) {
        printf("%d ", h_values[i]);
    }
    printf("\n");

    for (int i = 0; i < 32; i++) {
        h_values[i] = i;
    }

    // Allocate output array for butterfly results
    int *d_output_sums;
    cudaMalloc(&d_output_sums, sizeof(int));  // Only need 1 int for the sum
    
    cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice);
    long long *d_active_mask;
    cudaMalloc(&d_active_mask, sizeof(long long));
    butterfly<<< 1, 32 >>>(d_values, d_output_sums, 32, d_active_mask);
    cudaDeviceSynchronize();
    
    // Copy back the sum result
    int h_sum;
    cudaMemcpy(&h_sum, d_output_sums, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate expected sum on CPU for verification
    int expected_sum = 0;
    for (int i = 0; i < 32; i++) {
        expected_sum += i;  // Sum of 0+1+2+...+31
    }
    
    printf("Butterfly GPU result: %d\n", h_sum);
    printf("Expected CPU result: %d\n", expected_sum);
    printf("Verification: %s\n", (h_sum == expected_sum) ? "PASS" : "FAIL");
    printf("\n");

    // Test the butterfly reduction with 6 elements
    int h_values_6[12];
    for (int i = 0; i < 12; i++) {
        h_values_6[i] = i;
    }
    int *d_values_6;
    cudaMalloc(&d_values_6, 12 * sizeof(int));
    cudaMemcpy(d_values_6, h_values_6, 12 * sizeof(int), cudaMemcpyHostToDevice);

    long long *d_active_mask_6;
    cudaMalloc(&d_active_mask_6, sizeof(long long));

    int *d_output_sums_6;
    cudaMalloc(&d_output_sums_6, sizeof(int));
    butterfly<<< 1, 12 >>>(d_values_6, d_output_sums_6, 12, d_active_mask_6);
    cudaDeviceSynchronize();

    int h_sum_6;
    cudaMemcpy(&h_sum_6, d_output_sums_6, sizeof(int), cudaMemcpyDeviceToHost);

    long long h_active_mask;
    cudaMemcpy(&h_active_mask, d_active_mask_6, sizeof(long long), cudaMemcpyDeviceToHost);

    int expected_sum_6 = 0;
    for (int i = 0; i < 12; i++) {
        expected_sum_6 += i;
    }

    printf("Butterfly GPU result: %d\n", h_sum_6);
    printf("Expected CPU result: %d\n", expected_sum_6);
    printf("Verification: %s\n", (h_sum_6 == expected_sum_6) ? "PASS" : "FAIL");
    printf("Active mask: %lld\n", h_active_mask);
    printf("\n");

    cudaFree(d_values);
    cudaFree(d_output_sums);
    cudaFree(d_values_6);
    return 0;
}
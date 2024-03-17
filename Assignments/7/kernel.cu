#include "common.h"

#include "timer.h"

#define BLOCK_DIM 1024


__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {
    
    int tidx=threadIdx.x;
    int segment=2*blockIdx.x*blockDim.x;
    __shared__ float buffer_s[2*BLOCK_DIM];

    if (segment + tidx + BLOCK_DIM<N){
        buffer_s[tidx] = input[segment + tidx];
        buffer_s[tidx + BLOCK_DIM] = input[segment + tidx + BLOCK_DIM];
    }
    else if ((segment + tidx) < N){
        buffer_s[tidx] = input[segment + tidx];
        buffer_s[tidx + BLOCK_DIM] = 0;
    }
    else
    {
        buffer_s[tidx] = 0;
        buffer_s[tidx + BLOCK_DIM] = 0;
    }
    __syncthreads();

    for(int stride=1;stride<=BLOCK_DIM;stride*=2){
        int elem_idx=2*(tidx+1)*stride*-1;
        if(elem_idx<2*BLOCK_DIM)
        buffer_s[elem_idx]+=buffer_s[elem_idx-stride];
        __syncthreads();        
    }

    if(tidx==0){
        partialSums[blockIdx.x] = buffer_s[2*BLOCK_DIM-1];
        buffer_s[2*BLOCK_DIM-1]=0;
    }

    __syncthreads();

    for(int stride=BLOCK_DIM;stride>=1;stride/=2){

        int elem_idx=2*(tidx+1)*stride-1;
        if (elem_idx < 2 * BLOCK_DIM){
            float temp = buffer_s[elem_idx];
            buffer_s[elem_idx] += buffer_s[elem_idx - stride];
            buffer_s[elem_idx - stride] = temp;
        }
        __syncthreads();
    }

    if ((segment + tidx + BLOCK_DIM) < N){
        output[segment + tidx] = buffer_s[tidx];
        output[segment + tidx + BLOCK_DIM] = buffer_s[tidx + BLOCK_DIM];
    }
    else if ((segment + tidx) < N){
        output[segment + tidx] = buffer_s[tidx];
    }

}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int tidx=threadIdx.x;
    unsigned int segment = 2 * blockIdx.x * blockDim.x;

    if (blockIdx.x > 0){
        if ((segment + tidx + BLOCK_DIM) < N){
            output[segment + tidx] += partialSums[blockIdx.x];
            output[segment + tidx + BLOCK_DIM] += partialSums[blockIdx.x];
        }
        else if ((segment + tidx) < N){
            output[segment + tidx] += partialSums[blockIdx.x];
        }
    }
}


void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {

    Timer timer;

    // Configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    // Allocate partial sums
    startTime(&timer);
    float *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

    // Call kernel
    startTime(&timer);
    scan_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Scan partial sums then add
    if(numBlocks > 1) {

        // Scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // Add scanned sums
        add_kernel <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N);

    }

    // Free memory
    startTime(&timer);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}


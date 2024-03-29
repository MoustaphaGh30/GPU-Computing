#include "common.h"

#include "timer.h"

#define BLOCK_DIM 1024
#define WARP_SIZE 32
__global__ void reduce_kernel(float* input, float* sum, unsigned int N) {

    __shared__ float buffer_s[BLOCK_DIM];

    unsigned int tidx=threadIdx.x;
    unsigned int segment=blockDim.x*blockIdx.x*2;
    unsigned int i=segment+tidx+BLOCK_DIM;

    if(i<N){
        buffer_s[tidx]=input[i-BLOCK_DIM]+input[i];
    }
    else if((i-BLOCK_DIM)<N){
        buffer_s[tidx]=input[i-BLOCK_DIM];
    }
    else buffer_s[tidx]=0;

    __syncthreads();

    for(unsigned int stride=BLOCK_DIM/2;stride>WARP_SIZE;stride/=2){
            if(tidx>BLOCK_DIM-stride)
            buffer_s[tidx]+=buffer_s[tidx-stride];
            
            __syncthreads();
    }
    if(tidx>=BLOCK_DIM-WARP_SIZE){
        float partialSum=buffer_s[tidx]+buffer_s[tidx-WARP_SIZE];
        for(unsigned int stride=WARP_SIZE/2;stride>0;stride/=2){
            partialSum += __shfl_up_sync(0xffffffff, partialSum, stride);
        }
        if(tidx==BLOCK_DIM-1)
        atomicAdd(sum,partialSum);
    }
}

float reduce_gpu(float* input, unsigned int N) {

    Timer timer;

    // Allocate memory
    startTime(&timer);
    float *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    float *sum_d;
    cudaMalloc((void**) &sum_d, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sum_d, 0, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    reduce_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, sum_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    float sum;
    cudaMemcpy(&sum, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(sum_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;

}


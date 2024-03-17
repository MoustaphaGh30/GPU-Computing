
#include "common.h"

#include "timer.h"
#define COARSE_FACTOR 32

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int histogram[NUM_BINS];

    int i=blockDim.x*blockIdx.x+threadIdx.x;

    if(threadIdx.x<256)
        histogram[threadIdx.x]=0u;
    
    __syncthreads();

    if(i<width*height){
        unsigned char b=image[i];
        atomicAdd(&histogram[b],1); 
    }
    __syncthreads();
        
    
    if(threadIdx.x<256&&histogram[threadIdx.x]>0)
        atomicAdd(&bins[threadIdx.x],histogram[threadIdx.x]);
    
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;

    histogram_private_kernel <<<numBlocks, numThreadsPerBlock>>> (image_d, bins_d, width, height);



}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int histogram[NUM_BINS];

    if(threadIdx.x<256)
    histogram[threadIdx.x]=0u;

    __syncthreads();

    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    for(unsigned int idx=i;idx<width*height;idx+=blockDim.x*gridDim.x){
        unsigned char b=image[idx];
        atomicAdd(&histogram[b],1); 
    }

    __syncthreads();  
    
    if(threadIdx.x<256&&histogram[threadIdx.x]>0)
        atomicAdd(&bins[threadIdx.x],histogram[threadIdx.x]);

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks=(((height*width)/COARSE_FACTOR)+numThreadsPerBlock-1)/numThreadsPerBlock;

    histogram_private_coarse_kernel <<<numBlocks, numThreadsPerBlock>>>(image_d,bins_d,width,height);

}


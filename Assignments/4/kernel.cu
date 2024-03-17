
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];
__device__ __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    // TODO
    int tY=threadIdx.y;
    int tX=threadIdx.x;

    int outRow=blockIdx.y*OUT_TILE_DIM+tY;
    int outCol=blockIdx.x*OUT_TILE_DIM+tX;

    int inRow=outRow-FILTER_RADIUS;
    int inCol=outCol-FILTER_RADIUS;

    if(inRow>=0&&inRow<height&&inCol>=0&&inCol<width)
        tile[tY][tX]=input[inRow*width+inCol];
    
    else 
        tile[tY][tX]=0.0f;

    __syncthreads();
    int tileRow=tY-FILTER_RADIUS;
    int tileCol=tX-FILTER_RADIUS;
    if(inRow>=0&&inRow<height&&inCol>=0&&inCol<width){
        if(tileCol>=0&&tileCol<OUT_TILE_DIM&&tileRow>=0&&tileRow<OUT_TILE_DIM){
            float sum=0.0f; 
            for(int i=0;i<FILTER_DIM;++i){    
                for(int j=0;j<FILTER_DIM;++j){
                    sum+=tile[tileRow+i][tileCol+j]*filter_c[i][j];
                }
            }
            output[inRow*width+inCol]=sum;
        }
        
    }
    
    

}

void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memory

    // TODO
    cudaMemcpyToSymbol(filter_c,filter,FILTER_DIM*FILTER_DIM*sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    // TODO
    dim3 numThreadsPerBlock(IN_TILE_DIM,IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);

    convolution_tiled_kernel <<<numBlocks,numThreadsPerBlock>>> (input_d,output_d,width,height);
}


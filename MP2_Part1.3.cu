//Shayan Rahman - 20282946

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()

#include <iostream>

#define TILE_WIDTH 16 // Assuming TILE_WIDTH is defined somewhere

//GPU Tiled Matrix Multiplication Implementation
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];

    float Pvalue = 0;

    int numTiles = (Width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int mRow = row;
        int mCol = t * TILE_WIDTH + threadIdx.x;
        int nRow = t * TILE_WIDTH + threadIdx.y;
        int nCol = col;

        if (mRow < Width && mCol < Width)
            M_tile[threadIdx.y][threadIdx.x] = M[mRow * Width + mCol];
        else
            M_tile[threadIdx.y][threadIdx.x] = 0.0;

        if (nRow < Width && nCol < Width)
            N_tile[threadIdx.y][threadIdx.x] = N[nRow * Width + nCol];
        else
            N_tile[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += M_tile[threadIdx.y][k] * N_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < Width && col < Width)
        P[row * Width + col] = Pvalue;
}

int main() {
    cudaFuncAttributes attr;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    std::cout << "\nDevice: " << props.name << "\n";

    // Question 1
    std::cout << "In Question 1:\n";
    std::cout << "Number of SMs: " << props.multiProcessorCount << "\n";
    std::cout << "Max threads per SM: " << props.maxThreadsPerMultiProcessor << "\n";
    int totalThreads = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
    std::cout << "Total # of threads scheduled concurrently: " << totalThreads << "\n";

    // Question 2
    std::cout << "Question 2:\n";
    cudaFuncGetAttributes(&attr, TiledMatrixMulKernel);
    std::cout << "Number of registers per thread: " << attr.numRegs << "\n";
    std::cout << "Shared memory size per block: " << attr.sharedSizeBytes << "\n";
    int maxBlocksPerSM = props.maxThreadsPerMultiProcessor / attr.maxThreadsPerBlock;
    std::cout << "Number of Blocks per SM: " << maxBlocksPerSM << "\n";
    std::cout << "Maximum threads per block: " << attr.maxThreadsPerBlock << "\n";

    return 0;
}

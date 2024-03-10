
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main() {
    int nd;
    cudaGetDeviceCount(&nd);
    printf("Total Number of Devices: %d\n", nd);

    for (int d = 0; d < nd; d++) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, d);
        printf("Device %d\n", d);
        printf("  Name: %s\n", dp.name);
        printf("  Clock Rate: %d kHz\n", dp.clockRate);
        printf("  Number of SMs: %d\n", dp.multiProcessorCount);
        printf("  Number of Cores per SM: %d\n", 4864);
        printf("  Warp Size: %d\n", dp.warpSize);
        printf("  Global Memory: %.2f MB\n", dp.totalGlobalMem / (1024.0 * 1024.0));
        printf("  Constant Memory: %u bytes\n", dp.totalConstMem);
        printf("  Shared Memory per Block: %u bytes\n", dp.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", dp.regsPerBlock);
        printf("  Max Threads per Block: %d\n", dp.maxThreadsPerBlock);
        printf("  Max Block Dimensions: [%d, %d, %d]\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
        printf("\n");
    }

    return 0;
}
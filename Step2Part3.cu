#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()
#include <algorithm> // For std::sort
#include <fstream>

#define NUM_SIZES 5
#define WIDTH {100, 250, 500, 1000, 1500} // Matrix sizes to experiment with
#define NUM_BLOCK_WIDTHS 5
#define BLOCK_WIDTHS {2, 5, 10, 25, 32} // Different block widths to experiment with
#define REPETITIONS 1 // Number of repetitions for timing measurements

cudaError_t multWithCuda(float* P, float* M, float* N, int Width, dim3 threadsPerBlock);

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of P element and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < Width && Col < Width)
    {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k)
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        P[Row * Width + Col] = Pvalue;
    }
}

// Helper function to calculate average execution time and remove outliers
float calculateAverageExecutionTime(float* execution_times, int num_repetitions) {
    std::sort(execution_times, execution_times + num_repetitions);
    float sum = 0;
    for (int i = 10; i < num_repetitions - 10; ++i) { // Exclude first 10 and last 10 measurements to remove outliers
        sum += execution_times[i];
    }
    return sum / (num_repetitions - 20);
}

int main()
{
    // Define matrix sizes and block widths to experiment with
    int sizes[NUM_SIZES] = WIDTH;
    int block_widths[NUM_BLOCK_WIDTHS] = BLOCK_WIDTHS;
    int num_sizes = NUM_SIZES;
    int num_block_widths = NUM_BLOCK_WIDTHS;

    // Open CSV file for writing results
    std::ofstream csv_file("matrix_multiplication_results.csv");
    if (!csv_file.is_open()) {
        printf("Error opening file.\n");
        return 1;
    }

    // Write CSV header
    csv_file << "Matrix Size,Block Width,GPU Execution Time (ms),NumBlocks/BlockWidth\n";

    // Repeat for each block width
    for (int b = 0; b < num_block_widths; ++b) {
        int block_width = block_widths[b];
        dim3 threadsPerBlock(block_width, block_width);

        // Repeat for each matrix size
        for (int i = 0; i < num_sizes; ++i)
        {
            int width = sizes[i];
            int size = width * width * sizeof(float);

            // Allocate memory for matrices M, N, P
            float* h_M, * h_N, * h_P;
            h_M = (float*)malloc(size);
            h_N = (float*)malloc(size);
            h_P = (float*)malloc(size);

            // Initialize matrices M, N with random values
            srand(time(NULL));
            for (int j = 0; j < width * width; ++j) {
                h_M[j] = static_cast<float>(rand()) / RAND_MAX;
                h_N[j] = static_cast<float>(rand()) / RAND_MAX;
            }

            // Allocate memory on the device
            float* d_M, * d_N, * d_P;
            cudaMalloc((void**)&d_M, size);
            cudaMalloc((void**)&d_N, size);
            cudaMalloc((void**)&d_P, size);

            // Create events for timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Record start event
            cudaEventRecord(start, 0);

            // Invoke kernel
            multWithCuda(d_P, d_M, d_N, width, threadsPerBlock);

            // Record stop event
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            // Calculate elapsed time
            float elapsed_gpu;
            cudaEventElapsedTime(&elapsed_gpu, start, stop);

            // Calculate number of blocks per block width
            int num_blocks_per_block_width = (width + block_width - 1) / block_width;

            // Print results to terminal
            printf("Block Width: %d,\t\t Matrix Size: %d,\t Kernel Execution Time: %.2f ms,\t NumBlocks/BlockWidth: %.2f\n",
                block_width, width, elapsed_gpu, static_cast<float>(num_blocks_per_block_width));

            // Write results to CSV file
            csv_file << sizes[i] << "," << block_width << "," << elapsed_gpu << "," << num_blocks_per_block_width << "\n";

            // Free device memory
            cudaFree(d_M);
            cudaFree(d_N);
            cudaFree(d_P);

            // Free host memory
            free(h_M);
            free(h_N);
            free(h_P);
        }
    }

    // Close CSV file
    csv_file.close();

    printf("Matrix multiplication results saved to matrix_multiplication_results.csv\n");

    return 0;
}

// Helper function for using CUDA to perform matrix multiplication in parallel.
cudaError_t multWithCuda(float* P, float* M, float* N, int Width, dim3 threadsPerBlock)
{
    // Invoke kernel
    dim3 numBlocks(Width / threadsPerBlock.x, Width / threadsPerBlock.y);
    MatrixMulKernel << <numBlocks, threadsPerBlock >> > (M, N, P, Width);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // Wait for GPU to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}

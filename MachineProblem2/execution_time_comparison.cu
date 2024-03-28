#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()

#define NUM_SIZES 5
#define TILE_WIDTH 10 // Adjust this as needed
#define REPETITIONS 10 // Number of repetitions for timing measurements

cudaError_t multWithCuda(float* P, float* M, float* N, int Width);

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

// CPU Matrix Multiply Implementation
void MatrixMulCPU(float* M, float* N, float* P, int Width)
{
    for (int i = 0; i < Width; ++i)
    {
        for (int j = 0; j < Width; ++j)
        {
            float sum = 0;
            for (int k = 0; k < Width; ++k)
            {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}

// Helper function to calculate average execution time
float calculateAverageExecutionTime(float* execution_times, int num_repetitions) {
    float sum = 0;
    for (int i = 0; i < num_repetitions; ++i) {
        sum += execution_times[i];
    }
    return sum / num_repetitions;
}

int main()
{
    // Define matrix sizes to experiment with
    int sizes[NUM_SIZES] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = NUM_SIZES;

    // Arrays to store execution times
    float execution_times_cpu[NUM_SIZES][REPETITIONS];
    float execution_times_gpu[NUM_SIZES][REPETITIONS];
    float execution_times_tiled_gpu[NUM_SIZES][REPETITIONS];

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

        // Allocate memory for device matrices M, N, P
        float* d_M, * d_N, * d_P;
        cudaMalloc((void**)&d_M, size);
        cudaMalloc((void**)&d_N, size);
        cudaMalloc((void**)&d_P, size);

        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Repeat for multiple measurements
        for (int k = 0; k < REPETITIONS; ++k) {
            // Transfer matrices M, N from host to device
            cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

            // Measure CPU matrix multiplication time
            clock_t cpu_start = clock();
            MatrixMulCPU(h_M, h_N, h_P, width);
            clock_t cpu_end = clock();
            execution_times_cpu[i][k] = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

            // Measure GPU matrix multiplication time
            cudaEventRecord(start, 0);
            MatrixMulKernel << <1, dim3(width, width) >> > (d_M, d_N, d_P, width);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float gpu_execution_time;
            cudaEventElapsedTime(&gpu_execution_time, start, stop);
            execution_times_gpu[i][k] = gpu_execution_time;

            // Measure GPU tiled matrix multiplication time
            cudaEventRecord(start, 0);
            int numBlocks = (width + TILE_WIDTH - 1) / TILE_WIDTH;
            TiledMatrixMulKernel << <dim3(numBlocks, numBlocks), dim3(TILE_WIDTH, TILE_WIDTH) >> > (d_M, d_N, d_P, width);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float tiled_gpu_execution_time;
            cudaEventElapsedTime(&tiled_gpu_execution_time, start, stop);
            execution_times_tiled_gpu[i][k] = tiled_gpu_execution_time;
        }

        // Free device memory
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);

        // Free host memory
        free(h_M);
        free(h_N);
        free(h_P);
    }

    // Write the results to a CSV file
    FILE* fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    // Write header
    fprintf(fp, "Matrix Size,CPU Avg Time (ms),GPU Avg Time (ms),Tiled GPU Avg Time (ms)\n");

    // Write data
    for (int i = 0; i < num_sizes; ++i) {
        for (int j = 0; j < REPETITIONS; ++j) {
            fprintf(fp, "%d,%.2f,%.2f,%.2f\n", sizes[i], execution_times_cpu[i][j], execution_times_gpu[i][j], execution_times_tiled_gpu[i][j]);
        }
    }

    fclose(fp);
    printf("Execution times saved to execution_times.csv\n");

    return 0;
}

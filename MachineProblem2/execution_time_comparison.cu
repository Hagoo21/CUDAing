#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()

#define NUM_SIZES 5
#define REPETITIONS 10 // Number of repetitions for timing measurements

// Define different TILE_WIDTH configurations
#define TILE_WIDTH_2 2
#define TILE_WIDTH_5 5
#define TILE_WIDTH_10 10
#define TILE_WIDTH_25 25

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

//GPU Tiled Matrix Multiplication Implementation
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int Width, int Tile_Width) {
    int row = blockIdx.y * Tile_Width + threadIdx.y;
    int col = blockIdx.x * Tile_Width + threadIdx.x;

    __shared__ float M_tile[TILE_WIDTH_25][TILE_WIDTH_25];
    __shared__ float N_tile[TILE_WIDTH_25][TILE_WIDTH_25];

    float Pvalue = 0;

    int numTiles = (Width + Tile_Width - 1) / Tile_Width;

    for (int t = 0; t < numTiles; ++t) {
        int mRow = row;
        int mCol = t * Tile_Width + threadIdx.x;
        int nRow = t * Tile_Width + threadIdx.y;
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

        for (int k = 0; k < Tile_Width; ++k)
            Pvalue += M_tile[threadIdx.y][k] * N_tile[k][threadIdx.x];

        __syncthreads();
    }

    if (row < Width && col < Width)
        P[row * Width + col] = Pvalue;
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
    float execution_times_tiled_gpu[NUM_SIZES][REPETITIONS];
    float execution_times_regular_gpu[NUM_SIZES][REPETITIONS];

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

            // Measure tiled GPU matrix multiplication time
            cudaEvent_t start_tiled, stop_tiled;
            cudaEventCreate(&start_tiled);
            cudaEventCreate(&stop_tiled);
            cudaEventRecord(start_tiled, 0);
            TiledMatrixMulKernel << <1, dim3(width, width) >> > (d_M, d_N, d_P, width, TILE_WIDTH_25);
            cudaEventRecord(stop_tiled, 0);
            cudaEventSynchronize(stop_tiled);
            float tiled_gpu_execution_time;
            cudaEventElapsedTime(&tiled_gpu_execution_time, start_tiled, stop_tiled);
            execution_times_tiled_gpu[i][k] = tiled_gpu_execution_time;

            // Measure regular GPU matrix multiplication time
            cudaEvent_t start_regular, stop_regular;
            cudaEventCreate(&start_regular);
            cudaEventCreate(&stop_regular);
            cudaEventRecord(start_regular, 0);
            cudaEventRecord(stop_regular, 0);
            cudaEventSynchronize(stop_regular);
            float regular_gpu_execution_time;
            cudaEventElapsedTime(&regular_gpu_execution_time, start_regular, stop_regular);
            execution_times_regular_gpu[i][k] = regular_gpu_execution_time;
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

    // Print results
    printf("Matrix Size\tCPU Time (ms)\tTiled GPU Time (ms)\tRegular GPU Time (ms)\n");
    for (int i = 0; i < num_sizes; ++i)
    {
        printf("%d\t\t", sizes[i]);
        // CPU execution time
        printf("%.2f\t\t", calculateAverageExecutionTime(execution_times_cpu[i], REPETITIONS));
        // Tiled GPU execution time
        printf("%.2f\t\t", calculateAverageExecutionTime(execution_times_tiled_gpu[i], REPETITIONS));
        // Regular GPU execution time
        printf("%.2f\n", calculateAverageExecutionTime(execution_times_regular_gpu[i], REPETITIONS));
    }

    return 0;
}


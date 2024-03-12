#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()

#define NUM_SIZES 5
#define WIDTH {100, 250, 500, 1000, 1500} // Matrix sizes to experiment with
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

        // Transfer matrices M, N from host to device
        cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Repeat for multiple measurements
        for (int k = 0; k < REPETITIONS; ++k) {
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

    FILE* fp = fopen("execution_times.csv", "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    // Write header
    fprintf(fp, "Matrix Size,CPU execution time(ms),GPU execution time(ms)\n");

    // Write data
    for (int i = 0; i < num_sizes; ++i) {
        for (int j = 0; j < REPETITIONS; ++j) {
            fprintf(fp, "%d,%.2f,%.2f\n", sizes[i], execution_times_cpu[i][j], execution_times_gpu[i][j]);
        }
    }

    fclose(fp);
    printf("Execution times saved to execution_times.csv\n");

    // Calculate average execution times
    float average_times_cpu[NUM_SIZES];
    float average_times_gpu[NUM_SIZES];
    for (int i = 0; i < num_sizes; ++i) {
        average_times_cpu[i] = calculateAverageExecutionTime(execution_times_cpu[i], REPETITIONS);
        average_times_gpu[i] = calculateAverageExecutionTime(execution_times_gpu[i], REPETITIONS);
    }

    // Print execution times
    printf("Matrix Size\tCPU Time (ms)\tGPU Time (ms)\n");
    for (int i = 0; i < num_sizes; ++i)
    {
        printf("%d\t\t%.2f\t\t%.2f\n", sizes[i], average_times_cpu[i], average_times_gpu[i]);
    }

    return 0;
}

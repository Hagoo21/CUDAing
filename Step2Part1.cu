#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()
#include <algorithm> // For std::sort

#define NUM_SIZES 5
#define WIDTH {100, 250, 500, 1000, 1500} // Matrix sizes to experiment with
#define REPETITIONS 100 // Number of repetitions for timing measurements


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

// Helper function to calculate average transfer time and remove outliers
float calculateAverageTransferTime(float* transfer_times, int num_repetitions) {
    std::sort(transfer_times, transfer_times + num_repetitions);
    float sum = 0;
    for (int i = 10; i < num_repetitions - 10; ++i) { // Exclude first 10 and last 10 measurements to remove outliers
        sum += transfer_times[i];
    }
    return sum / (num_repetitions - 20);
}

int main()
{
    // Define matrix sizes to experiment with
    int sizes[NUM_SIZES] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = NUM_SIZES;

    // Arrays to store transfer times
    float transfer_times_host_to_device[NUM_SIZES][REPETITIONS];
    float transfer_times_device_to_host[NUM_SIZES][REPETITIONS];

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

        // Repeat for multiple measurements
        for (int k = 0; k < REPETITIONS; ++k) {
            // Measure host-to-device transfer time
            cudaEventRecord(start, 0);
            cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&transfer_times_host_to_device[i][k], start, stop);
            
            // Perform matrix multiplication on the device
            multWithCuda(d_P, d_M, d_N, width);

            // Measure device-to-host transfer time
            cudaEventRecord(start, 0);
            cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&transfer_times_device_to_host[i][k], start, stop);

        }

        FILE* fp = fopen("transfer_times.csv", "w");
        if (fp == NULL) {
            printf("Error opening file.\n");
            return 1;
        }

        // Write header
        fprintf(fp, "Matrix Size,Host to Device (ms),Device to Host (ms)\n");

        // Write data
        for (int i = 0; i < num_sizes; ++i) {
            for (int j = 0; j < REPETITIONS; ++j) {
                fprintf(fp, "%d,%.2f,%.2f\n", sizes[i], transfer_times_host_to_device[i][j], transfer_times_device_to_host[i][j]);
            }
        }

        fclose(fp);
        printf("Transfer times saved to transfer_times.csv\n");


        // Free device memory
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);

        // Free host memory
        free(h_M);
        free(h_N);
        free(h_P);
    }

    // Calculate average transfer times and remove outliers
    float average_times_host_to_device[NUM_SIZES];
    float average_times_device_to_host[NUM_SIZES];
    for (int i = 0; i < num_sizes; ++i) {
        average_times_host_to_device[i] = calculateAverageTransferTime(transfer_times_host_to_device[i], REPETITIONS);
        average_times_device_to_host[i] = calculateAverageTransferTime(transfer_times_device_to_host[i], REPETITIONS);
    }

    // Print transfer times
    printf("Matrix Size\tHost to Device (ms)\tDevice to Host (ms)\n");
    for (int i = 0; i < num_sizes; ++i)
    {
        printf("%d\t\t%.2f\t\t\t%.2f\n", sizes[i], average_times_host_to_device[i], average_times_device_to_host[i]);
    }

    return 0;
}

// Helper function for using CUDA to perform matrix multiplication in parallel.
cudaError_t multWithCuda(float* P, float* M, float* N, int Width)
{
    // Invoke kernel
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks(Width / 16, Width / 16); // Assuming a block size of 16x16 threads
    MatrixMulKernel <<<numBlocks, threadsPerBlock >>> (M, N, P, Width);

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

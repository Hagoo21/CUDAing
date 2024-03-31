//Shayan Rahman - 20282946

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For time()

#define TILE_WIDTH 16
#define TOLERANCE 1e-6

//GPU Tiled Matrix Multiplication Implementation
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
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

// Function to generate random matrix
void generateRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Function to compare matrices
int compareMatrices(float* A, float* B, int size) {
    for (int i = 0; i < size * size; ++i) {
        if (fabs(A[i] - B[i]) > TOLERANCE)
            return 0; // Not equal
    }
    return 1; // Equal
}
// Function to print matrix
void printMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.4f\t", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL));

    int sizes[] = { 100, 250, 500, 1000, 1500 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        int size = sizes[i];
        int matrix_size = size * size;
        size_t bytes = matrix_size * sizeof(float);

        // Allocate memory for matrices M, N, P
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P_CPU = (float*)malloc(bytes);
        float* h_P_GPU = (float*)malloc(bytes);

        // Generate random matrices M and N
        generateRandomMatrix(h_M, size);
        generateRandomMatrix(h_N, size);

        // Allocate memory for device matrices M, N, P
        float* d_M, * d_N, * d_P;
        cudaMalloc((void**)&d_M, bytes);
        cudaMalloc((void**)&d_N, bytes);
        cudaMalloc((void**)&d_P, bytes);

        // Copy matrices M and N from host to device
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

        // Launch kernel for GPU matrix multiplication
        dim3 dimGrid((size + TILE_WIDTH - 1) / TILE_WIDTH, (size + TILE_WIDTH - 1) / TILE_WIDTH);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, size);
        cudaDeviceSynchronize();

        // Copy result matrix P from device to host
        cudaMemcpy(h_P_GPU, d_P, bytes, cudaMemcpyDeviceToHost);

        // Compute result matrix P using CPU
        MatrixMulCPU(h_M, h_N, h_P_CPU, size);

        // Print matrices
        /*printf("Matrix size: %dx%d\n", size, size);
        printf("CPU Result:\n");
        printMatrix(h_P_CPU, size);
        printf("\nGPU Result:\n");
        printMatrix(h_P_GPU, size);
        printf("\n");*/

        // Compare the results
        int passed = compareMatrices(h_P_CPU, h_P_GPU, size);
        if (passed) {
            printf("Test PASSED for size %dx%d\n", size, size);
        }
        else {
            printf("Test FAILED for size %dx%d\n", size, size);
        }

        // Free memory
        free(h_M);
        free(h_N);
        free(h_P_CPU);
        free(h_P_GPU);
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
    }

    return 0;
}
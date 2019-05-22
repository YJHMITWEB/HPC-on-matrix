#include "gpu_entrance.h"

__global__ void gmm_gpu(int matrixRowsA, const int matrixColsARowsB, int matrixColsB,
	const float* __restrict__ matrixA,
	const float* __restrict__ matrixB,
	float* __restrict__ matrixProduct)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < matrixColsB&&row < matrixRowsA) {
		float sum = 0.0f;
		for (int i = 0; i < matrixColsARowsB; i++) {
			sum += matrixA[row*matrixColsARowsB + i] * matrixB[i*matrixColsARowsB + col];
		}
		/*
		说明matrix在全局存储器中，地址按行存储，访存B矩阵时，在一次for循环内，多个线程
		同时访问B的一行，同时warp也是按照block横向划分，即一行是一个warp。
		*/
		matrixProduct[row*matrixColsARowsB + col] = sum;
	}
}

__global__ void gmm_gpu_ABT(int matrixRowsA, const int matrixColsARowsB, int matrixColsB, const float *__restrict__ matrixA, const float *__restrict__ matrixBTrans, float *__restrict__ matrixProduct)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < matrixColsB&&row < matrixRowsA) {
		float sum = 0.0f;
		for (int i = 0; i < matrixColsARowsB; i++) {
			sum += matrixA[row*matrixColsARowsB + i] * matrixBTrans[col*matrixColsARowsB + i];
		}
		matrixProduct[row*matrixColsARowsB + col] = sum;
	}
}

__global__ void gmm_gpu_ATB(int matrixRowsA, const int matrixColsARowsB, int matrixColsB, const float *__restrict__ matrixATrans, const float *__restrict__ matrixB, float *__restrict__ matrixProduct)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col < matrixColsB&&row < matrixRowsA) {
		float sum = 0.0f;
		for (int i = 0; i < matrixColsARowsB; i++) {
			sum += matrixATrans[i*matrixColsARowsB + row] * matrixB[i*matrixColsARowsB + col];
		}
		matrixProduct[row*matrixColsARowsB + col] = sum;
	}
}

__global__ void gmm_gpu_TILE(int matrixRowsA, const int matrixColsARowsB, int matrixColsB, const float *__restrict__ matrixA, const float *__restrict__ matrixB, float *__restrict__ matrixProduct)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0;
	for (int m = 0; m < matrixColsARowsB / TILE_WIDTH; m++) {
		int m1_tile_start = by * TILE_WIDTH * matrixColsARowsB + m * TILE_WIDTH;
		int m2_tile_start = m * TILE_WIDTH * matrixColsB + bx * TILE_WIDTH;

		__shared__ float m1_tile[TILE_WIDTH][TILE_WIDTH];
		__shared__ float m2_tile[TILE_WIDTH][TILE_WIDTH];

		m1_tile[ty][tx] = matrixA[m1_tile_start + ty * matrixColsARowsB + tx];
		m2_tile[ty][tx] = matrixB[m2_tile_start + ty * matrixColsB + tx];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++) {
			sum += m1_tile[ty][k] * m2_tile[k][tx];
		}
		__syncthreads();
	}
	matrixProduct[(by*blockDim.y + ty)*matrixColsARowsB + bx * blockDim.x + tx] = sum;
}

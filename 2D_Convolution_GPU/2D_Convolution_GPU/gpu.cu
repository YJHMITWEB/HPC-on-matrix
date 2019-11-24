#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
using namespace std;

#include "gpu.h"

bool _abs_diff(float a, float b, float tolerance)
{
	if (a > b) {
		if (a - b < tolerance) return false;
		else return true;
	}
	else {
		if (b - a < tolerance) return false;
		else return true;
	}
}

void validate(float* m1, float* m2, config &c)
{
	bool flag = false;
	for (int i = 0; i < c.R_HEIGHT * c.R_WIDTH; i++) {
		if (_abs_diff(m1[i], m2[i], c.tolarance)) {
			printf("validate failed...\n");
			flag = true;
			break;
		}
	}
	if (flag == false) printf("validate pass...\n");
}

typedef void(*cpu_method)(const float* m, const float* k, config &c, float* result);

float* Timing(cpu_method f, const float* m, const float* k, config &c, string method, bool show)
{
	float* result = (float*)malloc(sizeof(float) * c.R_HEIGHT * c.R_WIDTH);
	LARGE_INTEGER freq, freqcnt;
	double dcnt, cnt1, cnt2;
	QueryPerformanceFrequency(&freq);
	dcnt = freq.QuadPart;
	QueryPerformanceCounter(&freqcnt);
	cnt1 = freqcnt.QuadPart;

	f(m, k, c, result);

	QueryPerformanceCounter(&freqcnt);
	cnt2 = freqcnt.QuadPart;
	if (show) {
		cout << "result" << endl;
		for (int i = 0; i < c.R_HEIGHT; i++) {
			for (int j = 0; j < c.R_WIDTH; j++) {
				printf("%f ", result[i*c.R_WIDTH + j]);
			}
			printf("\n");
		}
	}
	float time = (cnt2 - cnt1) / dcnt * 1000;
	float Gflops = c.GFlo / (time / 1000);
	cout << "cpu " << method << " --- Time to compute: " << time << " ms, " << Gflops << " GFlops" << endl;
	return result;
}

void cpu_sequential(const float* matrix, const float* kernel, config &c, float* result)
{
	int R_SIZE = SQUARE - K_SIZE + 1;
	for (int y = 0; y < R_SIZE; y++) {
		for (int x = 0; x < R_SIZE; x++) {
			float sum = 0.0;
			for (int fy = 0; fy < K_SIZE; fy++) {
				for (int fx = 0; fx < K_SIZE; fx++) {
					float kernelItem = kernel[fx + fy * K_SIZE];
					float matrixItem = matrix[x + fx + (y + fy)*SQUARE];
					sum += kernelItem * matrixItem;
				}
			}
			result[x + y * R_SIZE] = sum;
		}
	}
}

/*********************************************************************/

__device__ __constant__ float c_kernel[25];

__global__ void naive_gpu(int result_X, int result_Y, int matrix_X, int kernel_Size,
	const float * matrix, const float * kernel, float * result)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (y < result_Y) {
		if (x < result_X) {
			float sum = 0.0;
#pragma unroll
			for (int fy = 0; fy < kernel_Size; fy++) {
#pragma unroll
				for (int fx = 0; fx < kernel_Size; fx++) {
					float kernelItem = kernel[fx + fy * kernel_Size];
					float matrixItem = matrix[x + fx + (y + fy) * matrix_X];
					sum += kernelItem * matrixItem;
				}
			}
			result[x + y * result_X] = sum;
		}
	}
}

__global__ void ckernel_gpu(int result_X, int result_Y, int matrix_X, int kernel_Size,
	const float * matrix, float * result)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (y < result_Y) {
		if (x < result_X) {
			float sum = 0.0;
#pragma unroll
			for (int fy = 0; fy < kernel_Size; fy++) {
#pragma unroll
				for (int fx = 0; fx < kernel_Size; fx++) {
					float kernelItem = c_kernel[fx + fy * kernel_Size];
					float matrixItem = matrix[x + fx + (y + fy) * matrix_X];
					sum += kernelItem * matrixItem;
				}
			}
			result[x + y * result_X] = sum;
		}
	}
}

template<int ksize, int tw>
__global__ void sharemem_gpu(int result_X, int result_Y, int matrix_X, int kernel_Size,
	const float * matrix, float * result)
{
	float __shared__ s_matrix[(tw + ksize - 1)*(tw + ksize - 1)];
	int M_tile_size = tw + ksize - 1;
	int tidy = threadIdx.y;
	int y = tidy + blockDim.y*blockIdx.y;
	int tidx = threadIdx.x;
	int x = tidx + blockDim.x*blockIdx.x;
	//center
	s_matrix[tidx + tidy * M_tile_size] = matrix[x + y * matrix_X];
	//right
	if (tidx < kernel_Size - 1) {
		s_matrix[tidx + tw + tidy * M_tile_size] = matrix[x + tw + y * matrix_X];
	}
	//bottom
	if (tidy < kernel_Size - 1) {
		s_matrix[tidx + (tidy + tw) * M_tile_size] = matrix[x + (y + tw) * matrix_X];
	}
	//rightbottom corner
	if (tidy < kernel_Size - 1 && tidx < kernel_Size - 1) {
		s_matrix[tidx + tw + (tidy + tw) * M_tile_size] =
			matrix[x + tw + (y + tw) * matrix_X];
	}
	__syncthreads();

	float sum = 0.0;
#pragma unroll
	for (int fy = 0; fy < kernel_Size; fy++) {
#pragma unroll
		for (int fx = 0; fx < kernel_Size; fx++) {
			float kernelItem = c_kernel[fx + fy * kernel_Size];
			float s_matrixItem = s_matrix[tidx + fx + (tidy + fy) * M_tile_size];
			sum += kernelItem * s_matrixItem;
		}
	}
	result[x + y * result_X] = sum;
}

/****************************************************************************/

float* gpu_naive(const float* matrix, const float* kernel, config &c)
{
	float* g_matrix;
	float* g_kernel;
	float* g_result;
	float* result = (float*)malloc(sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH);
	cudaMalloc((void**)&g_kernel, sizeof(float)*c.k_size*c.k_size);
	cudaMalloc((void**)&g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaMemcpy(g_matrix, matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_kernel, kernel, sizeof(float)*c.k_size*c.k_size, cudaMemcpyHostToDevice);
	
	dim3 grid((c.R_WIDTH - 1) / TILE_WIDTH + 1, (c.R_WIDTH - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	naive_gpu << <grid, block >> > (c.R_WIDTH, c.R_HEIGHT, c.M_WIDTH, c.k_size,
		g_matrix, g_kernel, g_result);

	cudaMemcpy(result, g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	float Gflops = c.GFlo / (elapsedTime / 1000);
	printf("GPU naive --- Time to compute:  %3.3f ms, %3.5f GFLOPs\n", elapsedTime, Gflops);

	cudaFree(g_matrix);
	cudaFree(g_kernel);
	cudaFree(g_result);
	return result;
}

float* gpu_kernel_constant(const float* matrix, const float* kernel, config &c)
{
	float* g_matrix;
	float* g_result;
	float* result = (float*)malloc(sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH);
	cudaMalloc((void**)&g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaMemcpy(g_matrix, matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_kernel, kernel, sizeof(float)*c.k_size*c.k_size, 0, cudaMemcpyHostToDevice);

	dim3 grid((c.R_WIDTH - 1) / TILE_WIDTH + 1, (c.R_WIDTH - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	ckernel_gpu << <grid, block >> > (c.R_WIDTH, c.R_HEIGHT, c.M_WIDTH, c.k_size,
		g_matrix, g_result);

	cudaMemcpy(result, g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	float Gflops = c.GFlo / (elapsedTime / 1000);
	printf("GPU ckernel --- Time to compute:  %3.3f ms, %3.5f GFLOPs\n", elapsedTime, Gflops);

	cudaFree(g_matrix);
	cudaFree(c_kernel);
	cudaFree(g_result);
	return result;
}

float* gpu_kernel_constant_sharedmem(const float* matrix, const float* kernel, config &c)
{
	float* g_matrix;
	float* g_result;
	float* result = (float*)malloc(sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH);
	cudaMalloc((void**)&g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH);

	cudaMemcpy(g_matrix, matrix, sizeof(float)*c.M_HEIGHT*c.M_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_kernel, kernel, sizeof(float)*c.k_size*c.k_size, 0, cudaMemcpyHostToDevice);

	dim3 grid((c.R_WIDTH - 1) / TILE_WIDTH + 1, (c.R_WIDTH - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	sharemem_gpu<K_SIZE, TILE_WIDTH> << <grid, block >> > (c.R_WIDTH, c.R_HEIGHT, c.M_WIDTH, c.k_size,
		g_matrix, g_result);

	cudaMemcpy(result, g_result, sizeof(float)*c.R_HEIGHT*c.R_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	float Gflops = c.GFlo / (elapsedTime / 1000);
	printf("GPU ckernel_sharedmem --- Time to compute:  %3.3f ms, %3.5f GFLOPs\n", elapsedTime, Gflops);

	cudaFree(g_matrix);
	cudaFree(c_kernel);
	cudaFree(g_result);
	return result;
}

float * run_gpu(float * matrix, float * kernel, config & c)
{
	float* cpu_sequential_result;
	cpu_sequential_result = Timing(cpu_sequential, matrix, kernel, c, "sequential", c.cpu_sequential_show);

	float* gpu_naive_result;
	gpu_naive_result = gpu_naive(matrix, kernel, c);
	validate(cpu_sequential_result, gpu_naive_result, c);

	float* gpu_ckernel_result;
	gpu_ckernel_result = gpu_kernel_constant(matrix, kernel, c);
	validate(cpu_sequential_result, gpu_ckernel_result, c);

	float* gpu_ckernel_sharedmem_result;
	gpu_ckernel_sharedmem_result = gpu_kernel_constant_sharedmem(matrix, kernel, c);
	validate(cpu_sequential_result, gpu_ckernel_sharedmem_result, c);


	return nullptr;
}

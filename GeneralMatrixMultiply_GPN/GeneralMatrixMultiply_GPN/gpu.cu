#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>

#include "config.h"
#include "gpu_entrance.h"

inline bool _abs_diff(float a, float b, float tolerance)
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
	for (int i = 0; i < c.M1_HEIGHT * c.M2_WIDTH; i++) {
		if (_abs_diff(m1[i], m2[i], c.tolarance)) {
			printf("validate failed...\n");
			flag = true;
			break;
		}
	}
	if (flag == false) printf("validate pass...\n");
}

float* transpose_m(float* m, int h, int w, bool show)
{
	float* result = (float*)malloc(sizeof(float)*h*w);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int idx = i * w + j;
			int t_idx = j * h + i;
			result[t_idx] = m[idx];
		}
	}
	
	return result;
}

float4* transpose_m(float4* m, int h, int w, bool show)
{
	float4* result = (float4*)malloc(sizeof(float4)*h*w/4);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int idx = i * w + j;
			int t_idx = j * h + i;
			result[t_idx] = m[idx];
		}
	}

	return result;
}

float* gpu_naive_v4(float* m1, float* m2, config& c)
{
	float4* m1_v4 = (float4 *)m1;
	float4* m2_v4 = (float4 *)m2;
	float4* r_v4 = (float4*)malloc(sizeof(float4)*c.M1_HEIGHT*c.M2_WIDTH / 4);
	return nullptr;
}


float* start_gmm_gpu(float* m1, float* m2, config& c)
{
	/*
		AB^T
	*/
	//float* t_m2 = transpose_m(m2, c.M1_WIDTH, c.M2_WIDTH, true);
	
	float* g_m1;
	float* g_m2;
	float* g_r;
	float* r = (float*)calloc(c.M1_HEIGHT*c.M2_WIDTH, sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH);
	cudaMalloc((void**)&g_m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH);
	cudaMalloc((void**)&g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH);

	cudaMemcpy(g_m1, m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_m2, m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_r, r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyHostToDevice);

	dim3 grid((c.M2_WIDTH - 1) / TILE_WIDTH + 1, (c.M1_HEIGHT - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	gmm_gpu << <grid, block >> > (c.M1_HEIGHT, c.M1_WIDTH, c.M2_WIDTH, g_m1, g_m2, g_r);

	cudaMemcpy(r, g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU naive --- Time to compute:  %3.8f ms\n", elapsedTime);

	cudaFree(g_m1);
	cudaFree(g_m2);
	cudaFree(g_r);
	return r;
}

float* start_gmm_gpu_ABT(float* m1, float* m2, config& c)
{
	/*
	AB^T
	*/
	//float* t_m2 = transpose_m(m2, c.M1_WIDTH, c.M2_WIDTH, true);

	float* g_m1;
	float* g_m2;
	float* g_r;
	float* m2_T = transpose_m(m2, c.M2_WIDTH, c.M2_WIDTH, false);
	float* r = (float*)calloc(c.M1_HEIGHT*c.M2_WIDTH, sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH);
	cudaMalloc((void**)&g_m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH);
	cudaMalloc((void**)&g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH);

	cudaMemcpy(g_m1, m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_m2, m2_T, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_r, r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyHostToDevice);

	dim3 grid((c.M2_WIDTH - 1) / TILE_WIDTH + 1, (c.M1_HEIGHT - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	gmm_gpu_ABT << <grid, block >> > (c.M1_HEIGHT, c.M1_WIDTH, c.M2_WIDTH, g_m1, g_m2, g_r);

	cudaMemcpy(r, g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU ABT --- Time to compute:  %3.8f ms\n", elapsedTime);

	cudaFree(g_m1);
	cudaFree(g_m2);
	cudaFree(g_r);
	return r;
}

float* start_gmm_gpu_ATB(float* m1, float* m2, config& c)
{
	/*
	AB^T
	*/
	//float* t_m2 = transpose_m(m2, c.M1_WIDTH, c.M2_WIDTH, true);

	float* g_m1;
	float* g_m2;
	float* g_r;
	float* m1_T = transpose_m(m1, c.M1_WIDTH, c.M1_WIDTH, false);
	float* r = (float*)calloc(c.M1_HEIGHT*c.M2_WIDTH, sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH);
	cudaMalloc((void**)&g_m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH);
	cudaMalloc((void**)&g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH);

	cudaMemcpy(g_m1, m1_T, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_m2, m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_r, r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyHostToDevice);

	dim3 grid((c.M2_WIDTH - 1) / TILE_WIDTH + 1, (c.M1_HEIGHT - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	gmm_gpu_ATB << <grid, block >> > (c.M1_HEIGHT, c.M1_WIDTH, c.M2_WIDTH, g_m1, g_m2, g_r);

	cudaMemcpy(r, g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU ATB --- Time to compute:  %3.8f ms\n", elapsedTime);

	cudaFree(g_m1);
	cudaFree(g_m2);
	cudaFree(g_r);
	return r;
}

float* start_gmm_gpu_TILE(float* m1, float* m2, config& c)
{
	/*
	AB^T
	*/
	//float* t_m2 = transpose_m(m2, c.M1_WIDTH, c.M2_WIDTH, true);

	float* g_m1;
	float* g_m2;
	float* g_r;
	float* r = (float*)calloc(c.M1_HEIGHT*c.M2_WIDTH, sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&g_m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH);
	cudaMalloc((void**)&g_m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH);
	cudaMalloc((void**)&g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH);

	cudaMemcpy(g_m1, m1, sizeof(float)*c.M1_HEIGHT*c.M1_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_m2, m2, sizeof(float)*c.M1_WIDTH*c.M2_WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(g_r, r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyHostToDevice);

	dim3 grid((c.M2_WIDTH - 1) / TILE_WIDTH + 1, (c.M1_HEIGHT - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	gmm_gpu_TILE << <grid, block >> > (c.M1_HEIGHT, c.M1_WIDTH, c.M2_WIDTH, g_m1, g_m2, g_r);

	cudaMemcpy(r, g_r, sizeof(float)*c.M1_HEIGHT*c.M2_WIDTH, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU TILE --- Time to compute:  %3.8f ms\n", elapsedTime);

	cudaFree(g_m1);
	cudaFree(g_m2);
	cudaFree(g_r);
	return r;
}


float * run_gpu(float * m1, float * m2, config & c, float* cpu_result)
{
	float* res_gmm_gpu = start_gmm_gpu(m1, m2, c);
	validate(cpu_result, res_gmm_gpu, c);

	float* res_gmm_gpu_ABT = start_gmm_gpu_ABT(m1, m2, c);
	validate(cpu_result, res_gmm_gpu_ABT, c);

	float* res_gmm_gpu_ATB = start_gmm_gpu_ATB(m1, m2, c);
	validate(cpu_result, res_gmm_gpu_ATB, c);

	float* res_gmm_gpu_TILE = start_gmm_gpu_TILE(m1, m2, c);
	validate(cpu_result, res_gmm_gpu_TILE, c);

	return res_gmm_gpu;
}

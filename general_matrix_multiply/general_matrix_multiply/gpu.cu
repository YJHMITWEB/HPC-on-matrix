#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include <iostream>
#include <string>
using namespace std;

#include "gpu.h"
#include "config.h"

__global__ void gpu_common_kernel(int M_, int K_, int N_, const float4* __restrict__ A, const float* __restrict__ B, float4* __restrict__ result)
{

}

float* gpu_common(const float* m1, const float* m2, config &c)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	float* result = (float*)malloc(sizeof(float) * c.M1_HEIGHT * c.M2_WIDTH);
	//gpu_common_kernel << < 4, 4 >> > (M, K, N, m1, m2, result);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("gpu common --- Time to compute:  %3.8f ms\n", elapsedTime);

	return result;
}

void run_gpu(float * m1, float * m2, config & c)
{
	float* gpu_common_result = gpu_common(m1, m2, c);
}

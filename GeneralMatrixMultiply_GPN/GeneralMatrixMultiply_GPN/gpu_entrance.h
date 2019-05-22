#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>

#include "config.h"

__global__ void gmm_gpu(int matrixRowsA, const int matrixColsARowsB, int matrixColsB,
	const float* __restrict__ matrixA,
	const float* __restrict__ matrixB,
	float* __restrict__ matrixProduct);

__global__ void gmm_gpu_ABT(int matrixRowsA, const int matrixColsARowsB, int matrixColsB,
	const float* __restrict__ matrixA,
	const float* __restrict__ matrixBTrans,
	float* __restrict__ matrixProduct);

__global__ void gmm_gpu_ATB(int matrixRowsA, const int matrixColsARowsB, int matrixColsB,
	const float* __restrict__ matrixATrans,
	const float* __restrict__ matrixB,
	float* __restrict__ matrixProduct);

__global__ void gmm_gpu_TILE(int matrixRowsA, const int matrixColsARowsB, int matrixColsB,
	const float* __restrict__ matrixA,
	const float* __restrict__ matrixB,
	float* __restrict__ matrixProduct);

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>

#include "gpu.h"

template<int blockDimx, int blockDimy, int r>
__global__ void XY_Z(const float* cube, float* ret, int X, int Y, int Z, float WT) 
{
	float __shared__ plate[blockDimy + 2 * r][blockDimx + 2 * r];

	float local[r * 2];

	int lidx = threadIdx.x;
	int lidy = threadIdx.y;
	int gidx = blockDim.x*blockIdx.x + lidx + r;
	int gidy = blockDim.y*blockIdx.y + lidy + r;

	int posx = lidx + r;
	int posy = lidy + r;

	for (int load_l = 0; load_l < r; load_l++) {
		local[load_l] = cube[(r * X * Y + gidy * Y + gidx) - X * Y * (r - load_l)];
	}
	for (int load_l = r; load_l < 2 * r; load_l++) {
		local[load_l] = cube[(r * X * Y + gidy * Y + gidx) + X * Y * (load_l - r + 1)];
	}

	for (int z = r; z < Z-r; z++) {
		float l_ret = 0;
		int z_offset = z * X * Y;
		int global_plate_center = z_offset + gidy* Y + gidx;
		
		//load mid square into shared mem
		plate[posy][posx] = cube[global_plate_center];
		
		//load left
		if (lidx == 0) {
			for (int left = 0; left < r; left++) {
				plate[posy][left] = cube[global_plate_center - (r - left)];
			}
		}
		//load right
		if (lidx == blockDim.x - 1) {
			for (int right = 0; right < r; right++) {
				plate[posy][r + blockDim.x + right] = cube[global_plate_center + right + 1];
			}
		}
		//load top
		if (lidy == 0) {
			for (int top = 0; top < r; top++) {
				plate[top][posx] = cube[global_plate_center - (r - top)*Y];
			}
		}
		//load bottom
		if (lidy == blockDim.y - 1) {
			for (int bottom = 0; bottom < r; bottom++) {
				plate[r + blockDim.y + bottom][posx] = cube[global_plate_center + (bottom + 1)*Y];
			}
		}

		__syncthreads();
		//compute
		l_ret += plate[posy][posx];
		for (int r_ = 1; r_ < r + 1; r_++) {
			l_ret += plate[posy - r_][posx] + plate[posy + r_][posx] +
				plate[posy][posx - r_] + plate[posy][posx + r_];
		}
		for (int add_l = 0; add_l < 2 * r; add_l++) {
			l_ret += local[add_l];
		}

		ret[global_plate_center] = l_ret * WT;
		__syncthreads();

		for (int shift_l = 1; shift_l < r; shift_l++) {
			local[shift_l - 1] = local[shift_l];
		}
		local[r - 1] = plate[posy][posx];
		for (int shift_l = r+1; shift_l < 2 * r; shift_l++) {
			local[shift_l - 1] = local[shift_l];
		}
		local[r * 2 - 1] = cube[global_plate_center + (r + 1)*X * Y];

		__syncthreads();
	}
}

void gpu_XY_Z(const float* cube, float* ret, config& c)
{
	float* gpu_cube;
	float* gpu_ret;
	
	cudaMalloc((void**)&gpu_cube, sizeof(float)*c.X*c.Y*c.Z);
	cudaMalloc((void**)&gpu_ret, sizeof(float)*c.X*c.Y*c.Z);

	cudaMemcpy(gpu_cube, cube, sizeof(float)*c.X*c.Y*c.Z, cudaMemcpyHostToDevice);

	dim3 blockSize(BS, BS);
	dim3 gridSize((WORK_ZONE + BS - 1) / BS, (WORK_ZONE + BS - 1) / BS);

	cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

	XY_Z<BS, BS, RADIUS> << <gridSize, blockSize >> > (gpu_cube, gpu_ret, c.X, c.Y, c.Z, c.WT);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU --- Time to compute:  %3.8f ms\n", elapsedTime);

	cudaMemcpy(ret, gpu_ret, sizeof(float)*c.X*c.Y*c.Z, cudaMemcpyDeviceToHost);
	cudaFree(gpu_cube);
	cudaFree(gpu_ret);
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
}

void run_gpu(float* cube)
{
	config c;
	float* ret = (float*)malloc(sizeof(float)*c.X*c.Y*c.Z);

	gpu_XY_Z(cube, ret, c);

	if (false) {
		printf("GPU result...\n");
		for (int z = RADIUS; z < c.Z - RADIUS; z++) {
			for (int y = RADIUS; y < c.Y - RADIUS; y++) {
				for (int x = RADIUS; x < c.X - RADIUS; x++) {
					printf("%5.4f ", ret[z*c.Y*c.X + y * c.X + x]);
				}
				printf("\n");
			}
			printf("\n\n\n\n");
		}
	}
	free(ret);
}

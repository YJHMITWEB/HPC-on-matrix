#pragma once

#define M 1024
#define K 1024
#define N 25
#define GEMM_SAMPLE_TILE_SIZE 32
#define GEMM_DIMENSION_TILE_SIZE 128
#define GEMM_IN_PARALLEL 1
#define NORM_SAMPLE_TILE_SIZE 64
#define NORM_DIMENSION_TILE_SIZE 128
#define NORM_IN_PARALLEL 1
#define FUSE_T1_SIZE 128
#define FUSE_T2_SIZE 32
#define FUSE_IN_PARALLEL 1

typedef struct CONFIG {
	int M1_samples = M;
	int M2_samples = N;
	int dimensions = K;
	int Sample_tile_size = GEMM_SAMPLE_TILE_SIZE;
	int Dimension_tile_size = GEMM_DIMENSION_TILE_SIZE;

	bool data_show = false;

	bool cpu_ged_naive_show = false;
	bool cpu_ged_avx_show = false;

	float tolarance = 0.0001f;
}config;

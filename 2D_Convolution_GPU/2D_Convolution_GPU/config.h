#pragma once

#define SQUARE 4100
#define CHUNK 64
#define M CHUNK
#define N CHUNK
#define IN_PARALLEL 2
#define K_SIZE 5
#define CACHELINE_RATE 8
#define LOOP_UNROLL_RATE 5
#define TILE_WIDTH 32

typedef struct CONFIG {
	int M_WIDTH = SQUARE;
	int M_HEIGHT = SQUARE;
	int chunk = CHUNK;
	int k_size = K_SIZE;
	int testTime = 1;

	int R_WIDTH = SQUARE - K_SIZE + 1;
	int R_HEIGHT = SQUARE - K_SIZE + 1;
	int loop_unroll_rate = LOOP_UNROLL_RATE;
	float GFlo = R_WIDTH * R_HEIGHT * (K_SIZE * K_SIZE * 2.0) / 1000 / 1000 / 1000;

	float tolarance = 0.0001f;

	bool data_show = false;
	bool cpu_sequential_show = false;

	bool run_cpu_sequential_cacheLine = true;
	bool show_cpu_sequential_cacheLine = false;

	bool run_cpu_sequential_loop_unroll = true;
	bool show_cpu_sequential_loop_unroll = false;

	bool run_cpu_avx = true;
	bool show_cpu_avx = false;

	bool run_cpu_avx_chunk = true;
	bool show_cpu_avx_chunk = false;

	bool run_cpu_avx_omp = true;
	bool show_cpu_avx_omp = false;
}config;
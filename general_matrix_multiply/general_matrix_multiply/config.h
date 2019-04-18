#pragma once

#define SQUARE 1024
#define CHUNK 64
#define M CHUNK
#define K CHUNK
#define N CHUNK
#define IN_PARALLEL 2

typedef struct CONFIG {
	int M1_WIDTH = SQUARE;
	int M1_HEIGHT = SQUARE;
	int M2_WIDTH = SQUARE;
	int chunk = CHUNK;

	bool data_show = false;

	bool cpu_serial_show = false;
	bool cpu_chunk_show = false;
	bool cpu_serial_omp_show = false;
	bool cpu_chunk_omp_show = false;
	bool cpu_serial_chunk_avx_show = false;
	bool cpu_omp_chunk_avx_show = false;
	bool gpu_common_show = true;

	bool run_cpu_chunk = true;
	bool run_cpu_serial_omp = true;
	bool run_cpu_chunk_omp = true;
	bool run_cpu_serial_chunk_avx = true;
	bool run_cpu_omp_chunk_avx = true;
	bool run_gpu_common = true;

	float tolarance = 0.01f;
}config;


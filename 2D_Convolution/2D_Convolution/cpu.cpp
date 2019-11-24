#include <iostream>
#include <string>
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
using namespace std;

#include "cpu.h"

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

void cpu_avx(const float* matrix, const float* kernel, config &c, float* result)
{
	int R_SIZE = SQUARE - K_SIZE + 1;
	for (int y = 0; y < R_SIZE; y++) {
		for (int x = 0; x < R_SIZE; x += 8) {
			//float sum = 0.0;
			__m256 sum = _mm256_setzero_ps();
			for (int fy = 0; fy < K_SIZE; fy++) {
				for (int fx = 0; fx < K_SIZE; fx++) {
					__m256 kernelItem = _mm256_broadcast_ss(
						kernel + fx + fy * K_SIZE);
					__m256 matrixItem =
						_mm256_loadu_ps(
							matrix + x + fx + (y + fy)*SQUARE);
					sum = _mm256_fmadd_ps(
						kernelItem, matrixItem, sum);
				}
			}
			_mm256_storeu_ps(result + x + y * R_SIZE,
				sum);
		}
	}
}
//
//void cpu_sequential_cacheLine(const float* matrix, const float* kernel, config &c, float* result)
//{
//	int R_SIZE = SQUARE - K_SIZE + 1;
//	for (int y = 0; y < R_SIZE; y+= CACHELINE_RATE) {
//		for (int x = 0; x < R_SIZE; x+= CACHELINE_RATE) {
//			float sum[CACHELINE_RATE * CACHELINE_RATE] = { 0.0 };
//			for (int fy = 0; fy < K_SIZE; fy++) {
//				for (int fx = 0; fx < K_SIZE; fx++) {
//					float kernelItem = kernel[fx + fy * K_SIZE];
//					for (int j = 0; j < CACHELINE_RATE; j++) {
//						for (int i = 0; i < CACHELINE_RATE; i++) {
//							float matrixItem = matrix[x + fx + i + (y + fy + j)*SQUARE];
//							sum[j*CACHELINE_RATE + i] += kernelItem * matrixItem;
//						}
//					}
//				}
//			}
//			for (int j = 0; j < CACHELINE_RATE; j++) {
//				for (int i = 0; i < CACHELINE_RATE; i++) {
//					result[x + i + (y + j) * R_SIZE] = sum[i + j * CACHELINE_RATE];
//				}
//			}
//		}
//	}
//}

void cpu_sequential_loop_unroll(const float* matrix, const float* kernel, config &c, float* result)
{
	int R_SIZE = SQUARE - K_SIZE + 1;
	for (int y = 0; y < R_SIZE; y++) {
		for (int x = 0; x < R_SIZE; x++) {
			float sum = 0.0;
#
			for (int fy = 0; fy < K_SIZE; fy++) {
				float kernelItem1 = kernel[0 + fy * K_SIZE];
				float matrixItem1 = matrix[x + 0 + (y + fy)*SQUARE];
				sum += kernelItem1 * matrixItem1;

				float kernelItem2 = kernel[1 + fy * K_SIZE];
				float matrixItem2 = matrix[x + 1 + (y + fy)*SQUARE];
				sum += kernelItem2 * matrixItem2;

				float kernelItem3 = kernel[2 + fy * K_SIZE];
				float matrixItem3 = matrix[x + 2 + (y + fy)*SQUARE];
				sum += kernelItem3 * matrixItem3;

				float kernelItem4 = kernel[3 + fy * K_SIZE];
				float matrixItem4 = matrix[x + 3 + (y + fy)*SQUARE];
				sum += kernelItem4 * matrixItem4;

				float kernelItem5 = kernel[4 + fy * K_SIZE];
				float matrixItem5 = matrix[x + 4 + (y + fy)*SQUARE];
				sum += kernelItem5 * matrixItem5;
				
			}
			result[x + y * R_SIZE] = sum;
		}
	}
}

void cpu_avx_chunk(const float* matrix, const float* kernel, config &c, float* result)
{
	int R_SIZE = SQUARE - K_SIZE + 1;
	for (int y = 0; y < R_SIZE; y+= CACHELINE_RATE) {
		for (int x = 0; x < R_SIZE; x+= CACHELINE_RATE*8) {
			__m256 sum[CACHELINE_RATE * CACHELINE_RATE] = { _mm256_setzero_ps() };
			for (int fy = 0; fy < K_SIZE; fy++) {
				for (int fx = 0; fx < K_SIZE; fx++) {
					__m256 kernelItem = _mm256_broadcast_ss(
						kernel+fx + fy * K_SIZE);
					for (int j = 0; j < CACHELINE_RATE; j++) {
						for (int i = 0; i < CACHELINE_RATE; i++) {
							__m256 matrixItem = 
								_mm256_loadu_ps(
									matrix + x + fx + i*8 + (y + fy + j)*SQUARE);
							sum[j*CACHELINE_RATE + i] = _mm256_fmadd_ps(
								kernelItem, matrixItem, sum[j*CACHELINE_RATE + i]);
						}
					}
				}
			}
			for (int j = 0; j < CACHELINE_RATE; j++) {
				for (int i = 0; i < CACHELINE_RATE; i++) {
					_mm256_storeu_ps(result + x + i*8 + (y + j) * R_SIZE,
						sum[i + j * CACHELINE_RATE]);
				}
			}
		}
	}
}

void cpu_avx_omp(const float* matrix, const float* kernel, config &c, float* result)
{
	int R_SIZE = SQUARE - K_SIZE + 1;
#pragma omp parallel for 
	for (int y = 0; y < R_SIZE; y++) {
		for (int x = 0; x < R_SIZE; x += 8) {
			//float sum = 0.0;
			__m256 sum = _mm256_setzero_ps();
			for (int fy = 0; fy < K_SIZE; fy++) {
				for (int fx = 0; fx < K_SIZE; fx++) {
					__m256 kernelItem = _mm256_broadcast_ss(
						kernel + fx + fy * K_SIZE);
					__m256 matrixItem =
						_mm256_loadu_ps(
							matrix + x + fx + (y + fy)*SQUARE);
					sum = _mm256_fmadd_ps(
						kernelItem, matrixItem, sum);
				}
			}
			_mm256_storeu_ps(result + x + y * R_SIZE,
				sum);
		}
	}
}

float * run_cpu(float * matrix, float * kernel, config & c)
{
	float* cpu_sequential_result = NULL;
	for (int i = 0; i < c.testTime; i++) {
		cpu_sequential_result = Timing(cpu_sequential, matrix, kernel, c, "sequential", c.cpu_sequential_show);
		if (i != c.testTime - 1) 
			free(cpu_sequential_result);
	}

	/*if (c.run_cpu_sequential_cacheLine) {
		for (int i = 0; i < c.testTime; i++) {
			float* cpu_sequential_cacheLine_result = Timing(cpu_sequential_cacheLine, matrix, kernel, c, "sequential_cacheLine", false);
			validate(cpu_sequential_result, cpu_sequential_cacheLine_result, c);
			free(cpu_sequential_cacheLine_result);
		}
	}*/

	if (c.run_cpu_sequential_loop_unroll) {
		for (int i = 0; i < c.testTime; i++) {
			float* cpu_sequential_loop_unroll_result = Timing(cpu_sequential_loop_unroll, matrix, kernel, c, "sequential_loop_unroll", false);
			validate(cpu_sequential_result, cpu_sequential_loop_unroll_result, c);
			free(cpu_sequential_loop_unroll_result);
		}
	}

	if (c.run_cpu_avx) {
		for (int i = 0; i < c.testTime; i++) {
			float* cpu_avx_result = Timing(cpu_avx, matrix, kernel, c, "avx", c.show_cpu_avx);
			validate(cpu_sequential_result, cpu_avx_result, c);
			free(cpu_avx_result);
		}
	}

	if (c.run_cpu_avx_chunk) {
		for (int i = 0; i < c.testTime; i++) {
			float* cpu_avx_chunk_result = Timing(cpu_avx_chunk, matrix, kernel, c, "avx_chunk", c.show_cpu_avx_chunk);
			validate(cpu_sequential_result, cpu_avx_chunk_result, c);
			free(cpu_avx_chunk_result);
		}
	}

	if (c.run_cpu_avx_omp) {
		for (int i = 0; i < c.testTime; i++) {
			float* cpu_avx_omp_result = Timing(cpu_avx_omp, matrix, kernel, c, "avx_omp", c.show_cpu_avx_omp);
			validate(cpu_sequential_result, cpu_avx_omp_result, c);
			free(cpu_avx_omp_result);
		}
	}


	free(cpu_sequential_result);
	return cpu_sequential_result;
}

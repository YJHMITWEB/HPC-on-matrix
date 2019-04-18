#include <iostream>
#include <string>
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <immintrin.h>
using namespace std;

#include "cpu.h"
#include "config.h"


typedef void(*cpu_method)(const float* m1, const float* m2, config &c, float* result);

float* Timing(cpu_method f, const float* m1, const float* m2, config &c, string method, bool show)
{
	LARGE_INTEGER freq, freqcnt;
	double dcnt, cnt1, cnt2;
	QueryPerformanceFrequency(&freq);
	dcnt = freq.QuadPart;
	QueryPerformanceCounter(&freqcnt);
	cnt1 = freqcnt.QuadPart;

	float* result = (float*)malloc(sizeof(float) * c.M1_HEIGHT * c.M2_WIDTH);
	f(m1, m2, c, result);

	QueryPerformanceCounter(&freqcnt);
	cnt2 = freqcnt.QuadPart;
	if (show) {
		cout << "result" << endl;
		for (int i = 0; i < c.M1_HEIGHT; i++) {
			for (int j = 0; j < c.M2_WIDTH; j++) {
				printf("%f ", result[i*c.M2_WIDTH + j]);
			}
			printf("\n");
		}
	}
	cout << "cpu " << method << " --- Time to compute: " << (cnt2 - cnt1) / dcnt * 1000 << " ms" << endl;
	return result;
}


void cpu_serial(const float* m1, const float* m2, config &c, float* result)
{
	for (int i = 0; i < c.M1_HEIGHT; i++) {
		for (int j = 0; j < c.M2_WIDTH; j++) {
			float sum = 0.0f;
			for (int z = 0; z < c.M1_WIDTH; z++) {
				sum += m1[i*c.M1_WIDTH + z] * m2[z*c.M2_WIDTH + j];
			}
			result[i*c.M2_WIDTH + j] = sum;
		}
	}
}
//void cpu_chunk_serial(const float* m1, const float* m2, config &c, float* result)
//{
//	int chunk = 2;
//	float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
//	float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);
//
//	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
//		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
//			float* temp_result = (float*)calloc(chunk*chunk, sizeof(float));
//			for (int oc = 0; oc<c.M1_WIDTH / chunk; oc++) {
//				for (int ci = 0; ci < chunk; ci++) {
//					for (int cj = 0; cj < chunk; cj++) {
//						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
//						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
//					}
//				}
//				//printf("oc done\n");
//				for (int ci = 0; ci < chunk; ci++) {
//					for (int cj = 0; cj < chunk; cj++) {
//						for (int ic = 0; ic < chunk; ic++) {
//							temp_result[ci*chunk + cj] += cm1[ci*chunk + ic] * cm2[ic*chunk + cj];
//							//printf("%d, %d, %d, %d, %d, %f, %f, %f, %d\n", i, j, oc, ci, cj, cm1[ci*chunk + ic], cm2[ic*chunk + cj], temp_result[ci*chunk + cj], ci*chunk + cj);
//						}
//					}
//				}
//			}
//
//			for (int ci = 0; ci < chunk; ci++) {
//				for (int cj = 0; cj < chunk; cj++) {
//					result[(i*chunk + ci)*c.M1_WIDTH + j * chunk + cj] = temp_result[ci*chunk + cj];
//				}
//			}
//		}
//	}
//}

void cpu_chunk_serial(const float* m1, const float* m2, config &c, float* result)
{
	int chunk = c.chunk;
	float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
	float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);

	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
			float* temp_result = (float*)calloc(chunk*chunk, sizeof(float));
			for(int oc=0;oc<c.M1_WIDTH / chunk;oc++){
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
					}
				}
				// printf("oc done\n");
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						for (int ic = 0; ic < chunk; ic++) {
							temp_result[ci*chunk + cj] += cm1[ci*chunk + ic] * cm2[ic*chunk + cj];
							// printf("%d, %d, %d, %d, %d, %f, %f, %f, %d\n", i, j, oc, ci, cj, cm1[ci*chunk + ic], cm2[ic*chunk + cj], temp_result[ci*chunk + cj], ci*chunk + cj);
						}
					}
				}
			}
			
			for (int ci = 0; ci < chunk; ci++) {
				for (int cj = 0; cj < chunk; cj++) {
					result[(i*chunk + ci)*c.M2_WIDTH + j * chunk + cj] = temp_result[ci*chunk + cj];
				}
			}
			free(temp_result);
		}
	}
	free(cm1);
	free(cm2);
}


void cpu_omp(const float* m1, const float* m2, config &c, float* result)
{
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < c.M1_HEIGHT; i++) {
		for (int j = 0; j < c.M2_WIDTH; j++) {
			float sum = 0.0f;
			for (int z = 0; z < c.M1_WIDTH; z++) {
				sum += m1[i*c.M1_WIDTH + z] * m2[z*c.M2_WIDTH + j];
			}
			result[i*c.M2_WIDTH + j] = sum;
		}
	}
}


void cpu_chunk_omp(const float* m1, const float* m2, config &c, float* result)
{
	int chunk = c.chunk;

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
			float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
			float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);
			float* temp_result = (float*)calloc(chunk*chunk, sizeof(float));
			for (int oc = 0; oc<c.M1_WIDTH / chunk; oc++) {
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
					}
				}
				// printf("oc done\n");
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						for (int ic = 0; ic < chunk; ic++) {
							temp_result[ci*chunk + cj] += cm1[ci*chunk + ic] * cm2[ic*chunk + cj];
							// printf("%d, %d, %d, %d, %d, %f, %f, %f, %d\n", i, j, oc, ci, cj, cm1[ci*chunk + ic], cm2[ic*chunk + cj], temp_result[ci*chunk + cj], ci*chunk + cj);
						}
					}
				}
			}

			for (int ci = 0; ci < chunk; ci++) {
				for (int cj = 0; cj < chunk; cj++) {
					result[(i*chunk + ci)*c.M2_WIDTH + j * chunk + cj] = temp_result[ci*chunk + cj];
				}
			}
			free(cm1);
			free(cm2);
			free(temp_result);
		}
	}
}


void _in_avx(const float* cm1, const float* cm2, float* c_result)
{
	__m256 c_vec[IN_PARALLEL * N / 8];

	for (int i = 0; i < M; i+=IN_PARALLEL) {
		for (int k = 0; k < IN_PARALLEL*N / 8;k++) {
			c_vec[k] = _mm256_setzero_ps();
		}

		for (int k = 0; k < K; k++) {
			__m256 b_vec[N / 8];
			for (int jj = 0; jj < N / 8; jj++) {
				b_vec[jj] = _mm256_load_ps(cm2 + k * N + jj * 8);
			}

			for (int ii = 0; ii < IN_PARALLEL; ii++) {
				__m256 a_vec = _mm256_broadcast_ss(cm1 + (i + ii)*K + k);

				for (int jj = 0; jj < N / 8; jj++) {
					c_vec[ii*N / 8 + jj] = _mm256_fmadd_ps(a_vec, b_vec[jj], c_vec[ii*N / 8 + jj]);
					//__m256 temp = _mm256_mul_ps(a_vec, b_vec[jj]);
					//c_vec[ii*N / 8 + jj] = _mm256_add_ps(temp, c_vec[ii*N / 8 + jj]);
				}
			}
		}
		for (int ii = 0; ii < IN_PARALLEL; ii++) {
			for (int jj = 0; jj < N / 8; jj++) {
				_mm256_store_ps(c_result + (i + ii)*N + jj * 8, c_vec[ii*N / 8 + jj]);
			}
		}
	}
}


void cpu_avx(const float* m1, const float* m2, config &c, float* result)
{
	int chunk = c.chunk;
	float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
	float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);
	float* temp_avx_result = (float*)malloc(sizeof(float)*chunk*chunk);
	float* temp_result = (float*)malloc(sizeof(float)*chunk*chunk);

	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
			//printf("start new chunk\n");
			for (int i = 0; i < chunk*chunk; i++) temp_result[i] = 0.0f;
			for (int oc = 0; oc<c.M1_WIDTH / chunk; oc++) {
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
					}
				}
				/*printf("cm1\n");
				for (int pi = 0; pi < chunk; pi++) {
					for (int pj = 0; pj < chunk; pj++) {
						printf("%f ", cm1[pi*chunk + pj]);
					}
					printf("\n");
				}
				printf("cm2\n");
				for (int pi = 0; pi < chunk; pi++) {
					for (int pj = 0; pj < chunk; pj++) {
						printf("%f ", cm2[pi*chunk + pj]);
					}
					printf("\n");
				}*/
				//printf("accumulating..\n");
				_in_avx(cm1, cm2, temp_avx_result);
				
				for (int pi = 0; pi < chunk; pi++) {
					for (int pj = 0; pj < chunk; pj++) {
						temp_result[pi*chunk + pj] += temp_avx_result[pi*chunk + pj];
					}
				}
			}

			for (int ci = 0; ci < chunk; ci++) {
				for (int cj = 0; cj < chunk; cj++) {
					result[(i*chunk + ci)*c.M2_WIDTH + j * chunk + cj] = temp_result[ci*chunk + cj];
				}
			}
		}
	}
	free(cm1);
	free(cm2);
	free(temp_result);
	free(temp_avx_result);
}


void cpu_avx_omp(const float* m1, const float* m2, config &c, float* result)
{
	int chunk = c.chunk;
#pragma omp parallel for schedule(static)
	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
			//printf("start new chunk\n");
			float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
			float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);

			float* temp_result = (float*)calloc(chunk*chunk, sizeof(float));
			float* temp_avx_result = (float*)malloc(sizeof(float)*chunk*chunk);
			for (int oc = 0; oc<c.M1_WIDTH / chunk; oc++) {
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
					}
				}
				/*printf("cm1\n");
				for (int pi = 0; pi < chunk; pi++) {
				for (int pj = 0; pj < chunk; pj++) {
				printf("%f ", cm1[pi*chunk + pj]);
				}
				printf("\n");
				}
				printf("cm2\n");
				for (int pi = 0; pi < chunk; pi++) {
				for (int pj = 0; pj < chunk; pj++) {
				printf("%f ", cm2[pi*chunk + pj]);
				}
				printf("\n");
				}*/
				//printf("accumulating..\n");
				_in_avx(cm1, cm2, temp_avx_result);

				for (int pi = 0; pi < chunk; pi++) {
					for (int pj = 0; pj < chunk; pj++) {
						temp_result[pi*chunk + pj] += temp_avx_result[pi*chunk + pj];
					}
				}
			}

			for (int ci = 0; ci < chunk; ci++) {
				for (int cj = 0; cj < chunk; cj++) {
					result[(i*chunk + ci)*c.M2_WIDTH + j * chunk + cj] = temp_result[ci*chunk + cj];
				}
			}
			free(temp_result);
			free(temp_avx_result);
			free(cm1);
			free(cm2);
		}
	}
}


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


float* run_cpu(float * m1, float * m2, config &c)
{
	
	float* cpu_serial_result = Timing(cpu_serial, m1, m2, c, "serial", c.cpu_serial_show);

	if (c.run_cpu_chunk) {
		float* cpu_chunk_serial_result = Timing(cpu_chunk_serial, m1, m2, c, "chunk_serial", c.cpu_chunk_show);
		validate(cpu_serial_result, cpu_chunk_serial_result, c);
	}

	if (c.run_cpu_serial_omp) {
		float* cpu_serial_omp_result = Timing(cpu_omp, m1, m2, c, "omp", c.cpu_serial_omp_show);
		validate(cpu_serial_result, cpu_serial_omp_result, c);
	}

	if (c.run_cpu_chunk_omp) {
		float* cpu_chunk_omp_result = Timing(cpu_chunk_omp, m1, m2, c, "chunk_omp", c.cpu_chunk_omp_show);
		validate(cpu_serial_result, cpu_chunk_omp_result, c);
	}

	if (c.run_cpu_serial_chunk_avx) {
		float* cpu_avx_result = Timing(cpu_avx, m1, m2, c, "chunk_avx", c.cpu_serial_chunk_avx_show);
		validate(cpu_serial_result, cpu_avx_result, c);
	}

	if (c.run_cpu_omp_chunk_avx) {
		float* cpu_avx_omp_result = Timing(cpu_avx_omp, m1, m2, c, "chunk_avx_omp", c.cpu_omp_chunk_avx_show);
		validate(cpu_serial_result, cpu_avx_omp_result, c);
	}
	
	return cpu_serial_result;
}

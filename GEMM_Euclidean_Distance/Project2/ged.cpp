#include <iostream>
#include <string>
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <immintrin.h>
using namespace std;

#include "ged.h"
#include "config.h"

typedef void(*cpu_method)(const float* m1, const float* m2, config& c, float* result);

float* Timing(cpu_method f, const float* m1, const float* m2, config& c, string method, bool show)
{
	float* result = (float*)malloc(sizeof(float) * c.M1_samples * c.M2_samples);
	float time_collapse = 0.0;
	for (int t = 0; t < 5; t++) {
		LARGE_INTEGER freq, freqcnt;
		double dcnt, cnt1, cnt2;
		QueryPerformanceFrequency(&freq);
		dcnt = freq.QuadPart;
		QueryPerformanceCounter(&freqcnt);
		cnt1 = freqcnt.QuadPart;

		f(m1, m2, c, result);

		QueryPerformanceCounter(&freqcnt);
		cnt2 = freqcnt.QuadPart;
		time_collapse += (cnt2 - cnt1) / dcnt * 1000;
		if (show) {
			cout << "result" << endl;
			for (int i = 0; i < c.M1_samples; i++) {
				for (int j = 0; j < c.M2_samples; j++) {
					printf("%f ", result[i * c.M2_samples + j]);
				}
				printf("\n");
			}
		}
	}
	cout << "cpu " << method << " --- Average time to compute: " << time_collapse / 5.0 << " ms" << endl;
	return result;
}

void cpu_transpose_naive(const float* m, const int h, const int w, float* result){
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			result[j * h + i] = m[i * w + j];
		}
	}
}

void cpu_gmm_naive(const float* m1, const float* m2, config& c, float* result)
{
	for (int i = 0; i < c.M1_samples; i++) {
		for (int j = 0; j < c.M2_samples; j++) {
			float sum = 0.0f;
			for (int z = 0; z < c.dimensions; z++) {
				sum += m1[i * c.dimensions + z] * m2[j * c.dimensions + z];
			}
			result[i * c.M2_samples + j] = sum;
		}
	}
}

void cpu_square_reduce_naive(const float* m, const int samples, config& c, float* result) {
	for (int i = 0; i < samples; i++) {
		float sum = 0.0f;
		for (int j = 0; j < c.dimensions; j++) {
			sum += m[i * c.dimensions + j] * m[i * c.dimensions + j];
		}
		result[i] = sum;
	}
}

void cpu_operation_fuse_naive(const float* xx, const float* yy, const float* xy, config& c, float* result) {
	for (int i = 0; i < c.M1_samples; i++) {
		for (int j = 0; j < c.M2_samples; j++) {
			result[i * c.M2_samples + j] = sqrt( - 2.0 * xy[i * c.M2_samples + j] + xx[i] + yy[j]);
		}
	}
}

void cpu_ged_naive(const float* m1, const float* m2, config& c, float* result) {
	float* xx = (float*)malloc(sizeof(float) * c.M1_samples);
	float* yy = (float*)malloc(sizeof(float) * c.M2_samples);
	float* xy = (float*)malloc(sizeof(float) * c.M1_samples * c.M2_samples);
	
	cpu_square_reduce_naive(m1, c.M1_samples, c, xx);
	cpu_square_reduce_naive(m2, c.M2_samples, c, yy);
	cpu_gmm_naive(m1, m2, c, xy);
	cpu_operation_fuse_naive(xx, yy, xy, c, result);
}

void _gemm_in_avx(const float* cm1, const float* cm2, float* c_result)
{
	__m256 c_vec[GEMM_IN_PARALLEL * GEMM_SAMPLE_TILE_SIZE / 8];

	for (int i = 0; i < GEMM_SAMPLE_TILE_SIZE; i += GEMM_IN_PARALLEL) {
		for (int k = 0; k < GEMM_IN_PARALLEL * GEMM_SAMPLE_TILE_SIZE / 8; k++) {
			c_vec[k] = _mm256_setzero_ps();
		}
		for (int k = 0; k < GEMM_DIMENSION_TILE_SIZE; k++) {
			__m256 b_vec[GEMM_SAMPLE_TILE_SIZE / 8];
			for (int jj = 0; jj < GEMM_SAMPLE_TILE_SIZE / 8; jj++) {
				b_vec[jj] = _mm256_load_ps(cm2 + k * GEMM_SAMPLE_TILE_SIZE + jj * 8);
			}

			for (int ii = 0; ii < GEMM_IN_PARALLEL; ii++) {
				__m256 a_vec = _mm256_broadcast_ss(cm1 + (i + ii) * GEMM_DIMENSION_TILE_SIZE + k);
#pragma unroll
				for (int jj = 0; jj < GEMM_SAMPLE_TILE_SIZE / 8; jj++) {
					c_vec[ii * GEMM_SAMPLE_TILE_SIZE / 8 + jj] = _mm256_fmadd_ps(a_vec, b_vec[jj], c_vec[ii * GEMM_SAMPLE_TILE_SIZE / 8 + jj]);
				}
			}
		}
		for (int ii = 0; ii < GEMM_IN_PARALLEL; ii++) {
			for (int jj = 0; jj < GEMM_SAMPLE_TILE_SIZE / 8; jj++) {
				_mm256_store_ps(c_result + (i + ii) * GEMM_SAMPLE_TILE_SIZE + jj * 8, c_vec[ii * GEMM_SAMPLE_TILE_SIZE / 8 + jj]);
			}
		}
	}
}

void cpu_gemm_avx(const float* m1, const float* m2t, config& c, float* result)
{
	int sample_tile_size = c.Sample_tile_size;
	int dimension_tile_size = c.Dimension_tile_size;
	float* m2 = (float*)malloc(sizeof(float) * c.dimensions * c.M2_samples);
	float* cm1 = (float*)malloc(sizeof(float) * sample_tile_size * dimension_tile_size);
	float* cm2 = (float*)malloc(sizeof(float) * dimension_tile_size * sample_tile_size);
	float* temp_avx_result = (float*)malloc(sizeof(float) * sample_tile_size * sample_tile_size);
	float* temp_result = (float*)malloc(sizeof(float) * sample_tile_size * sample_tile_size);
	
	cpu_transpose_naive(m2t, c.M2_samples, c.dimensions, m2);

	for (int i = 0; i < c.M1_samples / sample_tile_size; i++) {
		for (int iii = 0; iii < sample_tile_size * sample_tile_size; iii++) temp_result[iii] = 0.0f;
		for (int oc = 0; oc < c.dimensions / dimension_tile_size; oc++) { //moving along dimensions
			for (int ci = 0; ci < sample_tile_size; ci++) { //moving tiled data to L1 cache
				for (int cj = 0; cj < dimension_tile_size; cj++) {
					cm1[ci * dimension_tile_size + cj] = m1[(i * sample_tile_size + ci) * c.dimensions + oc * dimension_tile_size + cj];
					if (ci < c.M2_samples) cm2[cj * sample_tile_size + ci] = m2[(oc * dimension_tile_size + cj) * c.M2_samples + ci];
				}
			}
			_gemm_in_avx(cm1, cm2, temp_avx_result); //compute tiled m1 and tiled m2 using AVX registers
			for (int pi = 0; pi < sample_tile_size; pi++) {
				for (int pj = 0; pj < sample_tile_size; pj++) {
					temp_result[pi * sample_tile_size + pj] += temp_avx_result[pi * sample_tile_size + pj]; //accumulate the result
				}
			}
		}
		for (int ci = 0; ci < sample_tile_size; ci++) {
			for (int cj = 0; cj < c.M2_samples; cj++) {
				result[(i * sample_tile_size + ci) * c.M2_samples + cj] = temp_result[ci * sample_tile_size + cj]; //write back to memory
			}
		}
	}
	free(cm1);
	free(cm2);
	free(temp_result);
	free(temp_avx_result);
}

float inline reduce256(__m256 r) {
	__m128 h = _mm256_extractf128_ps(r, 1);
	__m128 l = _mm256_extractf128_ps(r, 0);
	h = _mm_add_ps(h, l);
	float s[4];
	_mm_store_ps(s, h);
	return s[0] + s[1] + s[2] + s[3];
}

void _norm_in_avx(const float* cm, float* c_result)
{
	for (int i = 0; i < NORM_SAMPLE_TILE_SIZE; i += 1) { //default parallel == 1
		__m256 b_vec[NORM_DIMENSION_TILE_SIZE / 8];
		for (int jj = 0; jj < NORM_DIMENSION_TILE_SIZE / 8; jj++) {
			b_vec[jj] = _mm256_load_ps(cm + i * NORM_DIMENSION_TILE_SIZE + jj * 8);
		}
		b_vec[0] = _mm256_mul_ps(b_vec[0], b_vec[0]);
#pragma unroll
		for (int jj = 1; jj < NORM_DIMENSION_TILE_SIZE / 8; jj++) {
			b_vec[0] = _mm256_fmadd_ps(b_vec[jj], b_vec[jj], b_vec[0]);
		}
		float r = reduce256(b_vec[0]);
		c_result[i] = r;
	}
}

void cpu_norm_avx(const float* m, const int samples, config& c, float* result)
{
	int sample_tile_size = NORM_SAMPLE_TILE_SIZE;
	int dimension_tile_size = NORM_DIMENSION_TILE_SIZE;
	float* cm = (float*)malloc(sizeof(float) * sample_tile_size * dimension_tile_size);
	float* temp_avx_result = (float*)malloc(sizeof(float) * sample_tile_size);
	float* temp_result = (float*)malloc(sizeof(float) * sample_tile_size);

	if (samples < sample_tile_size){
		for (int i = 0; i < samples; i++) temp_result[i] = 0.0f;
		for (int oc = 0; oc < c.dimensions / dimension_tile_size; oc++) {
			for (int ci = 0; ci < samples; ci++) {
				for (int cj = 0; cj < dimension_tile_size; cj++) {
					cm[ci * dimension_tile_size + cj] = m[ci * c.dimensions + oc * dimension_tile_size + cj];
				}
			}
			_norm_in_avx(cm, temp_avx_result);
			for (int pi = 0; pi < samples; pi++) {
				temp_result[pi] += temp_avx_result[pi];
			}
		}
		for (int ci = 0; ci < samples; ci++) {
			result[ci] = temp_result[ci];
		}
	}else {
		for (int i = 0; i < samples / sample_tile_size; i++) {
			for (int iii = 0; iii < sample_tile_size; iii++) temp_result[iii] = 0.0f;
			for (int oc = 0; oc < c.dimensions / dimension_tile_size; oc++) {
				for (int ci = 0; ci < sample_tile_size; ci++) {
					for (int cj = 0; cj < dimension_tile_size; cj++) {
						cm[ci * dimension_tile_size + cj] = m[(i * sample_tile_size + ci) * c.dimensions + oc * dimension_tile_size + cj];
					}
				}
				_norm_in_avx(cm, temp_avx_result);
				for (int pi = 0; pi < sample_tile_size; pi++) {
					temp_result[pi] += temp_avx_result[pi];
				}
			}
			for (int ci = 0; ci < sample_tile_size; ci++) {
				result[i * sample_tile_size + ci] = temp_result[ci];
			}
		}
	}
	free(cm);
	free(temp_result);
	free(temp_avx_result);
}

void _fuse_in_avx(const float* cxx, const float* cxy, const __m256 yy_vec[], const __m256 xy_factor, float* result) {
	__m256 xy_vec[FUSE_T2_SIZE / 8];
	__m256 xx_vec;
	for (int i = 0; i < FUSE_T1_SIZE; i += 1) { //default parallel = 1
		xx_vec = _mm256_broadcast_ss(cxx + i);
		for (int jj = 0; jj < FUSE_T2_SIZE / 8; jj++) {
			xy_vec[jj] = _mm256_load_ps(cxy + i * FUSE_T2_SIZE + jj * 8);
			xy_vec[jj] = _mm256_mul_ps(xy_vec[jj], xy_factor);
			xy_vec[jj] = _mm256_add_ps(xy_vec[jj], xx_vec);
			xy_vec[jj] = _mm256_add_ps(xy_vec[jj], yy_vec[jj]);
			xy_vec[jj] = _mm256_sqrt_ps(xy_vec[jj]);
		}
		for (int jj = 0; jj < FUSE_T2_SIZE / 8; jj++) {
			_mm256_store_ps(result + i * FUSE_T2_SIZE + jj * 8, xy_vec[jj]);
		}
	}
}

void cpu_operation_fuse_avx(const float* xx, const float* yy, const float* xy, config& c, float* result) {

	float* cxx = (float*)malloc(sizeof(float) * FUSE_T1_SIZE);
	float* cxy = (float*)malloc(sizeof(float) * FUSE_T1_SIZE * FUSE_T2_SIZE);
	float* temp_result = (float*)malloc(sizeof(float) * FUSE_T1_SIZE * FUSE_T2_SIZE);

	__m256 cyy[FUSE_T2_SIZE / 8];
	__m256 xy_factor;
	xy_factor = _mm256_set1_ps(-2.0);
	for (int jj = 0; jj < FUSE_T2_SIZE / 8; jj++) {
		cyy[jj] = _mm256_load_ps(yy + jj * 8);
	}
	for (int i = 0; i < c.M1_samples / FUSE_T1_SIZE; i++) {
		for (int iii = 0; iii < FUSE_T1_SIZE * FUSE_T2_SIZE; iii++) temp_result[iii] = 0.0f;
		for (int ci = 0; ci < FUSE_T1_SIZE; ci++) {
			for (int cj = 0; cj < c.M2_samples; cj++) {
				cxy[ci * FUSE_T2_SIZE + cj] = xy[(i * FUSE_T1_SIZE + ci) * c.M2_samples + cj];
			}
			cxx[ci] = xx[i * FUSE_T1_SIZE + ci];
		}
		_fuse_in_avx(cxx, cxy, cyy, xy_factor, temp_result);
		for (int ci = 0; ci < FUSE_T1_SIZE; ci++) {
			for (int cj = 0; cj < c.M2_samples; cj++) {
				result[(i * FUSE_T1_SIZE + ci) * c.M2_samples + cj] = temp_result[ci * FUSE_T1_SIZE + cj];
			}
		}
	}
	free(cxx);
	free(cxy);
	free(temp_result);
}

void cpu_ged_avx(const float* m1, const float* m2, config& c, float* result) {
	float* xx = (float*)malloc(sizeof(float) * c.M1_samples);
	float* yy = (float*)malloc(sizeof(float) * c.M2_samples);
	float* xy = (float*)malloc(sizeof(float) * c.M1_samples * c.M2_samples);

	cpu_norm_avx(m1, c.M1_samples, c, xx);
	cpu_norm_avx(m2, c.M2_samples, c, yy);
	cpu_gemm_avx(m1, m2, c, xy);
	cpu_operation_fuse_naive(xx, yy, xy, c, result);
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

void validate(float* m1, float* m2, config& c)
{
	bool flag = false;
	for (int i = 0; i < c.M1_samples * c.M2_samples; i++) {
		if (_abs_diff(m1[i], m2[i], c.tolarance)) {
			printf("validation failed...\n");
			flag = true;
			break;
		}
	}
	if (flag == false) printf("validation pass...\n");
}

float* run_ged(float* m1, float* m2, config& c) {
	float* cpu_ged_naive_result = Timing(cpu_ged_naive, m1, m2, c, "naive", c.cpu_ged_naive_show);
	float* cpu_ged_avx_result = Timing(cpu_ged_avx, m1, m2, c, "avx", c.cpu_ged_avx_show);
	validate(cpu_ged_naive_result, cpu_ged_avx_result, c);
	return cpu_ged_naive_result;
}
#include <iostream>
#include <string>

#include"data_generator.h"
#include "config.h"
#include "gpu.h"

using std::string;
using std::cout;
using std::endl;

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
void cpu_chunk_serial(const float* m1, const float* m2, config &c, float* result)
{
	int chunk = c.chunk;
	float* cm1 = (float*)malloc(sizeof(float)*chunk*chunk);
	float* cm2 = (float*)malloc(sizeof(float)*chunk*chunk);

	for (int i = 0; i < c.M1_HEIGHT / chunk; i++) {
		for (int j = 0; j < c.M2_WIDTH / chunk; j++) {
			float* temp_result = (float*)calloc(chunk*chunk, sizeof(float));
			for (int oc = 0; oc<c.M1_WIDTH / chunk; oc++) {
				for (int ci = 0; ci < chunk; ci++) {
					for (int cj = 0; cj < chunk; cj++) {
						cm1[ci*chunk + cj] = m1[(i*chunk + ci)*c.M1_WIDTH + oc * chunk + cj];
						cm2[cj*chunk + ci] = m2[(oc * chunk + cj)*c.M2_WIDTH + j * chunk + ci];
					}
				}
				// printf("oc done\n");
#pragma omp parallel for num_threads(8) schedule(dynamic)
				for (int ci = 0; ci < chunk; ci++) {
#pragma unroll
					for (int cj = 0; cj < chunk; cj++) {
#pragma unroll
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

int main()
{
	config c;
	float** matrixes = matrix_brew(c);

	float* cpu_serial_result = Timing(cpu_chunk_serial, matrixes[0], matrixes[1], c, "chunk_serial", c.cpu_serial_show);

	float* result = run_gpu(matrixes[0], matrixes[1], c, cpu_serial_result);

	getchar();
	return 0;
}
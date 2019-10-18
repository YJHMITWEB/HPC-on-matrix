#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Windows.h>
#include <string>
using std::string;
using std::cout;
using std::endl;

#include "cpu.h"

typedef void(*cpu_method)(const float* cube, float* ret, config& c);
void Timing(cpu_method f, const float* cube, float* ret, config& c, string method, bool show)
{
	LARGE_INTEGER freq, freqcnt;
	double dcnt, cnt1, cnt2;
	int test_iters = 2;

	f(cube, ret, c);

	QueryPerformanceFrequency(&freq);
	dcnt = freq.QuadPart;
	QueryPerformanceCounter(&freqcnt);
	cnt1 = freqcnt.QuadPart;

	for(int i=0;i<test_iters;i++)
		f(cube, ret, c);

	QueryPerformanceCounter(&freqcnt);
	cnt2 = freqcnt.QuadPart;
	
	cout << "cpu " << method << " --- Time to compute: " << (cnt2 - cnt1) / test_iters / dcnt * 1000 << " ms" << endl;
}

void cpu_serial(const float* cube, float* ret, config& c)
{
	for (int z = RADIUS; z < c.Z - RADIUS; z++) {
		for (int y = RADIUS; y < c.Y - RADIUS; y++) {
			for (int x = RADIUS; x < c.X - RADIUS; x++) {
				float o = cube[z*c.Y*c.X + y * c.X + x];
				for (int r = 1; r < RADIUS + 1; r++) {
					o += cube[(z - r)*c.Y*c.X + y * c.X + x] +
						cube[(z + r)*c.Y*c.X + y * c.X + x] +
						cube[z*c.Y*c.X + (y - r) * c.X + x] +
						cube[z*c.Y*c.X + (y + r) * c.X + x] +
						cube[z*c.Y*c.X + y * c.X + x - r] +
						cube[z*c.Y*c.X + y * c.X + x + r];
				}
				ret[z*c.Y*c.X + y * c.X + x] = o * c.WT;
			}
		}
	}
}

void run_cpu(float* cube)
{
	config c;
	float* ret = (float*)malloc(sizeof(float)*c.X*c.Y*c.Z);

	Timing(cpu_serial, cube, ret, c, "serial", false);

	if (false) {
		printf("CPU result...\n");
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

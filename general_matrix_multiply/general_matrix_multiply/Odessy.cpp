#include<stdio.h>

#include"data_generator.h"
#include "config.h"
#include "cpu.h"
#include "gpu.h"

int main()
{
	config c;
	float** matrixes = matrix_brew(c);
	float* result = run_cpu(matrixes[0], matrixes[1], c);
	run_gpu(matrixes[0], matrixes[1], c);

	getchar();
	return 0;
}
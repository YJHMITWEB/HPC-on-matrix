#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "gpu.h"

int main()
{
	config c;
	float* cube = (float*)malloc(sizeof(float)*c.X*c.Y*c.Z);
	data_brew(cube, c);

	run_cpu(cube);
	run_gpu(cube);

	free(cube);

	getchar();
	return 0;
}
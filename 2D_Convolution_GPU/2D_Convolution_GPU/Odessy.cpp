#include <stdio.h>
#include "data_generator.h"
#include "gpu.h"

int main()
{
	config c;
	float* matrix = matrix_brew(c);
	float* kernel = kernel_brew(c);

	run_gpu(matrix, kernel, c);
	getchar();
	return 0;

}
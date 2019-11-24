#include <stdio.h>
#include "data_generator.h"
#include "cpu.h"

int main()
{
	config c;
	float* matrix = matrix_brew(c);
	float* kernel = kernel_brew(c);

	float* result = run_cpu(matrix, kernel, c);
	
	getchar();
	return 0;
}
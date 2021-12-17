#include<stdio.h>

#include"data_generator.h"
#include "config.h"
#include "ged.h"

int main()
{
	config c;
	float** matrixes = matrix_brew(c);
	float* result = run_ged(matrixes[0], matrixes[1], c);

	return 0;
}
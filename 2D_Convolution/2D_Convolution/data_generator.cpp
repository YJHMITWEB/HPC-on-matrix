#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;

#include "data_generator.h"

void matrix_brew_op(float* m, int w, int h)
{
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			m[i*w + j] = rand() / (float)(RAND_MAX);
		}
	}
}

void matrix_show(float* m, int w, int h, string name)
{
	cout << name << endl;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			printf("%f ", m[i*w + j]);
		}
		printf("\n");
	}
	printf("\n");
}

float * matrix_brew(config & c)
{
	float* m = (float*)malloc(sizeof(float)*c.M_HEIGHT*c.M_WIDTH);

	matrix_brew_op(m, c.M_WIDTH, c.M_HEIGHT);
	if (c.data_show) {
		string m_name("Matrix");
		matrix_show(m, c.M_WIDTH, c.M_HEIGHT, m_name);
	}
	return m;
}

float * kernel_brew(config & c)
{
	float* m = (float*)malloc(sizeof(float)*c.k_size*c.k_size);

	matrix_brew_op(m, c.k_size, c.k_size);
	if (c.data_show) {
		string m_name("Kernel");
		matrix_show(m, c.k_size, c.k_size, m_name);
	}
	return m;
}

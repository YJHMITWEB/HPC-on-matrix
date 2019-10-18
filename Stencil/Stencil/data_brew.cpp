#include <stdio.h>
#include <stdlib.h>

#include "data_brew.h"

void data_brew(float* cube, config& c)
{
	for (int z = 0; z < c.Z; z++) {
		for (int y = 0; y < c.Y; y++) {
			for (int x = 0; x < c.X; x++) {
				cube[z*c.Y*c.X + y * c.X + x] = rand() / (float)(RAND_MAX);
				//printf("%5.4f ", cube[z*c.Y*c.X + y * c.X + x]);
			}
			//printf("\n");
		}
		//printf("\n\n\n");
	}
}

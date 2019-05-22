#pragma once

#define SQUARE 1024
#define CHUNK 64
#define TILE_WIDTH 32

typedef struct CONFIG {
	int M1_WIDTH = SQUARE;
	int M1_HEIGHT = SQUARE;
	int M2_WIDTH = SQUARE;
	int chunk = CHUNK;

	bool data_show = false;

	bool cpu_serial_show = false;

	float tolarance = 0.001f;
}config;


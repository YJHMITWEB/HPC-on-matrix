#pragma once
#define WORK_ZONE 512
#define RADIUS 6
#define BS 32

typedef struct Config {
	int X = WORK_ZONE + 2 * RADIUS;
	int Y = WORK_ZONE + 2 * RADIUS;
	int Z = WORK_ZONE + 2 * RADIUS;

	float WT = 1.0 / (RADIUS * 6 + 1);

}config;
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>

#include "config.h"
#include "gpu_entrance.h"

float* run_gpu(float* m1, float* m2, config& c, float* cpu_result);

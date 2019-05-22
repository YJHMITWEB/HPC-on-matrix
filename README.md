# HPC-on-matrix
* General Matrix Multiply:  
  * Main Entrance: Odessy.cpp  
  * GPU Main Entrance: GPN/../Odessy_G.cpp
* Environment:  
  * Windows 10  
  * VS2017 Release x64  
  * Intel Core i7 4720HQ (4 cores, 8 threads)  
  * 8G memory, 1.6GHz  
  * GTX 960M, 640 cu
* Implemented Matrix Multiply(MM):  

  |  Methods        |   OpenMP    |   CHUNK     |     SSE/AVX    | CUDA |   Time(ms)      |       Validation   |      speed-up|
  |-----------------|:-------------:|:-------------:|:----------------:|:--------------:|:--------------------:|:-----------:|
  |CPU serial:      |       |    |     |      |  2771.54  |    pass   |        1x  |
  |CPU chunk-serial:   |    |  ✔   |    | |   936.43   |   pass   |     2.96x  |
  |CPU omp:        |     ✔  |     |   |      |   1562.25  |    pass  |      1.77x  |
  |CPU chunk-omp:    |   ✔    |  ✔   |   |       |    176.31  |    pass  |     15.72x  |
  |CPU chunk-avx:    |      | ✔   |   ✔  |      |    182.70   |   pass   |    15.17x  |
  |CPU chunk-avx-omp:  | ✔   |   ✔   |   ✔ | |     44.61  |    pass  |     62.13x  |
  |GPU AB:         |          |      |           |     ✔      |35.14 | pass | 78.87x|
  |GPU ABT:         |          |      |            |     ✔      |168.00 | pass | 16.50x|
  |GPU ATB:         |          |      |            |     ✔      |32.61 | pass | 85.00x|
  |GPU AB (shared memory):         |               |      |      |     ✔      |24.00 | pass | 115.48x|


* General settings:  
  
  * Size of Matrix1 \* Matrix2: (1024, 1024) \* (1024, 1024)  
  * Chunk size: 64\*64 elements  
  * OMP threads: 8  
  * AVX SIMD parallel: 8\*float(32 bits)  
  * AVX parallel in chunk: 2 rows (depends on how many AVX registers on the machine)  
  * CUDA: Block(32, 32)

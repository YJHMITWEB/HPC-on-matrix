# HPC-on-matrix:
The purpose of this project is to show how high performance computing, which is also perceived as parallel computing, distributed computing or heterogeneous computing, could accelerate very common algorithms in nowadays Artificial Intelligence. Contains matrix multiplying, 2D convolution computing, and 3D Stencil computing.
## Matrix Multiply
* General Matrix Multiply:  
  * Main Entrance: Odessy.cpp  
  * GPU Main Entrance: GPN/../Odessy_G.cpp
* Environment:  
  * Windows 10  
  * VS2017 Release x64  
  * Intel Core i7 4720HQ (4 cores, 8 threads, L1c = 128 KB, L2c = 1024 KB, L3c = 6144 KB)  
  * 8G memory, 1.6GHz  
  * GTX 960M, 640 cu
* Implemented Matrix Multiply(MM):  

  |  Methods        |   OpenMP    |   CHUNK     |     SSE/AVX    | CUDA |NEON|   Time(ms)      |       Validation   |      speed-up|  % of peak performance
  |-----------------|:-------------:|:-------------:|:--------------:|:--:|:----------------:|:--------------:|:--------------------:|:-----------:|:----------:|
  |CPU sequential:      |       |    |     |     |  |  2771.54  |    pass   |        1x  | |
  |CPU chunk-sequential:   |    |  ✔   |    |  ||   936.43   |   pass   |     2.96x  |
  |CPU omp:        |     ✔  |     |   |  |     |   1562.25  |    pass  |      1.77x   ||
  |CPU chunk-omp:    |   ✔    |  ✔   |   |        ||    176.31  |    pass  |     15.72x  | |
  |CPU chunk-avx:    |      | ✔   |   ✔  |       ||    182.70   |   pass   |    15.17x  | |
  |CPU chunk-avx-omp:  | ✔   |   ✔   |   ✔ |  ||     44.61  |    pass  |     62.13x  | |
  |GPU AB:         |          |      |           |     ✔      | |35.14 | pass | 78.87x ||
  |GPU ABT:         |          |      |            |     ✔      | |168.00 | pass | 16.50x ||
  |GPU ATB:         |          |      |            |     ✔      | |32.61 | pass | 85.00x ||
  |GPU AB (shared memory):         |               |      |      |     ✔    |   |24.00 | pass | 115.48x ||
  |ARM A15 (L1 Cache Tiling):         |               |    ✔    |      |         |   ✔  | | pass ||70.56|


* General settings:  
  
  * Size of Matrix1 \* Matrix2: (1024, 1024) \* (1024, 1024)  
  * Chunk size: 64\*64 elements  
  * OMP threads: 8  
  * AVX SIMD parallel: 8\*float(32 bits)  
  * AVX parallel in chunk: 2 rows (depends on how many AVX registers on the machine)  
  * CUDA: Block(32, 32)
## HPC-on-matrix: 3D Stencil computing
* General Matrix Multiply:  
  * Main Entrance: Odessy.cpp
  
* Environment:  
  * Windows 10  
  * VS2017 Release x64  
  * Intel Core i7 4720HQ (4 cores, 8 threads, L1c = 128 KB, L2c = 1024 KB, L3c = 6144 KB)  
  * 8G memory, 1.6GHz  
  * GTX 960M, 640 cu
  
* Implemented 3D Stencil computing:  

|  Methods          |   OpenMP        |   CUDA      | Time(ms)         |       Validation |      speed-up  | GFlops|
|-----------------  |:---------------:|:-----------:|:----------------:|:----------------:|:--------------:|:--:|
|CPU sequential:        |                 |             |3222.1            |  pass            | 1x             | |
|GPU:               |                 |     ✔       |71.04            |  pass            | 45.36x             |117.5|

* General settings:  
  
  * Size of Tensor: (512, 512, 512)
  * Stencil radius: 6  
  * Kernel size: 13  
  * CUDA: Block(32, 32)
* Reference:

  * [3D Finite Difference Computation on GPUs using CUDA, Paulius Micikevicius, NVIDIA](https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf)

* X86 CPU's dilemma:

   * 3D Stencil requires accessing data on three dimensions consecutively, however, the cache on X86 is one dimensional. Let's see what happens when doing 3D tiling on L1 cache. Given that X86's L1 cache size is 32KB, 8 * 1024 32-bit-float values could be stored. If using 3D cublic tiling, the length of each dimension is 16. Then, we do 3D tiling again on L2 cache. Given that X86's L2 cache size is 256KB, if using 3D cublic tiling, the length of each dimension is 32. Here comes that problem --- the length of cache line on X86 is 64 Bytes, therefore hardware pre-fetching cannot be used properly, which may even cause negative effects on performance. 
 
   * Considering cache tiling, no matter doing it on L1 or L2, the tile size will be too small to exploit the capacity of X86 cache hierarchy, which is why in any 3D Stencil implementation on X86 will always be limited by L3 cache and RAM's performance.
   
   * This problem will happen again on NVIDIA GPU if trying to store 3D data on shared memory. A way to solve it is on above reference, and my code shows exactly how to do.

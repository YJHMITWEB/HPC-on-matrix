# HPC-on-statistics-and-matrix
* General Matrix Multiply:  
  * Main Extrance: Odessy.cpp  
* Environment:  
  * Windows 10  
  * VS2017 Release x64  
  * Intel Core I7 4720HQ (4 cores, 8 threads)  
  * 8G memory, 1.6GHz  
* Implemented Matrix Multiply(MM):  

  |  Methods        |   OpenMP    |   CHUNK     |     SSE/AVX    |   Time(ms)      |       Validation   |      speed-up|
  |-----------------|:-------------:|:-------------:|:----------------:|--------------|:--------------------:|-----------|
  |CPU serial:      |       |    |     |  2771.54  |    pass   |        1x  |
  |CPU chunk-serial:   |    |  ✔   |     |   936.43   |   pass   |     2.96x  |
  |CPU omp:        |     ✔  |     |   |   1562.25  |    pass  |      1.77x  |
  |CPU chunk-omp:    |   ✔    |  ✔   |    |    176.31  |    pass  |     15.72x  |
  |CPU chunk-avx:    |      | ✔   |   ✔  |    182.70   |   pass   |    15.17x  |
  |CPU chunk-avx-omp:  | ✔   |   ✔   |   ✔  |     44.61  |    pass  |     62.13x  |
    
* General settings:  
  
  * Matrix1 \* Matrix2: (1024, 1024) \* (1024, 1024)  
  * chunk size: 64\*64 elements  
  * OMP threads: 8  
  * AVX SIMD parallel: 8\*float(32 bits)  
  * AVX parallel on Matrix: 2 rows (depends on how many AVX registers on the machine)  

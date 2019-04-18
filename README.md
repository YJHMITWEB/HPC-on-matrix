# HPC-on-statistics-and-matrix
General Matrix Multiply:  
  Main Extrance: Odessy.cpp  
Environment:  
  Windows 10  
  VS2017 Release x64  
  Intel Core I7 4720HQ (4 cores, 8 threads)  
  8G memory, 1.6GHz  
Implemented Matrix Multiply(MM):  
  
  CPU serial:          off-OMP     off-chunk     off-SSE/AVX    2771.54ms      validation-pass           1x  
  CPU chunk-serial:    off-OMP     on-chunk      off-SSE/AVX     936.43ms      validation-pass        2.96x  
  CPU omp:             on-OMP      off-chunk     off-SSE/AVX    1562.25ms      validation-pass        1.77x  
  CPU chunk-omp:       on-OMP      on-chunk      off-SSE/AVX     176.31ms      validation-pass       15.72x  
  CPU chunk-avx:       off-OMP     on-chunk      on-SSE/AVX      182.70ms      validation-pass       15.17x  
  CPU chunk-avx-omp:   on-OMP      on-chunk      on-SSE/AVX       44.61ms      validation-pass       62.13x  
    
General settings:  
  
  Matrix1 \* Matrix2: (1024, 1024) \* (1024, 1024)  
  chunk size: 64\*64 elements  
  OMP threads: 8  
  AVX SIMD parallel: 8\*float(32 bits)  
  AVX parallel on Matrix: 2 rows (depends on how many AVX registers on the machine)  

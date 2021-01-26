# Mandelbrot example

This folder contains a CPython (pure Python implementation) of the Mandelbrot
example from the presentation.


## Exercise

Modify the example so that it is:

- Compiled with the @jit decorator, to run as native code on the CPU.
- Compiled with the @cuda.jit decorator, to run on the GPU.

Measure the performance of each variant. How quickly can you get each variant to
run?


### Further exercise

Can you also rewrite the mandel function using @vectorize for the CUDA target?


## Solutions

- `mandel_cpython.py`:   Baseline version, no compilation
- `mandel_jit.py`:       JIT compilation for CPU with Numba
- `mandel_vectorize.py`: Implementation using UFuncs for CUDA
- `mandel_cuda.py`:      JIT compilation for CUDA with Numba


## Timings

On my machine (Intel Core i7-6700K, Quadro RTX 8000), timings are as follows:

- CPython:    79.95 seconds (  1x)
- JIT:         1.12 seconds ( 71x)
- CUDA JIT:    0.42 seconds (190x)
- CUDA UFunc:  0.23 seconds (347x)

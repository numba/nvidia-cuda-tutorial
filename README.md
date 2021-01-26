# Numba for CUDA Programmers

Author: Graham Markall, NVIDIA <gmarkall@nvidia.com>.


## What is this course?

This is an adapted version of one delivered internally at NVIDIA - its primary
audience is those who are familiar with CUDA C/C++ programming, but perhaps less
so with Python and its ecosystem. That said, it should be useful to those
familiar with the Python and PyData ecosystem.

It focuses on using CUDA concepts in Python, rather than going over basic CUDA
concepts - those unfamiliar with CUDA may want to build a base understanding by
working through Mark Harris's [An Even Easier Introduction to
CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) blog
post, and briefly reading through the [CUDA Programming
Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
Chapters 1 and 2 (*Introduction* and *Programming Model*). Other concepts
discussed in the course (such as shared memory) are discussed in later chapters.
For expediency, it is recommended to look up concepts in those sections when
necessary, rather than reading all the reference material in detail.


## What is in this course?

The course is broken into 5 sessions, designed for a session to be presented
then the examples and exercises worked through before participants move to the
next session. This could be presented at a cadence of one session per week with
an hour of presentation time to fit the course around other tasks. Alternatively
it could be delivered as a tutorial session over the course of 2-3 days.


### Session 1: An introduction to Numba and CUDA Python

Session 1 files are in the [session-1](session-1) folder. Contents:

- *Presentation*: The presentation for this session, along with notes.
- *Mandelbrot example*: See [the README](session-1/mandelbrot/README.md) for 
  exercises.
- *CUDA Kernels notebook*: In the [exercises](session-1/exercises) folder. Open the 
  notebook using Jupyter.
- *UFuncs notebooks* In the [exercises](session-1/exercises) folder. Open the 
  notebooks using Jupyter. Contains two notebooks on `vectorize` and 
  `guvectorize` on the CPU (as it's a little easier to experiment with them on
  the CPU target) and one notebook on CUDA ufuncs and memory management.


### Session 2: Typing

Session 2 files are in the [session-2](session-2) folder. Contents:

- *Presentation*: The presentation for this session, along with notes.
- *Exercises*: In the [exercises](session-2/exercises) folder. Open the notebook
  using Jupyter.


### Session 3: Porting strategies, performance, interoperability, debugging

Session 3 files are in the [session-3](session-3) folder. Contents:

- *Presentation*: The presentation for this session, along with notes.
- *Exercises*: In the [exercises](session-3/exercises) folder. Open the notebook
  using Jupyter.
- *Examples*: In the [examples](session-3/examples) folder. These are mostly
  executable versions of the examples given in the slides.


### Session 4: Extending Numba

Session 4 files are in the [session-4](session-4) folder. Contents:

- *Presentation*: The presentation for this session, along with notes.
- *Exercises*: In the [exercises](session-4/exercises) folder. Open the notebook
  using Jupyter. A solution to the exercise is also provided.
- *Examples*: In the [examples](session-4/examples) folder. This contains a
  notebook working through the Interval example presented in the slides.


### Session 5: Memory Management

Session 5 files are in the [session-5](session-5) folder. Contents:

- *Presentation*: The presentation for this session, along with notes.
- *Exercises*: In the [exercises](session-5/exercises) folder. Open the notebook
  using Jupyter.
- *Examples*: In the [examples](session-5/examples) folder. This contains
  examples of a simple EMM Plugin wrapping cudaMalloc, and an EMM Plugin for
  using the CuPy pool allocator with Numba.


## Sources

Some of the material in this course is derived from various sources. These
sources, are:

- [PyData Amsterdam 2019 Numba
  Tutorial](https://github.com/ContinuumIO/pydata-amsterdam2019-numba). This
  material was used under the [Creative Commons Attribution 4.0 International
  license](https://github.com/ContinuumIO/pydata-amsterdam2019-numba/blob/c5944f1cf0a5244dcd43d690d3676ade19ce6e16/LICENSE).
  Minor edits were made to this source material to make the wording more
  consistent with the rest of this tutorial.
- [Numba tutorials I've previously presented](https://github.com/gmarkall/tutorials)
- [An example extending Numba's CUDA target](https://github.com/gmarkall/extending-numba-cuda)
- The Life of a Numba Kernel: [Notebook](https://github.com/gmarkall/life-of-a-numba-kernel/blob/master/Life%20of%20a%20Numba%20Kernel%20-%20with-%20output.ipynb) and [blog post](https://medium.com/rapids-ai/the-life-of-a-numba-kernel-a-compilation-pipeline-taking-user-defined-functions-in-python-to-cuda-71cc39b77625).

## References

The following references can be useful for studying CUDA programming in general,
and the intermediate languages used in the implementation of Numba:

- [The CUDA C/C++ Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
  Early chapters provide some background on the CUDA parallel execution model
  and programming model.
- [LLVM 7.0.0 Language reference
  manual](https://releases.llvm.org/7.0.0/docs/LangRef.html). Documents the
  instructions in LLVM IR for version 7, the current version used in Numba for
  CUDA.
- [NVVM IR Specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html).
  Provides details of additions to, and deviations from, the LLVM 7 IR
  specification.
- [PTX ISA
  documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
  The Parallel Thread Execution ISA, which targets CUDA GPUs. This is a useful
  reference when dumping the assembly of CUDA code from Numba.

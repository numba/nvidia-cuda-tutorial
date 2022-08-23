# An example CUDA Array Interface implementation that wraps a pointer provided
# by cudaMalloc.

from numba import cuda
from ctypes import CDLL, POINTER, byref, c_void_p, c_size_t
import cupy as cp


class MyArray:
    def __init__(self, shape, typestr, data):
        if isinstance(shape, int):
            shape = (shape,)

        self._shape = shape
        self._data = data
        self._typestr = typestr

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': self._shape,
            'typestr': self._typestr,
            'data': (self._data, False),
            'version': 2
        }


# Use ctypes to get the cudaMalloc function from Python
cudart = CDLL('libcudart.so')
cudaMalloc = cudart.cudaMalloc
cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]

# Allocate some Numba-external memory with cudaMalloc
ptr = c_void_p()
float32_size = 4
nelems = 32
alloc_size = float32_size * nelems
cudaMalloc(byref(ptr), alloc_size)

# Wrap our memory in a CUDA Array Interface object
arr = MyArray(nelems, 'f4', ptr.value)


# Call a kernel on our object wrapping the pointer

@cuda.jit
def initialize(x):
    i = cuda.grid(1)
    if i < len(x):
        x[i] = 3.14


initialize[1, nelems](arr)


# Use CuPy for a convenient way to print our data to show that the kernel
# initialized it
print(cp.asarray(arr))

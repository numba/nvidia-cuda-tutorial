from ctypes import CDLL, POINTER, byref, c_void_p, c_size_t
from numba import cuda
from numba.cuda import (HostOnlyCUDAMemoryManager, GetIpcHandleMixin,
                        MemoryPointer, MemoryInfo)


# Open the CUDA runtime DLL and create bindings for the cudaMalloc, cudaFree,
# and cudaMemGetInfo functions.

cudart = CDLL('libcudart.so')

cudaMalloc = cudart.cudaMalloc
cudaMalloc.argtypes = [POINTER(c_size_t), c_size_t]

cudaFree = cudart.cudaFree
cudaFree.argtypes = [c_void_p]

cudaMemGetInfo = cudart.cudaMemGetInfo
cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]


# Python functions for allocation, deallocation, and memory info

def my_alloc(size):
    """
    Allocate `size` bytes of device memory and return a device pointer to the
    allocated memory.
    """
    ptr = c_size_t()
    ret = cudaMalloc(byref(ptr), size)
    if ret:
        raise RuntimeError(f'Unexpected return code {ret} from cudaMalloc')
    return ptr


def my_free(ptr):
    """
    Free device memory pointed to by `ptr`.
    """
    cudaFree(ptr)


def my_memory_info():
    free = c_size_t()
    total = c_size_t()
    cudaMemGetInfo(byref(free), byref(total))
    return free, total


# EMM Plugin implementation

class MyEMMPlugin(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    def memalloc(self, size):
        ptr = my_alloc(size)
        ctx = self.context
        finalizer = make_finalizer(ptr.value)
        return MemoryPointer(ctx, ptr, size, finalizer=finalizer)

    def initialize(self):
        # No setup required to use the EMM Plugin in a given context
        pass

    def get_memory_info(self):
        free, total = my_memory_info()
        return MemoryInfo(free=free.value, total=total.value)

    @property
    def interface_version(self):
        return 1


def make_finalizer(ptr):
    def finalizer():
        my_free(ptr)

    return finalizer


# If NUMBA_CUDA_MEMORY_MANAGER is set to this module (e.g.
# `NUMBA_CUDA_MEMORY_MANAGER=simple_emm_plugin`), then Numba will look at the
# _numba_memory_manager global to determine what class to use for memory
# management.
#
# This can be used to run the Numba test suite with the plugin, to verify that
# the plugin is working correctly. For example, if the directory of this module
# is on PYTHONPATH, then running:
#
#   NUMBA_CUDA_MEMORY_MANAGER=simple_emm_plugin python -m numba.runtests \
#       numba.cuda.tests
#
# will run all Numba CUDA tests with the plugin enabled.

_numba_memory_manager = MyEMMPlugin


if __name__ == '__main__':
    # Quick test of setting the memory manager and allocating/deleting an array
    cuda.set_memory_manager(MyEMMPlugin)
    ctx = cuda.current_context()
    print(f"Free before creating device array: {ctx.get_memory_info().free}")
    x = cuda.device_array(1000)
    print(f"Free after creating device array: {ctx.get_memory_info().free}")
    del x
    print(f"Free after freeing device array: {ctx.get_memory_info().free}")

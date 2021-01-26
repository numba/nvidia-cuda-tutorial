# An example EMM plugin that enables Numba to use CuPy's memory pool instead of
# its internal memory management.
#
# This is sufficient for passing all Numba tests when the testsuite is run
# with:
#
#   export NUMBA_CUDA_MEMORY_MANAGER=cupy_emm_plugin
#   python -m numba.runtests numba.cuda.tests

from contextlib import contextmanager

from numba import cuda
from numba.cuda import (GetIpcHandleMixin, HostOnlyCUDAMemoryManager,
                        MemoryPointer, MemoryInfo)

import ctypes
import cupy

# Set to False for a quieter run
LOGGING = True


class CuPyNumbaManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logging = LOGGING
        # We keep a record of all allocations, and remove each allocation
        # record in the finalizer, which results in it being returned back to
        # the CuPy memory pool.
        self._allocations = {}
        # The CuPy memory pool.
        self._mp = None

    def memalloc(self, nbytes):
        # Allocate from the CuPy pool and wrap the result in a MemoryPointer as
        # required by Numba.
        cp_mp = self._mp.malloc(nbytes)
        if self._logging:
            print("Allocated %d bytes at %x" % (nbytes, cp_mp.ptr))
        self._allocations[cp_mp.ptr] = cp_mp
        return MemoryPointer(
            cuda.current_context(),
            ctypes.c_uint64(int(cp_mp.ptr)),
            nbytes,
            finalizer=self._make_finalizer(cp_mp, nbytes)
        )

    def _make_finalizer(self, cp_mp, nbytes):
        allocations = self._allocations
        ptr = cp_mp.ptr
        logging = self._logging

        def finalizer():
            if logging:
                print("Freeing %d bytes at %x" % (nbytes, ptr))
            # Removing the last reference to the allocation causes it to be
            # garbage-collected and returned to the pool.
            allocations.pop(ptr)

        return finalizer

    def get_memory_info(self):
        # Memory info can be obtained from CuPy's memory pool. This is not the
        # total GPU memory, but pertains only to the pool.
        return MemoryInfo(free=self._mp.free_bytes(),
                          total=self._mp.total_bytes())

    def initialize(self):
        super().initialize()
        # Get a memory pool for this context.
        self._mp = cupy.get_default_memory_pool()

    def reset(self):
        # Free all blocks when reset.
        if self._mp:
            self._mp.free_all_blocks()

    @contextmanager
    def defer_cleanup(self):
        # This doesn't actually defer returning memory back to the pool, but
        # returning memory to the pool will not interrupt async operations like
        # an actual cudaFree / cuMemFree would.
        with super().defer_cleanup():
            yield

    @property
    def interface_version(self):
        return 1


# For Numba to find the EMM plugin in this module when instructed to do so with
# the environment variable NUMBA_CUDA_MEMORY_MANAGER.
_numba_memory_manager = CuPyNumbaManager

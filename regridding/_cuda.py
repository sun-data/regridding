"""
Allocating and filling arrays which live on a CUDA device.

Both device kernels need the same few operations on device memory, and
:mod:`numba` gives its allocation and its kernels annotations which describe
how the compiler calls them rather than how a caller does, so the waivers
for that are kept here instead of at each use.
"""

from typing import Any
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver

__all__ = [
    "allocate",
    "fill",
    "zeros",
]

_threads = 256
"""The number of threads in each block, where nothing better is known."""


# this runs on the device, where `coverage` cannot follow it, so it reports
# the body as missed even when it does the work
@cuda.jit
def _fill(a, value):  # pragma: nocover
    """Fill a device array, which is cheaper than sending one from the host."""
    i = cuda.grid(1)  # type: ignore[call-arg]
    if i < a.size:
        a[i] = value


def allocate(shape: Any, dtype: np.typing.DTypeLike) -> Any:
    """
    Allocate an array on the device, without initializing it.

    Parameters
    ----------
    shape
        The shape of the array.
    dtype
        The type of the array's elements.
    """
    return cuda.device_array(shape, dtype)  # type: ignore[arg-type]


def fill(a: Any, value: Any, threads: int = _threads) -> Any:
    """
    Fill a device array with a value, and return it.

    A value whose bytes are all the same, such as zero or an integer of
    every bit set, is filled by the driver rather than by a kernel.  That
    is most of an order of magnitude cheaper: about 0.1 ms against 2 ms
    for the four million elements an ESIS-sized grid reserves.

    Parameters
    ----------
    a
        The array to fill, which has to be contiguous.  An empty one is
        left alone, since a kernel cannot be launched with zero blocks.
    value
        The value to fill it with.
    threads
        The number of threads in each block.
    """
    if not a.size:
        return a

    pattern = np.array(value, dtype=a.dtype).tobytes()

    if len(set(pattern)) == 1:
        driver.device_memset(a, pattern[0], a.nbytes)
    else:
        flat = a.reshape(-1)
        _fill[(flat.size + threads - 1) // threads, threads](flat, value)  # type: ignore[index]

    return a


def zeros(shape: Any, dtype: np.typing.DTypeLike) -> Any:
    """
    Allocate an array of zeros on the device.

    Parameters
    ----------
    shape
        The shape of the array.
    dtype
        The type of the array's elements.
    """
    return fill(allocate(shape, dtype), 0)

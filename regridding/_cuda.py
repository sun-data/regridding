"""
Allocating and filling arrays which live on a CUDA device.

The device kernels need the same few operations on device memory, and
:mod:`numba` gives its allocation and its kernels annotations which describe
how the compiler calls them rather than how a caller does, so the waivers
for that are kept here instead of at each use.
"""

from typing import Any
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver

__all__ = [
    "threads",
    "available",
    "on_device",
    "allocate",
    "fill",
    "rows",
    "zeros",
    "prefix_sum",
]

threads = 256
"""
The number of threads in each block.

One number for every kernel here: none of them has been shown to want a
different one, so a difference between them would only look deliberate
without being so.
"""


def available() -> bool:
    """
    Test whether there is a CUDA device to build weights on.

    This is what ``device="cuda"`` needs, so it is worth asking before
    passing it rather than catching the failure.
    """
    try:
        return cuda.is_available()
    except Exception:  # pragma: nocover
        return False


def on_device(weights: np.ndarray) -> bool:
    """
    Test whether a set of weights lives in device memory.

    Parameters
    ----------
    weights
        Weights built by :func:`regridding.weights`.
    """
    flat = np.asarray(weights).reshape(-1)
    if not flat.size:  # pragma: nocover
        return False
    return cuda.is_cuda_array(flat[0][2])


# this runs on the device, where `coverage` cannot follow it, so it reports
# the body as missed even when it does the work.  `a` is a device array,
# which `numba` ships no type for, so it is annotated as `Any`
@cuda.jit
def _fill(a: Any, value: Any) -> None:  # pragma: nocover
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


def fill(a: Any, value: Any, threads: int = threads) -> Any:
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

    Raises
    ------
    ValueError
        If the array is not contiguous.
    """
    if not a.size:
        return a

    if not a.is_c_contiguous():
        raise ValueError(
            "a device array has to be contiguous to be filled: the driver "
            "is asked for `nbytes` from the start of it, which is not where "
            "a strided view keeps its elements"
        )

    pattern = np.array(value, dtype=a.dtype).tobytes()

    if len(set(pattern)) == 1:
        driver.device_memset(a, pattern[0], a.nbytes)
    else:
        flat = a.reshape(-1)
        _fill[(flat.size + threads - 1) // threads, threads](flat, value)  # type: ignore[index]

    return a


def rows(
    a: np.ndarray,
    ndim: int,
    shape_orthogonal: tuple[int, ...],
) -> tuple[Any, np.ndarray]:
    """
    Send an array which is broadcast along the orthogonal axes of a set of
    weights to the device, once for each distinct row of it.

    Every orthogonal axis along which `a` has been broadcast, which is to
    say has a stride of zero, is reduced to a length of one before it is
    sent, so an array which the caller broadcast is sent no larger than it
    was before.  The orthogonal axes are then flattened into one axis of
    rows.

    Parameters
    ----------
    a
        The array, with its orthogonal axes first, matched to
        `shape_orthogonal` from the right, followed by the axes of each row.
    ndim
        The number of axes of each row.
    shape_orthogonal
        The orthogonal shape of the weights which read the rows.

    Returns
    -------
    The rows on the device, and the row which each element of the
    orthogonal axes reads, as an array of `shape_orthogonal`.
    """
    num_orthogonal = a.ndim - ndim
    a = a[
        tuple(
            slice(0, 1) if a.strides[i] == 0 else slice(None)
            for i in range(num_orthogonal)
        )
    ]
    shape = a.shape[:num_orthogonal]
    num_rows = int(np.prod(shape, dtype=int))
    a = np.ascontiguousarray(a).reshape((num_rows,) + a.shape[num_orthogonal:])
    lookup = np.arange(num_rows).reshape(shape)
    return cuda.to_device(a), np.broadcast_to(lookup, shape_orthogonal)


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


def prefix_sum(counts: Any, num: int) -> tuple[Any, int]:
    """
    Compute the exclusive prefix sum of a device array of counts, on the
    device.

    :mod:`numba` has no scan, so this borrows :func:`torch.cumsum`, writing
    into memory which :mod:`numba` allocated and owns.  The result can then
    be read by a kernel launched after this returns without anything having
    to keep a tensor alive: :mod:`torch` returns memory to its own cache when
    a tensor is dropped and hands it out again without waiting for kernels
    it does not know about, whereas memory :mod:`numba` frees is only
    released once the device is done with it.

    The sum runs on the default stream, which is where :mod:`numba` launches
    the kernels which write `counts` and read the result, so it is ordered
    after the one and before the other whatever stream :mod:`torch` has been
    told to use.

    Parameters
    ----------
    counts
        The counts, on the device.
    num
        The number of counts.

    Returns
    -------
    The ``num + 1`` offsets, as a device array whose last element is the
    total, and that total.
    """
    try:
        # an optional dependency, so it is absent from the environment the
        # type checker runs in
        import torch  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: nocover
        raise ImportError(
            "weights on a device need `torch`, which provides the prefix sum; "
            "install `regridding[cuda]`"
        ) from error

    offset = allocate(num + 1, np.int64)

    with torch.cuda.stream(torch.cuda.default_stream()):
        view = torch.as_tensor(offset, device="cuda")
        view[0] = 0
        torch.cumsum(torch.as_tensor(counts, device="cuda"), dim=0, out=view[1:])
        total = int(view[~0].item())

    return offset, total

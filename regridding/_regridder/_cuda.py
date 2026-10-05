"""
Applying a :class:`regridding.Regridder` on a CUDA device.
"""

from typing import Any
from numba import cuda
from regridding import _cuda

__all__ = [
    "matmul",
]

_warp = 32
"""The number of threads in a warp, which sums one row of the matrix."""


def matmul(
    indptr: Any,
    indices: Any,
    data: Any,
    x: Any,
    y: Any,
) -> None:
    """
    Multiply a matrix in CSR form by a dense matrix on the device, writing
    the product to `y`.

    Each row is summed by one warp: each thread of it sums every 32nd entry,
    in order, and the 32 sums are then added in a fixed tree, so the result
    is the same every time.

    Parameters
    ----------
    indptr
        Where each row starts in `indices` and `data`, on the device.
    indices
        The column of each entry, on the device.
    data
        The value of each entry, on the device.
    x
        The dense matrix, with a row for each column of the sparse one.
    y
        The product, with a row for each row of the sparse one.
    """
    num_rows = y.shape[0]
    blocks = (num_rows * _warp + _cuda.threads - 1) // _cuda.threads
    if blocks:
        _matmul[blocks, _cuda.threads](  # type: ignore[index]
            indptr, indices, data, x, y
        )


# this runs on the device, where `coverage` cannot follow it, so it reports
# the body as missed even when it does the work.  The arrays are device
# arrays, which `numba` ships no type for, so they are annotated as `Any`
@cuda.jit
def _matmul(
    indptr: Any,
    indices: Any,
    data: Any,
    x: Any,
    y: Any,
) -> None:  # pragma: nocover
    """Sum one row of the product for each warp, see :func:`matmul`."""
    thread: int = cuda.grid(1)  # type: ignore[call-arg, assignment]
    row = thread // _warp
    lane = thread % _warp

    num_rows, num_batch = y.shape

    # every thread of a warp has the same row, so a warp leaves together and
    # the shuffles below always have all 32 threads
    if row >= num_rows:
        return

    start = indptr[row]
    stop = indptr[row + 1]

    for t in range(num_batch):
        total = 0.0
        for k in range(start + lane, stop, _warp):
            total += data[k] * x[indices[k], t]
        offset = _warp // 2
        while offset > 0:
            total += cuda.shfl_down_sync(0xFFFFFFFF, total, offset)
            offset //= 2
        if lane == 0:
            y[row, t] = total

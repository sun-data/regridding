"""
The convolution of a set of weights with a kernel, on the CPU.

The algorithm lives in :mod:`._shared`, which :mod:`._cuda` compiles for a
CUDA device from the same source.  What remains here is how the runs are
spread over the threads and where the scratch space comes from.
"""

from typing import Any, Callable
import numpy as np
import numba
from ._shared import (
    num_axis as _num_axis,
    build as _build_shared,
)

__all__ = [
    "convolve_weights_host",
]


def _jit(function: Callable) -> Any:
    """Compile one of the shared kernel bodies for the CPU."""
    return numba.njit(cache=True, inline="always", error_model="numpy")(function)


_starts_run, _box_run, _convolve_run = _build_shared(_jit)

_size_block = 1024
"""
The number of weights each thread is handed at a time.

Most weights do not start a run, so a thread per weight would mostly be
asked whether it has anything to do.  Handing out blocks instead also lets
each thread keep its scratch space from one run to the next rather than
allocating it for every run.
"""


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _convolve_runs(
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    values: np.ndarray,
    kernel: np.ndarray,
    shape_grid: np.ndarray,
    shape_output: np.ndarray,
    shape_kernel: np.ndarray,
    offsets: np.ndarray,
    write: bool,
    counts: np.ndarray,
    offset: np.ndarray,
    result_input: np.ndarray,
    result_output: np.ndarray,
    result_values: np.ndarray,
) -> None:
    """
    Visit every run, either counting what it becomes or writing it.

    Each thread keeps one scratch box, which it grows whenever a run needs a
    larger one, so the scratch space is allocated a handful of times rather
    than once for every run.

    Parameters
    ----------
    indices_input
        The flattened index of the input cell of each weight.
    indices_output
        The flattened index of the output cell of each weight.
    values
        The value of each weight.
    kernel
        The kernel, with one row for each cell of `shape_grid`.
    shape_grid
        The number of rows of the kernel along each resampled axis.
    shape_output
        The number of output cells along each resampled axis.
    shape_kernel
        The length of the kernel along each resampled axis.
    offsets
        The offset of each element of the kernel from its center.
    write
        Whether to write the result into the result arrays at `offset`,
        rather than to count it into `counts`.
    counts
        An output array for the number of weights each run becomes, indexed
        by the weight which starts it.
    offset
        Where each run begins in the result arrays, indexed the same way.
    result_input
        An output array for the flattened index of the input cell.
    result_output
        An output array for the flattened index of the output cell.
    result_values
        An output array for the convolved weights.
    """

    num = indices_input.shape[0]
    num_block = (num + _size_block - 1) // _size_block

    for b in numba.prange(num_block):

        lower = np.empty(_num_axis, dtype=np.int64)
        extent = np.empty(_num_axis, dtype=np.int64)
        coordinates = np.empty(_num_axis, dtype=np.int64)
        strides = np.empty(_num_axis, dtype=np.int64)

        scratch_values = np.empty(0, dtype=np.float64)
        scratch_reached = np.empty(0, dtype=np.bool_)

        stop_block = min((b + 1) * _size_block, num)

        for w in range(b * _size_block, stop_block):

            if not _starts_run(indices_input, indices_output, w, False):
                if not write:
                    counts[w] = 0
                continue

            stop, size = _box_run(
                indices_input,
                indices_output,
                shape_output,
                shape_kernel,
                False,
                w,
                lower,
                extent,
            )

            if size > scratch_values.shape[0]:
                scratch_values = np.empty(2 * size, dtype=np.float64)
                scratch_reached = np.empty(2 * size, dtype=np.bool_)

            count = _convolve_run(
                indices_input,
                indices_output,
                values,
                kernel,
                shape_grid,
                shape_output,
                offsets,
                w,
                stop,
                lower,
                extent,
                coordinates,
                strides,
                scratch_values,
                scratch_reached,
                0,
                write,
                offset[w],
                result_input,
                result_output,
                result_values,
            )

            if not write:
                counts[w] = count


def convolve_weights_host(
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    values: np.ndarray,
    kernel: np.ndarray,
    shape_grid: np.ndarray,
    shape_output: np.ndarray,
    shape_kernel: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convolve one element of a set of weights with a kernel, on the CPU.

    Parameters
    ----------
    indices_input
        The flattened index of the input cell of each weight.  A negative
        index counts from the end of the grid, and is kept as it is.
    indices_output
        The flattened index of the output cell of each weight, which has to
        address a cell of the grid.  A negative index counts from the end of
        the grid.
    values
        The value of each weight.
    kernel
        The kernel, with one row for each cell of `shape_grid` and one
        column for each element of the kernel.
    shape_grid
        The number of rows of the kernel along each resampled axis.
    shape_output
        The number of output cells along each resampled axis.
    shape_kernel
        The length of the kernel along each resampled axis.
    offsets
        The offset of each element of the kernel from its center.

    Returns
    -------
    The convolved ``(indices_input, indices_output, values)``, of the same
    types as the weights given.
    """

    num = values.shape[0]

    size_output = int(np.prod(shape_output))

    # a negative index counts from the end of the grid; the shared kernel
    # splits the output index into one per axis, so it needs the positive one
    if num and indices_output.min() < 0:
        indices_output = np.where(
            indices_output < 0,
            indices_output + size_output,
            indices_output,
        ).astype(indices_output.dtype)

    kernel = np.ascontiguousarray(kernel, dtype=np.float64)

    counts = np.empty(num, dtype=np.int64)
    offset = np.zeros(num + 1, dtype=np.int64)

    empty_input = np.empty(0, dtype=indices_input.dtype)
    empty_output = np.empty(0, dtype=indices_output.dtype)
    empty_values = np.empty(0, dtype=values.dtype)

    _convolve_runs(
        indices_input,
        indices_output,
        values,
        kernel,
        shape_grid,
        shape_output,
        shape_kernel,
        offsets,
        False,
        counts,
        offset,
        empty_input,
        empty_output,
        empty_values,
    )

    np.cumsum(counts, out=offset[1:])
    num_result = int(offset[~0])

    result_input = np.empty(num_result, dtype=indices_input.dtype)
    result_output = np.empty(num_result, dtype=indices_output.dtype)
    result_values = np.empty(num_result, dtype=values.dtype)

    if num_result:
        _convolve_runs(
            indices_input,
            indices_output,
            values,
            kernel,
            shape_grid,
            shape_output,
            shape_kernel,
            offsets,
            True,
            counts,
            offset,
            result_input,
            result_output,
            result_values,
        )

    return result_input, result_output, result_values

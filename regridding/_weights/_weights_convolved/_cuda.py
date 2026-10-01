"""
The convolution of a set of weights with a kernel on a CUDA device, leaving
the result in device memory.

The algorithm lives in :mod:`._shared`, which :mod:`._host` compiles for the
CPU from the same source.  What remains here is only what a device needs and
a CPU does not: mapping threads onto weights, allocating the scratch space in
local memory, the prefix sum, and the device allocations.
"""

import functools
from typing import Any, Callable
import numpy as np
import numba
from numba import cuda
from regridding import _cuda
from ._shared import (
    num_axis as _num_axis,
    build as _build_shared,
)

__all__ = [
    "convolve_weights_cuda",
]


def _jit(function: Callable) -> Any:
    """Compile one of the shared kernel bodies for a CUDA device."""
    return cuda.jit(device=True, inline=True)(function)


@functools.cache
def _build() -> Any:
    """
    Build the kernel.

    Compiling takes a moment, so the kernel is kept.
    """

    starts_run, convolve_run = _build_shared(_jit)

    # the arrays are device arrays, which are annotated as `Any` here as they
    # are elsewhere in this package, since `numba` does not ship types for them
    @cuda.jit
    def convolve_runs(  # pragma: nocover
        indices_input: Any,
        indices_output: Any,
        values: Any,
        kernel: Any,
        varying: bool,
        shape_output: Any,
        shape_kernel: Any,
        write: bool,
        counts: Any,
        offset: Any,
        result_input: Any,
        result_output: Any,
        result_values: Any,
    ) -> None:
        """
        Visit every run, one thread per weight, either counting what it
        becomes or writing it.

        A weight which does not start a run counts as nothing.  Weights
        built on a device keep their empty slots, which carry an index of
        ``-1``, so those are skipped as well.
        """
        w = cuda.grid(1)  # type: ignore[call-arg]
        if w >= indices_input.shape[0]:
            return

        if not starts_run(indices_input, indices_output, w, True):
            if not write:
                counts[w] = 0
            return

        # `numba` annotates the shape of a local array as a `local`, so each
        # of these needs the annotation waived
        lower = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]
        extent = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]

        count = convolve_run(
            indices_input,
            indices_output,
            values,
            kernel,
            varying,
            shape_output,
            shape_kernel,
            True,
            w,
            lower,
            extent,
            write,
            offset[w],
            result_input,
            result_output,
            result_values,
        )

        if not write:
            counts[w] = count

    return convolve_runs


def convolve_weights_cuda(
    indices_input: Any,
    indices_output: Any,
    values: Any,
    kernel: Any,
    shape_output: Any,
    shape_kernel: Any,
    threads: int = _cuda.threads,
) -> tuple[Any, Any, Any]:
    """
    Convolve one element of a set of weights with a kernel on a CUDA device,
    and leave the result there.

    The result has the same form as the weights built by
    :func:`regridding.weights` with ``device="cuda"``, a flat
    ``(indices_input, indices_output, values)`` triple of
    :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`, except that it
    has no empty slots: every run is counted before it is written, so the
    result is exactly as long as it needs to be.

    Parameters
    ----------
    indices_input
        The flattened index of the input cell of each weight, on the
        device.  A negative index marks an empty slot.
    indices_output
        The flattened index of the output cell of each weight, on the
        device.
    values
        The value of each weight, on the device.
    kernel
        The kernel, on the device, with one row for every output cell or a
        single row, and one column for every element of the kernel.
    shape_output
        The number of output cells along each resampled axis, on the
        device.
    shape_kernel
        The length of the kernel along each resampled axis, on the device.
    threads
        The number of threads in each block.
    """

    convolve_runs = _build()

    num = values.shape[0]

    dtype_input = np.dtype(indices_input.dtype)
    dtype_output = np.dtype(indices_output.dtype)
    dtype_values = np.dtype(values.dtype)

    if not num:
        return (
            _cuda.allocate(0, dtype_input),
            _cuda.allocate(0, dtype_output),
            _cuda.allocate(0, dtype_values),
        )

    varying = kernel.shape[0] > 1

    blocks = (num + threads - 1) // threads

    counts = _cuda.allocate(num, np.int64)

    # stand-ins for the arrays the counting pass never touches
    empty_input = _cuda.allocate(1, dtype_input)
    empty_output = _cuda.allocate(1, dtype_output)
    empty_values = _cuda.allocate(1, dtype_values)

    convolve_runs[blocks, threads](  # type: ignore[index]
        indices_input,
        indices_output,
        values,
        kernel,
        varying,
        shape_output,
        shape_kernel,
        False,
        counts,
        counts,
        empty_input,
        empty_output,
        empty_values,
    )

    offset, num_result = _cuda.prefix_sum(counts, num)

    result_input = _cuda.allocate(num_result, dtype_input)
    result_output = _cuda.allocate(num_result, dtype_output)
    result_values = _cuda.allocate(num_result, dtype_values)

    if num_result:
        convolve_runs[blocks, threads](  # type: ignore[index]
            indices_input,
            indices_output,
            values,
            kernel,
            varying,
            shape_output,
            shape_kernel,
            True,
            counts,
            cuda.as_cuda_array(offset),
            result_input,
            result_output,
            result_values,
        )

    return result_input, result_output, result_values

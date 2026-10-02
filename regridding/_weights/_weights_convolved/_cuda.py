"""
The convolution of a set of weights with a kernel on a CUDA device, leaving
the result in device memory.

The algorithm lives in :mod:`._shared`, which :mod:`._host` compiles for the
CPU from the same source.  What remains here is only what a device needs and
a CPU does not: mapping threads onto weights, allocating the small arrays in
local memory, giving every run its own slice of one scratch allocation, the
prefix sums, and the device allocations.
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
def _build() -> tuple[Any, Any]:
    """
    Build the kernels.

    Compiling takes a moment, so the kernels are kept.
    """

    starts_run, box_run, convolve_run = _build_shared(_jit)

    # the arrays the two kernels below take are device arrays, which `numba`
    # ships no type for, so they are annotated as `Any`
    @cuda.jit
    def size_runs(  # pragma: nocover
        indices_input: Any,
        indices_output: Any,
        shape_output: Any,
        shape_kernel: Any,
        size_output: int,
        sizes: Any,
        outside: Any,
    ) -> None:
        """
        Measure the scratch box each run needs, one thread per weight, and
        note any weight whose output cell lies outside the grid.

        A weight which does not start a run needs no box.  Weights built on
        a device keep their empty slots, which carry an index of ``-1``, so
        those are skipped as well.
        """
        w = cuda.grid(1)  # type: ignore[call-arg]
        if w >= indices_input.shape[0]:
            return

        if indices_input[w] >= 0 and indices_output[w] >= size_output:
            outside[0] = 1

        if not starts_run(indices_input, indices_output, w, True):
            sizes[w] = 0
            return

        # `numba` annotates the shape of a local array as a `local`, so each
        # of these needs the annotation waived
        lower = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]
        extent = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]

        _, size = box_run(
            indices_input,
            indices_output,
            shape_output,
            shape_kernel,
            True,
            w,
            lower,
            extent,
        )
        sizes[w] = size

    @cuda.jit
    def convolve_runs(  # pragma: nocover
        indices_input: Any,
        indices_output: Any,
        values: Any,
        kernel: Any,
        shape_grid: Any,
        shape_output: Any,
        shape_kernel: Any,
        offsets: Any,
        offset_scratch: Any,
        scratch_values: Any,
        scratch_reached: Any,
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

        Each run works in its own slice of the scratch arrays, which
        `size_runs` measured, so the runs do not interfere.
        """
        w = cuda.grid(1)  # type: ignore[call-arg]
        if w >= indices_input.shape[0]:
            return

        if not starts_run(indices_input, indices_output, w, True):
            if not write:
                counts[w] = 0
            return

        lower = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]
        extent = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]
        coordinates = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]
        strides = cuda.local.array(_num_axis, numba.int64)  # type: ignore[arg-type]

        stop, _ = box_run(
            indices_input,
            indices_output,
            shape_output,
            shape_kernel,
            True,
            w,
            lower,
            extent,
        )

        count = convolve_run(
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
            offset_scratch[w],
            write,
            offset[w],
            result_input,
            result_output,
            result_values,
        )

        if not write:
            counts[w] = count

    return size_runs, convolve_runs


def convolve_weights_cuda(
    indices_input: Any,
    indices_output: Any,
    values: Any,
    kernel: Any,
    shape_grid: Any,
    shape_output: Any,
    shape_kernel: Any,
    offsets: Any,
    size_output: int,
    outside: Any,
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
        The kernel, on the device, with one row for each cell of
        `shape_grid` and one column for each element of the kernel.
    shape_grid
        The number of rows of the kernel along each resampled axis, on the
        device.
    shape_output
        The number of output cells along each resampled axis, on the
        device.
    shape_kernel
        The length of the kernel along each resampled axis, on the device.
    offsets
        The offset of each element of the kernel from its center, on the
        device.
    size_output
        The number of output cells.
    outside
        A device array of one element, which is set to one if any weight's
        output cell lies outside the grid, and is otherwise left alone.  It
        is checked by the caller rather than here, so that checking it costs
        one transfer for all the elements rather than one each.
    threads
        The number of threads in each block.
    """

    size_runs, convolve_runs = _build()

    num = values.shape[0]

    dtype_input = np.dtype(indices_input.dtype)
    dtype_output = np.dtype(indices_output.dtype)
    dtype_values = np.dtype(values.dtype)

    def empty() -> tuple[Any, Any, Any]:
        return (
            _cuda.allocate(0, dtype_input),
            _cuda.allocate(0, dtype_output),
            _cuda.allocate(0, dtype_values),
        )

    if not num:
        return empty()

    blocks = (num + threads - 1) // threads

    sizes = _cuda.allocate(num, np.int64)

    size_runs[blocks, threads](  # type: ignore[index]
        indices_input,
        indices_output,
        shape_output,
        shape_kernel,
        size_output,
        sizes,
        outside,
    )

    offset_scratch, num_scratch = _cuda.prefix_sum(sizes, num)

    if not num_scratch:
        return empty()

    # every run clears its own slice before using it, so these need not be
    # cleared here
    scratch_values = _cuda.allocate(num_scratch, np.float64)
    scratch_reached = _cuda.allocate(num_scratch, np.bool_)

    counts = _cuda.allocate(num, np.int64)

    # stand-ins for the arrays the counting pass never touches
    empty_input, empty_output, empty_values = (
        _cuda.allocate(1, dtype_input),
        _cuda.allocate(1, dtype_output),
        _cuda.allocate(1, dtype_values),
    )

    convolve_runs[blocks, threads](  # type: ignore[index]
        indices_input,
        indices_output,
        values,
        kernel,
        shape_grid,
        shape_output,
        shape_kernel,
        offsets,
        offset_scratch,
        scratch_values,
        scratch_reached,
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

    convolve_runs[blocks, threads](  # type: ignore[index]
        indices_input,
        indices_output,
        values,
        kernel,
        shape_grid,
        shape_output,
        shape_kernel,
        offsets,
        offset_scratch,
        scratch_values,
        scratch_reached,
        True,
        counts,
        offset,
        result_input,
        result_output,
        result_values,
    )

    return result_input, result_output, result_values

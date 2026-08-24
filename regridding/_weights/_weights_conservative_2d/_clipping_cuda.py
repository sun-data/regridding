"""
The clipping kernel on a CUDA device, leaving the result in device memory.

This is a port of
:func:`~regridding._weights._weights_conservative_2d._clipping.weights_conservative_2d_clipping`.
The algorithm is unchanged: each input cell is clipped against the output
cells its bounding box touches, and writes into its own slice of the
result.  That slice is what makes the port straightforward, since it means
a cell needs nothing from any other cell and the output does not depend on
the order the cells are visited.

The device functions below repeat arithmetic that
:mod:`regridding.geometry` already provides on the host.  A kernel cannot
call a :func:`numba.njit` function, so the two cannot share an
implementation.
"""

import numpy as np
import numba
from numba import cuda

__all__ = [
    "weights_conservative_2d_clipping_cuda",
]

_num_slot = 16
"""
The number of vertex slots reserved for a polygon being clipped.

See :data:`regridding._weights._weights_conservative_2d._clipping._num_slot`
for why this is not eight.
"""


def _build(ftype):
    """
    Build the kernels for a given floating-point type.

    Parameters
    ----------
    ftype
        The :mod:`numba` type of the coordinates and weights.
    """

    @cuda.jit(device=True, inline=True)
    def corners(x, y, index_x, index_y):
        return (
            x[index_x, index_y],
            x[index_x + 1, index_y],
            x[index_x + 1, index_y + 1],
            x[index_x, index_y + 1],
            y[index_x, index_y],
            y[index_x + 1, index_y],
            y[index_x + 1, index_y + 1],
            y[index_x, index_y + 1],
        )

    @cuda.jit(device=True, inline=True)
    def bounds(x1, x2, x3, x4, y1, y2, y3, y4, num_output_x, num_output_y):
        """The block of output cells this cell's bounding box touches."""
        lower_x = min(min(x1, x2), min(x3, x4))
        upper_x = max(max(x1, x2), max(x3, x4))
        lower_y = min(min(y1, y2), min(y3, y4))
        upper_y = max(max(y1, y2), max(y3, y4))

        index_lower_x = int(lower_x)
        if lower_x < 0:
            index_lower_x -= 1
        index_lower_y = int(lower_y)
        if lower_y < 0:
            index_lower_y -= 1
        index_upper_x = int(upper_x) + 1
        index_upper_y = int(upper_y) + 1

        if index_lower_x < 0:
            index_lower_x = 0
        if index_lower_y < 0:
            index_lower_y = 0
        if index_upper_x > num_output_x:
            index_upper_x = num_output_x
        if index_upper_y > num_output_y:
            index_upper_y = num_output_y

        return index_lower_x, index_upper_x, index_lower_y, index_upper_y

    @cuda.jit
    def count_cells(x, y, num_output_x, num_output_y, counts):
        """
        Count the output cells each input cell can touch.

        The prefix sum of this is where each cell writes its result, which
        is what lets the cells run independently.
        """
        index = cuda.grid(1)
        num_y = x.shape[1] - 1
        if index >= (x.shape[0] - 1) * num_y:
            return
        index_x = index // num_y
        index_y = index - index_x * num_y

        x1, x2, x3, x4, y1, y2, y3, y4 = corners(x, y, index_x, index_y)
        lower_x, upper_x, lower_y, upper_y = bounds(
            x1, x2, x3, x4, y1, y2, y3, y4, num_output_x, num_output_y
        )

        span_x = upper_x - lower_x
        span_y = upper_y - lower_y
        if span_x < 0:
            span_x = 0
        if span_y < 0:
            span_y = 0
        counts[index] = span_x * span_y

    @cuda.jit(device=True, inline=True)
    def clip_halfplane(x_in, y_in, num_in, axis, sign, bound, x_out, y_out):
        """Clip a polygon against a half-plane, Sutherland-Hodgman style."""
        num_out = 0
        for k in range(num_in):
            x1 = x_in[k]
            y1 = y_in[k]
            k_next = k + 1
            if k_next == num_in:
                k_next = 0
            x2 = x_in[k_next]
            y2 = y_in[k_next]
            if axis == 0:
                distance_1 = sign * (x1 - bound)
                distance_2 = sign * (x2 - bound)
            else:
                distance_1 = sign * (y1 - bound)
                distance_2 = sign * (y2 - bound)
            inside_1 = distance_1 >= 0
            inside_2 = distance_2 >= 0
            if inside_1:
                x_out[num_out] = x1
                y_out[num_out] = y1
                num_out += 1
            if inside_1 != inside_2:
                denominator = distance_1 - distance_2
                if denominator != 0:
                    t = distance_1 / denominator
                    x_out[num_out] = x1 + t * (x2 - x1)
                    y_out[num_out] = y1 + t * (y2 - y1)
                    num_out += 1
        return num_out

    @cuda.jit(device=True, inline=True)
    def area_signed(x, y, num):
        """The signed area of a polygon, by the shoelace sum."""
        result = ftype(0)
        for k in range(num):
            k_next = k + 1
            if k_next == num:
                k_next = 0
            result += x[k] * y[k_next] - x[k_next] * y[k]
        return ftype(0.5) * result

    @cuda.jit
    def clip_cells(
        x,
        y,
        weights_input,
        num_output_x,
        num_output_y,
        offset,
        indices_input,
        indices_output,
        values,
    ):
        """
        Clip every input cell against the output cells it touches.

        Slots which receive no overlap keep the sentinel index of ``-1``
        they were initialized with, and are dropped by the caller.
        """
        index = cuda.grid(1)
        num_y = x.shape[1] - 1
        if index >= (x.shape[0] - 1) * num_y:
            return
        index_x = index // num_y
        index_y = index - index_x * num_y

        subject_x = cuda.local.array(_num_slot, ftype)
        subject_y = cuda.local.array(_num_slot, ftype)
        clipped_x = cuda.local.array(_num_slot, ftype)
        clipped_y = cuda.local.array(_num_slot, ftype)

        x1, x2, x3, x4, y1, y2, y3, y4 = corners(x, y, index_x, index_y)
        lower_x, upper_x, lower_y, upper_y = bounds(
            x1, x2, x3, x4, y1, y2, y3, y4, num_output_x, num_output_y
        )

        # Shift the cell onto its own block of candidates.  The shoelace
        # sums differences of coordinates, so working at the scale of the
        # block rather than of the whole grid keeps the significant figures
        # that single precision would otherwise lose to cancellation.
        x1 -= lower_x
        x2 -= lower_x
        x3 -= lower_x
        x4 -= lower_x
        y1 -= lower_y
        y2 -= lower_y
        y3 -= lower_y
        y4 -= lower_y

        area_cell = ftype(0.5) * (
            (x1 * y2 - x2 * y1)
            + (x2 * y3 - x3 * y2)
            + (x3 * y4 - x4 * y3)
            + (x4 * y1 - x1 * y4)
        )
        if area_cell == 0:
            return

        weight_cell = weights_input[index_x, index_y] / area_cell

        write = offset[index]
        for cell_x in range(lower_x, upper_x):
            for cell_y in range(lower_y, upper_y):

                subject_x[0] = x1
                subject_x[1] = x2
                subject_x[2] = x3
                subject_x[3] = x4
                subject_y[0] = y1
                subject_y[1] = y2
                subject_y[2] = y3
                subject_y[3] = y4

                num = clip_halfplane(
                    subject_x, subject_y, 4, 0,
                    ftype(1.0), ftype(cell_x - lower_x), clipped_x, clipped_y,
                )
                num = clip_halfplane(
                    clipped_x, clipped_y, num, 0,
                    ftype(-1.0), ftype(cell_x - lower_x + 1), subject_x, subject_y,
                )
                if num < 3:
                    continue
                num = clip_halfplane(
                    subject_x, subject_y, num, 1,
                    ftype(1.0), ftype(cell_y - lower_y), clipped_x, clipped_y,
                )
                if num < 3:
                    continue
                num = clip_halfplane(
                    clipped_x, clipped_y, num, 1,
                    ftype(-1.0), ftype(cell_y - lower_y + 1), subject_x, subject_y,
                )
                if num < 3:
                    continue

                area = area_signed(subject_x, subject_y, num)
                if area == 0:
                    continue

                indices_input[write] = index
                indices_output[write] = cell_x * num_output_y + cell_y
                values[write] = area * weight_cell
                write += 1

    return count_cells, clip_cells


@cuda.jit
def _fill(a, value):
    """Fill a device array, which is cheaper than sending one from the host."""
    i = cuda.grid(1)
    if i < a.size:
        a[i] = value


_kernels = {}


def _prefix_sum(counts, num_cell):
    """
    The exclusive prefix sum of the per-cell counts, on the device.

    :mod:`numba` has no scan, so this borrows :func:`torch.cumsum`, which
    shares the memory rather than copying it.
    """
    try:
        import torch
    except ImportError as error:  # pragma: nocover
        raise ImportError(
            "building weights on a device needs `torch`, which provides the "
            "prefix sum; install `regridding[cuda]`"
        ) from error

    offset = torch.zeros(num_cell + 1, dtype=torch.int64, device="cuda")
    torch.cumsum(torch.as_tensor(counts, device="cuda"), dim=0, out=offset[1:])
    return offset, int(offset[~0].item())


def weights_conservative_2d_clipping_cuda(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    weights_input: None | np.ndarray = None,
    dtype: np.typing.DTypeLike = np.float64,
    threads: int = 128,
) -> tuple:
    """
    Compute 2D conservative weights on a CUDA device, and leave them there.

    The result uses the same convention as the host kernel, a flat
    ``(indices_input, indices_output, values)`` triple, but the three
    arrays are :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`.
    They expose ``__cuda_array_interface__``, so :func:`torch.as_tensor`
    wraps them without copying.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.  May already be on the device, in
        which case they are expected in output-cell units and nothing is
        transferred.
    grid_output
        The vertices of the new grid, which must be a uniform,
        axis-aligned lattice.
    weights_input
        Optional weights applied to the values of the input grid before
        resampling.
    dtype
        The floating-point type of the clipping and of the result.
    threads
        The number of threads in each block.
    """

    ftype = numba.float32 if np.dtype(dtype) == np.float32 else numba.float64
    if ftype not in _kernels:
        _kernels[ftype] = _build(ftype)
    count_cells, clip_cells = _kernels[ftype]

    x_output, y_output = grid_output
    num_output_x = x_output.shape[0] - 1
    num_output_y = y_output.shape[1] - 1

    x_input, y_input = grid_input
    if cuda.is_cuda_array(x_input):
        x, y = x_input, y_input
    else:
        origin_x = float(x_output[0, 0])
        origin_y = float(y_output[0, 0])
        step_x = float(x_output[1, 0] - x_output[0, 0])
        step_y = float(y_output[0, 1] - y_output[0, 0])
        x = cuda.to_device(
            np.ascontiguousarray(
                (np.asarray(x_input, float) - origin_x) / step_x, dtype
            )
        )
        y = cuda.to_device(
            np.ascontiguousarray(
                (np.asarray(y_input, float) - origin_y) / step_y, dtype
            )
        )

    num_x = x.shape[0] - 1
    num_y = x.shape[1] - 1
    num_cell = num_x * num_y
    blocks = (num_cell + threads - 1) // threads

    if weights_input is None:
        factor = cuda.device_array((num_x, num_y), dtype)
        _fill[blocks, threads](factor.reshape(-1), 1)
    elif cuda.is_cuda_array(weights_input):
        factor = weights_input
    else:
        factor = cuda.to_device(np.ascontiguousarray(weights_input, dtype))

    counts = cuda.device_array(num_cell, np.int64)
    count_cells[blocks, threads](x, y, num_output_x, num_output_y, counts)

    offset, num_total = _prefix_sum(counts, num_cell)

    indices_input = cuda.device_array(num_total, np.int64)
    _fill[(num_total + threads - 1) // threads, threads](indices_input, -1)
    indices_output = cuda.device_array(num_total, np.int64)
    values = cuda.device_array(num_total, dtype)

    clip_cells[blocks, threads](
        x,
        y,
        factor,
        num_output_x,
        num_output_y,
        cuda.as_cuda_array(offset),
        indices_input,
        indices_output,
        values,
    )

    return indices_input, indices_output, values

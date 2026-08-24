"""
An alternative 2D conservative weights kernel based on polygon clipping.

:func:`~regridding._weights._weights_conservative_2d.weights_conservative_2d`
sweeps the grid lines of *both* grids and accumulates boundary integrals,
which is fully general but forces a sequential walk along each line and a
pass over the output grid whose cost is independent of how small the input
grid is.

When the output grid is a uniform, axis-aligned lattice, the intersection of
an input cell with an output cell can instead be computed directly by
clipping, and the output grid is never swept.  Each input cell is clipped
against only the output cells its bounding box touches, so the work per cell
is bounded and every cell is independent of every other.

The clipping itself lives in
:mod:`~regridding._weights._weights_conservative_2d._clipping_shared`, whose
source is compiled here for the CPU and in
:mod:`~regridding._weights._weights_conservative_2d._clipping_cuda` for a
CUDA device.
"""

import numpy as np
import numba
import regridding as rg
from ._clipping_shared import num_slot as _num_slot, build as _build_shared

__all__ = [
    "grid_is_uniform_rectilinear",
    "weights_conservative_2d_clipping",
]


def _jit(function):
    """
    Compile one of the shared kernel bodies for the CPU.

    Parameters
    ----------
    function
        The plain Python function to compile.
    """
    return numba.njit(cache=True, inline="always", error_model="numpy")(function)


_num_pair, _clip_cell = _build_shared(_jit, rg.geometry.cross_2d)


def grid_is_uniform_rectilinear(
    grid: tuple[np.ndarray, np.ndarray],
    rtol: float = 1e-10,
) -> bool:
    """
    Test whether a grid of cell vertices is a uniform, axis-aligned lattice.

    This is the condition under which
    :func:`weights_conservative_2d_clipping` may be used in place of
    :func:`~regridding._weights._weights_conservative_2d.weights_conservative_2d`.

    Parameters
    ----------
    grid
        A 2D grid of cell vertices.
    rtol
        The relative tolerance used to decide whether the vertex spacing is
        constant.
    """

    x, y = grid

    if x.ndim != 2:  # pragma: nocover
        return False

    # `x` must vary only along the first axis and `y` only along the second
    if not np.array_equal(x, np.broadcast_to(x[:, :1], x.shape)):
        return False
    if not np.array_equal(y, np.broadcast_to(y[:1, :], y.shape)):
        return False

    step_x = np.diff(x[:, 0])
    step_y = np.diff(y[0, :])

    if step_x.size == 0 or step_y.size == 0:  # pragma: nocover
        return False
    if not np.all(np.isfinite(step_x)) or not np.all(np.isfinite(step_y)):
        return False
    if step_x[0] == 0 or step_y[0] == 0:  # pragma: nocover
        return False

    if not np.allclose(step_x, step_x[0], rtol=rtol, atol=0):
        return False
    if not np.allclose(step_y, step_y[0], rtol=rtol, atol=0):
        return False

    return True


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _count_cells(
    x: np.ndarray,
    y: np.ndarray,
    num_cell_output_x: int,
    num_cell_output_y: int,
    counts: np.ndarray,
) -> None:
    """
    Count the output cells each input cell can touch.

    The prefix sum of this reserves each cell a slice of the result, which is
    what makes the clipping pass independent of the thread schedule.

    Parameters
    ----------
    x
        The :math:`x` coordinates of the input grid's vertices, expressed in
        output-cell units.
    y
        The :math:`y` coordinates of the input grid's vertices, expressed in
        output-cell units.
    num_cell_output_x
        The number of output cells along the first axis.
    num_cell_output_y
        The number of output cells along the second axis.
    counts
        An output array for the per-cell counts.
    """

    num_x = x.shape[0] - 1
    num_y = x.shape[1] - 1

    for index_x in numba.prange(num_x):
        for index_y in range(num_y):
            counts[index_x * num_y + index_y] = _num_pair(
                x,
                y,
                index_x,
                index_y,
                num_cell_output_x,
                num_cell_output_y,
            )


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _clip_cells(
    x: np.ndarray,
    y: np.ndarray,
    weights_input: np.ndarray,
    num_cell_output_x: int,
    num_cell_output_y: int,
    offset: np.ndarray,
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    values: np.ndarray,
) -> None:
    """
    Clip every input cell against the output cells its bounding box touches.

    Each input cell writes into its own slice of the output arrays, given by
    `offset`, so the result does not depend on how the work is scheduled
    across threads.  Slots that receive no overlap keep the sentinel index
    of ``-1`` they were initialized with.

    Parameters
    ----------
    x
        The :math:`x` coordinates of the input grid's vertices, expressed in
        output-cell units.
    y
        The :math:`y` coordinates of the input grid's vertices, expressed in
        output-cell units.
    weights_input
        Weights applied to the values of the input grid before resampling.
    num_cell_output_x
        The number of output cells along the first axis.
    num_cell_output_y
        The number of output cells along the second axis.
    offset
        The index at which each input cell's slice of the output arrays
        begins, with one extra trailing element holding the total.
    indices_input
        An output array for the flattened index of the input cell.
    indices_output
        An output array for the flattened index of the output cell.
    values
        An output array for the weights.
    """

    num_x = x.shape[0] - 1
    num_y = x.shape[1] - 1

    for index_x in numba.prange(num_x):

        subject_x = np.empty(_num_slot)
        subject_y = np.empty(_num_slot)
        clipped_x = np.empty(_num_slot)
        clipped_y = np.empty(_num_slot)

        for index_y in range(num_y):

            index_cell = index_x * num_y + index_y

            _clip_cell(
                x,
                y,
                weights_input,
                num_cell_output_x,
                num_cell_output_y,
                index_x,
                index_y,
                index_cell,
                offset[index_cell],
                subject_x,
                subject_y,
                clipped_x,
                clipped_y,
                indices_input,
                indices_output,
                values,
            )


def weights_conservative_2d_clipping(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    weights_input: None | np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D first-order conservative weights by clipping each input cell
    against the output cells its bounding box touches.

    The result uses the same convention as
    :func:`~regridding._weights._weights_conservative_2d.weights_conservative_2d`:
    a flat ``(indices_input, indices_output, values)`` triple in which each
    value is the fraction of the input cell shared with the output cell,
    optionally scaled by `weights_input`.  Duplicate pairs are summed by
    :func:`regridding.regrid_from_weights`, so the two kernels are
    interchangeable.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
        Both components must have the same 2D shape.
    grid_output
        The vertices of the new grid, which must satisfy
        :func:`grid_is_uniform_rectilinear`.
    weights_input
        Optional weights applied to the values of the input grid before
        resampling.

    Raises
    ------
    ValueError
        If `grid_output` is not a uniform, axis-aligned lattice.

    Notes
    -----
    Signed areas are used throughout, so an input cell wound in the opposite
    sense to its grid is handled correctly, and so is a cell which is not
    convex.  An input cell whose edges cross each other (a "bowtie") is not,
    since Sutherland-Hodgman assumes a simple polygon.

    Unlike the sweep, this kernel does not need the coordinates to be
    perturbed to break degeneracies: coincident vertices and collinear edges
    are handled exactly.
    """

    if not grid_is_uniform_rectilinear(grid_output):
        raise ValueError(
            "`grid_output` must be a uniform, axis-aligned lattice to use the "
            "clipping kernel; use `weights_conservative_2d` instead"
        )

    x_input, y_input = grid_input
    x_output, y_output = grid_output

    num_cell_output_x = x_output.shape[0] - 1
    num_cell_output_y = y_output.shape[1] - 1

    # express the input grid in output-cell units, so that output cell
    # (i, j) is the unit square [i, i + 1] x [j, j + 1].  The weight is a
    # ratio of areas, so this change of scale cancels.
    step_x = x_output[1, 0] - x_output[0, 0]
    step_y = y_output[0, 1] - y_output[0, 0]
    x = (np.ascontiguousarray(x_input, dtype=float) - x_output[0, 0]) / step_x
    y = (np.ascontiguousarray(y_input, dtype=float) - y_output[0, 0]) / step_y

    num_x = x.shape[0] - 1
    num_y = x.shape[1] - 1

    if weights_input is None:
        weights_input = np.ones((num_x, num_y))
    else:
        weights_input = np.ascontiguousarray(
            np.broadcast_to(weights_input, (num_x, num_y)),
            dtype=float,
        )

    # an upper bound on the number of output cells each input cell can
    # touch, which reserves each cell a slice of the result and makes the
    # kernel's output independent of the thread schedule
    counts = np.empty(num_x * num_y, dtype=np.int64)
    _count_cells(x, y, num_cell_output_x, num_cell_output_y, counts)

    offset = np.zeros(num_x * num_y + 1, dtype=np.int64)
    np.cumsum(counts, out=offset[1:])

    num_total = int(offset[~0])

    indices_input = np.full(num_total, -1, dtype=np.int64)
    indices_output = np.zeros(num_total, dtype=np.int64)
    values = np.zeros(num_total, dtype=float)

    if num_total:
        _clip_cells(
            x=x,
            y=y,
            weights_input=weights_input,
            num_cell_output_x=num_cell_output_x,
            num_cell_output_y=num_cell_output_y,
            offset=offset,
            indices_input=indices_input,
            indices_output=indices_output,
            values=values,
        )

    keep = indices_input >= 0

    return indices_input[keep], indices_output[keep], values[keep]

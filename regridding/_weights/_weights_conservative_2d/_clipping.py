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
"""

import numpy as np
import numba
import regridding as rg

__all__ = [
    "grid_is_uniform_rectilinear",
    "weights_conservative_2d_clipping",
]

_num_slot = 12
"""
The number of vertex slots reserved for a polygon being clipped.

Clipping a quadrilateral against the four edges of a cell cannot produce
more than eight vertices, so this is generous.
"""


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


@numba.njit(cache=True, inline="always", error_model="numpy")
def _clip_halfplane(
    x_in: np.ndarray,
    y_in: np.ndarray,
    num_in: int,
    axis: int,
    sign: float,
    bound: float,
    x_out: np.ndarray,
    y_out: np.ndarray,
) -> int:
    """
    Clip a polygon against a half-plane, Sutherland-Hodgman style.

    Vertices are kept where ``sign * (vertex[axis] - bound) >= 0``, and a new
    vertex is emitted wherever an edge crosses the boundary.  The winding
    order of the input is preserved, so the signed area of the result carries
    the orientation of the input.

    Parameters
    ----------
    x_in
        The :math:`x` coordinates of the polygon's vertices.
    y_in
        The :math:`y` coordinates of the polygon's vertices.
    num_in
        The number of valid vertices in the input polygon.
    axis
        The coordinate axis of the half-plane boundary, 0 or 1.
    sign
        The side of the boundary to keep.
    bound
        The boundary coordinate.
    x_out
        An output array for the :math:`x` coordinates of the result.
    y_out
        An output array for the :math:`y` coordinates of the result.
    """

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


@numba.njit(cache=True, inline="always", error_model="numpy")
def _area_signed(
    x: np.ndarray,
    y: np.ndarray,
    num: int,
) -> float:
    """
    Compute the signed area of a polygon.

    Each edge contributes the signed area of the triangle it forms with the
    origin, via :func:`regridding.geometry.area_triangle`, which is the
    shoelace sum.

    Parameters
    ----------
    x
        The :math:`x` coordinates of the polygon's vertices.
    y
        The :math:`y` coordinates of the polygon's vertices.
    num
        The number of valid vertices.
    """

    result = 0.0

    for k in range(num):
        k_next = k + 1
        if k_next == num:
            k_next = 0
        result += rg.geometry.area_triangle(
            (x[k], y[k]),
            (x[k_next], y[k_next]),
        )

    return result


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

            x1 = x[index_x, index_y]
            x2 = x[index_x + 1, index_y]
            x3 = x[index_x + 1, index_y + 1]
            x4 = x[index_x, index_y + 1]

            y1 = y[index_x, index_y]
            y2 = y[index_x + 1, index_y]
            y3 = y[index_x + 1, index_y + 1]
            y4 = y[index_x, index_y + 1]

            area_cell = (
                rg.geometry.area_triangle((x1, y1), (x2, y2))
                + rg.geometry.area_triangle((x2, y2), (x3, y3))
                + rg.geometry.area_triangle((x3, y3), (x4, y4))
                + rg.geometry.area_triangle((x4, y4), (x1, y1))
            )

            if area_cell == 0:
                continue

            weight_cell = weights_input[index_x, index_y] / area_cell

            lower_x = int(np.floor(min(min(x1, x2), min(x3, x4))))
            lower_y = int(np.floor(min(min(y1, y2), min(y3, y4))))
            upper_x = int(np.ceil(max(max(x1, x2), max(x3, x4))))
            upper_y = int(np.ceil(max(max(y1, y2), max(y3, y4))))

            if lower_x < 0:
                lower_x = 0
            if lower_y < 0:
                lower_y = 0
            if upper_x > num_cell_output_x:
                upper_x = num_cell_output_x
            if upper_y > num_cell_output_y:
                upper_y = num_cell_output_y

            index_write = offset[index_cell]

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

                    # the candidate cells come from the bounding box, so the
                    # cell always overlaps the slab being clipped against in
                    # `x` and these two clips cannot empty the polygon; only
                    # the `y` clips below need an early exit
                    num = _clip_halfplane(
                        subject_x, subject_y, 4, 0, +1.0, cell_x, clipped_x, clipped_y
                    )
                    num = _clip_halfplane(
                        clipped_x,
                        clipped_y,
                        num,
                        0,
                        -1.0,
                        cell_x + 1,
                        subject_x,
                        subject_y,
                    )
                    num = _clip_halfplane(
                        subject_x,
                        subject_y,
                        num,
                        1,
                        +1.0,
                        cell_y,
                        clipped_x,
                        clipped_y,
                    )
                    if num < 3:
                        continue
                    num = _clip_halfplane(
                        clipped_x,
                        clipped_y,
                        num,
                        1,
                        -1.0,
                        cell_y + 1,
                        subject_x,
                        subject_y,
                    )
                    if num < 3:
                        continue

                    area = _area_signed(subject_x, subject_y, num)

                    if area == 0:
                        continue

                    indices_input[index_write] = index_cell
                    indices_output[index_write] = cell_x * num_cell_output_y + cell_y
                    values[index_write] = area * weight_cell
                    index_write += 1


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
    sense to its grid is handled correctly.  An input cell whose edges cross
    each other (a "bowtie") is not, since Sutherland-Hodgman assumes a simple
    polygon.

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
    lower_x = np.floor(np.minimum(x[:-1, :-1], x[1:, 1:]))
    lower_x = np.floor(np.minimum(lower_x, np.minimum(x[1:, :-1], x[:-1, 1:])))
    upper_x = np.ceil(np.maximum(x[:-1, :-1], x[1:, 1:]))
    upper_x = np.ceil(np.maximum(upper_x, np.maximum(x[1:, :-1], x[:-1, 1:])))
    lower_y = np.floor(np.minimum(y[:-1, :-1], y[1:, 1:]))
    lower_y = np.floor(np.minimum(lower_y, np.minimum(y[1:, :-1], y[:-1, 1:])))
    upper_y = np.ceil(np.maximum(y[:-1, :-1], y[1:, 1:]))
    upper_y = np.ceil(np.maximum(upper_y, np.maximum(y[1:, :-1], y[:-1, 1:])))

    span_x = np.clip(upper_x, 0, num_cell_output_x) - np.clip(
        lower_x, 0, num_cell_output_x
    )
    span_y = np.clip(upper_y, 0, num_cell_output_y) - np.clip(
        lower_y, 0, num_cell_output_y
    )
    num_pair = np.clip(span_x, 0, None) * np.clip(span_y, 0, None)

    offset = np.zeros(num_x * num_y + 1, dtype=np.int64)
    np.cumsum(num_pair.reshape(-1).astype(np.int64), out=offset[1:])

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

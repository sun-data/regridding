import sys
import math
import numpy as np
import numba
from numba import literal_unroll
import regridding as rg
from . import (
    _arrays,
    _grids,
)
from ._arrays import axis_x, axis_y

axes = (
    axis_x,
    axis_y,
)

__all__ = [
    "weights_conservative_2d",
]


@numba.njit(
    cache=True,
    fastmath=True,
)
def weights_conservative_2d(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    weights_input: None | np.ndarray,
) -> numba.typed.List[tuple[int, int, float]]:
    """
    For each cell of `grid_output`,
    compute the fraction of area shared with each cell of `grid_input`
    and save as a list of weights.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
        Every component must have the same 2D shape.
    grid_output
        The vertices of the new grid.
        Every component must have the same 2D shape.
    weights_input
        Optional weights applied to the values of the input grid before resampling.
    """

    weights_output = numba.typed.List()
    for x in range(0):
        weights_output.append((0, 0, 0.0))

    volume_input = _grids.grid_volume(grid_input)

    for sweep_input in (False, True):
        _sweep_grid(
            grid_input=grid_input,
            grid_output=grid_output,
            volume_input=volume_input,
            weights_input=weights_input,
            weights_output=weights_output,
            sweep_input=sweep_input,
        )

    return weights_output


@numba.njit(
    cache=True,
    fastmath=True,
)
def _sweep_grid(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    weights_input: None | np.ndarray,
    weights_output: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
) -> None:
    """
    Sweep along each of the two axes of either `grid_input` or `grid_output`
    and compute the weights.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
    grid_output
        The vertices of the new grid.
    volume_input
        The area of the input grid which will used to compute the weight.
    weights_input
        Optional weights applied to the values of the input grid before resampling.
    weights_output
        The current list of output weights.
        New weights will be appended to this list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    """

    shape_input = grid_input[0].shape
    shape_output = grid_output[0].shape

    shape_cells_input = _grids.shape_centers(shape_input)
    shape_cells_output = _grids.shape_centers(shape_output)

    grid_sweep, grid_static = _grid_sweep_static(
        grid_input=grid_input,
        grid_output=grid_output,
        sweep_input=sweep_input,
    )

    x_sweep, y_sweep = grid_sweep
    x_static, y_static = grid_static

    bbox_static_lower = (x_static.min(), y_static.min())
    bbox_static_upper = (x_static.max(), y_static.max())

    boundary_static_x, boundary_static_y = _grids.grid_boundary(grid_static)

    for axis in literal_unroll(axes):

        _sweep_along_axis(
            grid_sweep=(
                _arrays.align_axis_right(x_sweep, axis),
                _arrays.align_axis_right(y_sweep, axis),
            ),
            grid_static=grid_static,
            bbox_static_lower=bbox_static_lower,
            bbox_static_upper=bbox_static_upper,
            boundary_static_x=boundary_static_x,
            boundary_static_y=boundary_static_y,
            volume_input=volume_input,
            shape_cells_input=shape_cells_input,
            shape_cells_output=shape_cells_output,
            weights_input=weights_input,
            weights_output=weights_output,
            sweep_input=sweep_input,
            axis_sweep=axis,
        )


@numba.njit(
    cache=True,
    fastmath=True,
    parallel=True,
)
def _sweep_along_axis(
    grid_sweep: tuple[np.ndarray, np.ndarray],
    grid_static: tuple[np.ndarray, np.ndarray],
    bbox_static_lower: tuple[float, float],
    bbox_static_upper: tuple[float, float],
    boundary_static_x: np.ndarray,
    boundary_static_y: np.ndarray,
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int],
    shape_cells_output: tuple[int, int],
    weights_input: None | np.ndarray,
    weights_output: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
    axis_sweep: int,
) -> None:
    """
    Sweep along one logical axis of either `grid_input` or `grid_output`.
    At each edge, compute and save the corresponding weights to the neighboring
    cells.

    Parameters
    ----------
    grid_sweep
        The vertices of the sweep grid.
        The last axis of this grid must be the sweep axis.
    grid_static
        Coordinates of the static grid.
    bbox_static_lower
        The lower-left corner of the static grid's bounding box.
    bbox_static_upper
        The upper-right corner of the static grid's bounding box
    boundary_static_x
        The :math:`x` coordinate of the boundary of the static grid.
    boundary_static_y
        The :math:`y` coordinate of the boundary of the static grid.
    volume_input
        The volume of each cell in the input grid.
    shape_cells_input
        The number of cells along each axis of the input grid.
        This could be calculated from the other arguments,
        but we don't to save computation time.
    shape_cells_output
        The number of cells along each axis of the output grid.
        Like `shape_cells_input`,
        this could be calculated from the other arguments,
        but we don't to save computation time.
    weights_input
        Optional weights applied to the values of the input grid before resampling.
    weights_output
        The current list of weights.
        New weights will be appended to this list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the sweep grid to iterate along.
    """

    x_sweep, y_sweep = grid_sweep
    x_static, y_static = grid_static

    shape_sweep = x_sweep.shape
    shape_static = x_static.shape

    shape_cells_static = _grids.shape_centers(shape_static)

    shape_sweep_x, shape_sweep_y = shape_sweep

    weight_output = numba.typed.List()

    for i in range(shape_sweep_x):
        w = numba.typed.List()
        for _ in range(0):
            w.append((0, 0, 0.0))
        weight_output.append(w)

    for i in numba.prange(shape_sweep_x):

        i = numba.types.int64(i)
        j = 0

        index_sweep = i, j

        index_static = sys.maxsize, sys.maxsize

        sweep_is_outside_static = True

        x_sweep_ij = x_sweep[index_sweep]
        y_sweep_ij = y_sweep[index_sweep]

        point_1 = x_sweep_ij, y_sweep_ij

        bbox_static = (bbox_static_lower, bbox_static_upper)

        if rg.geometry.point_is_inside_box_2d(
            point=point_1,
            box=bbox_static,
        ):
            if rg.geometry.point_is_inside_polygon(
                x=x_sweep_ij,
                y=y_sweep_ij,
                vertices_x=boundary_static_x,
                vertices_y=boundary_static_y,
            ):
                index_static = _grids.index_of_point_brute(
                    point=point_1,
                    grid=grid_static,
                )
                sweep_is_outside_static = False

        while index_sweep[1] < (shape_sweep_y - 1):

            index_sweep_new = index_sweep[0], index_sweep[1] + 1

            point_2 = (
                x_sweep[index_sweep_new],
                y_sweep[index_sweep_new],
            )

            line = point_1, point_2

            if sweep_is_outside_static:

                line, index_sweep, index_static = _step_outside_static(
                    line=line,
                    index_sweep=index_sweep,
                    index_static=index_static,
                    grid_static=grid_static,
                )

                sweep_is_outside_static = index_static[0] == sys.maxsize

            else:

                line, index_sweep, index_static = _step_inside_static(
                    line=line,
                    index_sweep=index_sweep,
                    index_static=index_static,
                    grid_sweep=grid_sweep,
                    grid_static=grid_static,
                    volume_input=volume_input,
                    shape_cells_input=shape_cells_input,
                    shape_cells_output=shape_cells_output,
                    weights_input=weights_input,
                    weights_output=weight_output[i],
                    sweep_input=sweep_input,
                    axis_sweep=axis_sweep,
                )

                if not _arrays.index_in_bounds(
                    index=index_static,
                    shape=shape_cells_static,
                ):
                    break

            point_1 = line[1]

    for i in range(shape_sweep_x):
        weight_output_i = weight_output[i]
        for w in range(len(weight_output_i)):
            weights_output.append(weight_output_i[w])


@numba.njit(
    cache=True,
    fastmath=True,
    # inline="always",
    error_model="numpy",
)
def _step_outside_static(
    line: tuple[
        tuple[float, float],
        tuple[float, float],
    ],
    index_sweep: tuple[int, int],
    index_static: tuple[int, int],
    grid_static: tuple[np.ndarray, np.ndarray],
) -> tuple[
    tuple[
        tuple[float, float],
        tuple[float, float],
    ],
    tuple[int, int],
    tuple[int, int],
]:
    """
    Check if the current line segment crosses the boundary.
    If it does, compute the intersection and return the intersection point
    and the indices of the `grid_static` where the intersection occurs.

    Parameters
    ----------
    line
        The current line segment of the sweep grid.
    index_sweep
        The index of the current vertex in the sweep grid
        This is assumed to be a valid, positive index.
    index_static
        The index of the current cell in the static grid.
        This is assumed to be a valid, positive index.
    grid_static
        The vertices of the static grid.
    """

    p1, p2 = line

    x, y = grid_static

    shape_cells_static = _grids.shape_centers(x.shape)

    found_intercept = False
    t_min = np.inf

    for axis in _arrays.axes:

        x_aligned = _arrays.align_axis_right(x, axis)
        y_aligned = _arrays.align_axis_right(y, axis)

        num_i, num_j = x_aligned.shape

        for direction_face in (-1, 1):

            if direction_face > 0:
                j_edge = ~0
            else:
                j_edge = 0

            for i0 in range(num_i - 1):

                i1 = i0 + 1

                vertex_0 = (x_aligned[i0, j_edge], y_aligned[i0, j_edge])
                vertex_1 = (x_aligned[i1, j_edge], y_aligned[i1, j_edge])

                edge = (vertex_0, vertex_1)

                t, u = rg.geometry.two_line_segment_intersection_parameters(
                    line_1=line,
                    line_2=edge,
                )

                if rg.geometry.two_line_segments_intersect(t, u):

                    if t < t_min:

                        found_intercept = True
                        t_min = t

                        if direction_face > 0:
                            index_static = i0, shape_cells_static[axis] - 1
                        else:
                            index_static = i0, 0

                        if axis == axis_x:
                            m, n = index_static
                            index_static = n, m

                        p2 = rg.geometry.two_line_segment_intersection(
                            line=line,
                            t=t,
                        )

    if not found_intercept:
        i, j = index_sweep
        index_sweep = i, j + 1

    return (p1, p2), index_sweep, index_static


@numba.njit(
    cache=True,
    fastmath=True,
    # inline="always",
    error_model="numpy",
)
def _step_inside_static(
    line: tuple[
        tuple[float, float],
        tuple[float, float],
    ],
    index_sweep: tuple[int, int],
    index_static: tuple[int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray],
    grid_static: tuple[np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int],
    shape_cells_output: tuple[int, int],
    weights_input: None | np.ndarray,
    weights_output: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
    axis_sweep: int,
) -> tuple[
    tuple[
        tuple[float, float],
        tuple[float, float],
    ],
    tuple[int, int],
    tuple[int, int],
]:
    """
    Check if the current line segment crosses any edge of the current cell
    in the static grid.
    If it does, compute the intersection point and indices of the new
    cell in the static grid.
    Otherwise, increment the index of the sweep grid.

    Parameters
    ----------
    line
        The current line segment of the sweep grid.
    index_sweep
        The index of the current vertex in the sweep grid
        This is assumed to be a valid, positive index.
    index_static
        The index of the current cell in the static grid.
        This is assumed to be a valid, positive index.
    grid_sweep
        The vertices of the sweep grid.
        The last axis of this grid must be the sweep axis.
    grid_static
        The vertices of the static grid.
    volume_input
        The volume of each cell in the input grid.
    shape_cells_input
        The number of cells along each axis of the input grid.
        This could be calculated from the other arguments,
        but we don't to save computation time.
    shape_cells_output
        The number of cells along each axis of the output grid.
        Like `shape_cells_input`,
        this could be calculated from the other arguments,
        but we don't to save computation time.
    weights_input
        Optional weights applied to the values of the input grid before resampling.
    weights_output
        The current list of weights.
        New weights will be appended to this list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the sweep grid to iterate along.
    """

    p1, p2 = line

    x_vertices, y_vertices = _grids.cell_boundary(
        index=index_static,
        grid=grid_static,
    )

    for v in range(len(x_vertices)):

        x0 = x_vertices[v - 1]
        y0 = y_vertices[v - 1]

        x1 = x_vertices[v]
        y1 = y_vertices[v]

        edge = (
            (x0, y0),
            (x1, y1),
        )

        t, u = rg.geometry.two_line_segment_intersection_parameters(
            line_1=line,
            line_2=edge,
        )

        if rg.geometry.two_line_segments_intersect(t, u):

            if t > 1e-6:

                p2 = rg.geometry.two_line_segment_intersection(
                    line=line,
                    t=t,
                )

                index_static_new = rg.math.sum_2d(
                    a=index_static,
                    b=_grids.cell_normals[v],
                )

                index_sweep_new = index_sweep

                break

    else:
        i, j = index_sweep
        index_sweep_new = i, j + 1
        index_static_new = index_static

    line = p1, p2

    _calc_and_save_weights(
        line=line,
        index_sweep=index_sweep,
        index_static=index_static,
        grid_sweep=grid_sweep,
        volume_input=volume_input,
        shape_cells_input=shape_cells_input,
        shape_cells_output=shape_cells_output,
        weights_input=weights_input,
        weights_output=weights_output,
        sweep_input=sweep_input,
        axis_sweep=axis_sweep,
    )

    return line, index_sweep_new, index_static_new


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def _calc_and_save_weights(
    line: tuple[
        tuple[float, float],
        tuple[float, float],
    ],
    index_sweep: tuple[int, int],
    index_static: tuple[int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int],
    shape_cells_output: tuple[int, int],
    weights_input: None | np.ndarray,
    weights_output: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
    axis_sweep: int
):

    x_sweep, y_sweep = grid_sweep

    shape_sweep = x_sweep.shape

    i, j = index_sweep

    i_left = i - 1
    i_right = i

    p1, p2 = line

    area_sweep = rg.geometry.area_triangle(p1, p2)

    if axis_sweep == 0:
        area_sweep = -area_sweep

    if i_left >= 0:

        index_sweep_left = i_left, j

        index_input_left, index_output_left = _index_input_output(
            index_sweep=index_sweep_left,
            index_static=index_static,
            sweep_input=sweep_input,
            axis_sweep=axis_sweep,
        )

        area_input_left = volume_input[index_input_left]

        weight_left = area_sweep / area_input_left

        if weights_input is not None:
            weight_left *= weights_input[index_input_left]

        index_input_left = _arrays.index_flat(index_input_left, shape_cells_input)
        index_output_left = _arrays.index_flat(index_output_left, shape_cells_output)

        weights_output.append((index_input_left, index_output_left, weight_left))

    if i_right < (shape_sweep[0] - 1):

        index_sweep_right = i_right, j

        index_input_right, index_output_right = _index_input_output(
            index_sweep=index_sweep_right,
            index_static=index_static,
            sweep_input=sweep_input,
            axis_sweep=axis_sweep,
        )

        area_input_right = volume_input[index_input_right]

        weight_right = -area_sweep / area_input_right

        if weights_input is not None:
            weight_right *= weights_input[index_input_right]

        index_input_right = _arrays.index_flat(index_input_right, shape_cells_input)
        index_output_right = _arrays.index_flat(index_output_right, shape_cells_output)

        weights_output.append((index_input_right, index_output_right, weight_right))


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def _grid_sweep_static(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    sweep_input: bool,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Given the input and output grids,
    prepare the sweep and static grids.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
    grid_output
        The vertices of the new grid.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    """
    if sweep_input:
        grid_sweep = grid_input
        grid_static = grid_output
    else:
        grid_sweep = grid_output
        grid_static = grid_input

    return grid_sweep, grid_static


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def _index_input_output(
    index_sweep: tuple[int, int],
    index_static: tuple[int, int],
    sweep_input: bool,
    axis_sweep: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Convert indices on the sweep and static grids into indices on the
    input/output grids.

    Parameters
    ----------
    index_sweep
        The index on the sweep grid to convert.
    index_static
        The index on the static grid to convert.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the `grid_sweep` we're iterating along.
    """

    i_sweep, j_sweep = index_sweep

    if axis_sweep == axis_x:
        index_sweep = j_sweep, i_sweep

    if sweep_input:
        index_input = index_sweep
        index_output = index_static
    else:
        index_input = index_static
        index_output = index_sweep

    return index_input, index_output

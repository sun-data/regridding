import sys
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
    for x in range(0):  # pragma: nocover
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
        for _ in range(0):  # pragma: nocover
            w.append((0, 0, 0.0))
        weight_output.append(w)

    for index_sweep_x in numba.prange(shape_sweep_x):

        index_sweep_x = numba.types.int64(index_sweep_x)
        index_sweep_y = 0

        index_static_x = sys.maxsize
        index_static_y = sys.maxsize

        sweep_is_outside_static = True
        index_edge_last = sys.maxsize

        x1 = x_sweep[index_sweep_x, index_sweep_y]
        y1 = y_sweep[index_sweep_x, index_sweep_y]

        point_1 = x1, y1

        bbox_static = (bbox_static_lower, bbox_static_upper)

        if rg.geometry.point_is_inside_box_2d(
            point=point_1,
            box=bbox_static,
        ):
            if rg.geometry.point_is_inside_polygon(
                x=x1,
                y=y1,
                vertices_x=boundary_static_x,
                vertices_y=boundary_static_y,
            ):
                index_static_x, index_static_y = _grids.index_of_point_brute(
                    point=point_1,
                    grid=grid_static,
                )
                sweep_is_outside_static = False

        while index_sweep_y < (shape_sweep_y - 1):

            x2 = x_sweep[index_sweep_x, index_sweep_y + 1]
            y2 = y_sweep[index_sweep_x, index_sweep_y + 1]

            if sweep_is_outside_static:

                (
                    x1,
                    y1,
                    x2,
                    y2,
                    index_sweep_y,
                    index_static_x,
                    index_static_y,
                    index_edge_last,
                ) = _step_outside_static(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    index_sweep_y=index_sweep_y,
                    index_static_x=index_static_x,
                    index_static_y=index_static_y,
                    index_edge_last=index_edge_last,
                    grid_static=grid_static,
                )

                sweep_is_outside_static = index_static_x == sys.maxsize

            else:

                (
                    x1,
                    y1,
                    x2,
                    y2,
                    index_sweep_y,
                    index_static_x,
                    index_static_y,
                    index_edge_last,
                ) = _step_inside_static(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    index_sweep_x=index_sweep_x,
                    index_sweep_y=index_sweep_y,
                    index_static_x=index_static_x,
                    index_static_y=index_static_y,
                    index_edge_last=index_edge_last,
                    grid_sweep=grid_sweep,
                    grid_static=grid_static,
                    volume_input=volume_input,
                    shape_cells_input=shape_cells_input,
                    shape_cells_output=shape_cells_output,
                    weights_input=weights_input,
                    weights_output=weight_output[index_sweep_x],
                    sweep_input=sweep_input,
                    axis_sweep=axis_sweep,
                )

                if not _arrays.index_in_bounds(
                    index=(index_static_x, index_static_y),
                    shape=shape_cells_static,
                ):
                    break

            x1 = x2
            y1 = y2

    for i in range(shape_sweep_x):
        weight_output_i = weight_output[i]
        weights_output.extend(weight_output_i)


@numba.njit(
    cache=True,
    fastmath=True,
    # inline="always",
    error_model="numpy",
)
def _step_outside_static(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    index_sweep_y: int,
    index_static_x: int,
    index_static_y: int,
    index_edge_last: int,
    grid_static: tuple[np.ndarray, np.ndarray],
) -> tuple[
    float,
    float,
    float,
    float,
    # int,
    int,
    int,
    int,
    int,
]:
    """
    Check if the current line segment crosses the boundary.
    If it does, compute the intersection and return the intersection point
    and the indices of the `grid_static` where the intersection occurs.

    Parameters
    ----------
    x1
        The :math:`x`-coordinate of the line segment's starting point.
    y1
        The :math:`y`-coordinate of the line segment's starting point.
    x2
        The :math:`x`-coordinate of the line segment's ending point.
    y2
        The :math:`y`-coordinate of the line segment's ending point.
    index_sweep_y
        The vertical index of the current vertex in the sweep grid.
    index_static_x
        The horizontal index of the current vertex in the static grid.
    index_static_y
        The vertical index of the current vertex in the static grid.
    index_edge_last
        The 1D index corresponding to the edge of the static grid crossed in
        the last step.
    grid_static
        The vertices of the static grid.
    """

    p1 = (x1, y1)
    p2 = (x2, y2)

    x, y = grid_static

    shape_cells_static = _grids.shape_centers(x.shape)

    found_intercept = False
    t_min = np.inf

    for axis in axes:

        x_aligned = _arrays.align_axis_right(x, axis)
        y_aligned = _arrays.align_axis_right(y, axis)

        num_i, num_j = x_aligned.shape

        for direction_face in (0, 1):

            if direction_face > 0:
                j_edge = ~0
            else:
                j_edge = 0

            for i0 in range(num_i - 1):

                i1 = i0 + 1

                x3 = x_aligned[i0, j_edge]
                y3 = y_aligned[i0, j_edge]
                x4 = x_aligned[i1, j_edge]
                y4 = y_aligned[i1, j_edge]

                q1 = (x3, y3)
                q2 = (x4, y4)

                t, u = rg.geometry.two_line_segment_intersection_parameters(
                    p1=p1,
                    p2=p2,
                    q1=q1,
                    q2=q2,
                )

                if rg.geometry.two_line_segments_intersect(t, u):

                    if t < t_min:

                        found_intercept = True
                        t_min = t

                        m = i0
                        if direction_face > 0:
                            n = shape_cells_static[axis] - 1
                        else:
                            n = 0

                        if axis == axis_x:
                            index_static_x = n
                            index_static_y = m
                        else:
                            index_static_x = m
                            index_static_y = n

                        x2, y2 = rg.geometry.two_line_segment_intersection(
                            p1=p1,
                            p2=p2,
                            t=t,
                        )

                        index_edge_last = _arrays.index_flat(
                            index=(direction_face, axis),
                            shape=(2, 2),
                        )

    if not found_intercept:
        index_sweep_y = index_sweep_y + 1

    return (
        x1,
        y1,
        x2,
        y2,
        index_sweep_y,
        index_static_x,
        index_static_y,
        index_edge_last,
    )


@numba.njit(
    cache=True,
    fastmath=True,
    # inline="always",
    error_model="numpy",
)
def _step_inside_static(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    index_sweep_x: int,
    index_sweep_y: int,
    index_static_x: int,
    index_static_y: int,
    index_edge_last: int,
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
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
]:
    """
    Check if the current line segment crosses any edge of the current cell
    in the static grid.
    If it does, compute the intersection point and indices of the new
    cell in the static grid.
    Otherwise, increment the index of the sweep grid.

    Parameters
    ----------
    x1
        The :math:`x`-coordinate of the line segment's starting point.
    y1
        The :math:`y`-coordinate of the line segment's starting point.
    x2
        The :math:`x`-coordinate of the line segment's ending point.
    y2
        The :math:`y`-coordinate of the line segment's ending point.
    index_sweep_x
        The horizontal index of the current vertex in the sweep grid.
    index_sweep_y
        The vertical index of the current vertex in the sweep grid.
    index_static_x
        The horizontal index of the current vertex in the static grid.
    index_static_y
        The vertical index of the current vertex in the static grid.
    index_edge_last
        The 1D index corresponding to the edge of the static grid crossed in
        the last step.
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
    p1 = (x1, y1)
    p2 = (x2, y2)

    x, y = grid_static

    for v in range(len(_grids.indices_cell_vertex)):

        if v == index_edge_last:
            continue

        i3_vertex, j3_vertex = _grids.indices_cell_vertex[v - 1]
        i4_vertex, j4_vertex = _grids.indices_cell_vertex[v]

        i3_vertex = i3_vertex + index_static_x
        i4_vertex = i4_vertex + index_static_x

        j3_vertex = j3_vertex + index_static_y
        j4_vertex = j4_vertex + index_static_y

        x3 = x[i3_vertex, j3_vertex]
        y3 = y[i3_vertex, j3_vertex]

        x4 = x[i4_vertex, j4_vertex]
        y4 = y[i4_vertex, j4_vertex]

        q1 = (x3, y3)
        q2 = (x4, y4)

        t, u = rg.geometry.two_line_segment_intersection_parameters(
            p1=p1,
            p2=p2,
            q1=q1,
            q2=q2,
        )

        if rg.geometry.two_line_segments_intersect(t, u):

            x2, y2 = rg.geometry.two_line_segment_intersection(
                p1=p1,
                p2=p2,
                t=t,
            )

            nx, ny = _grids.cell_normals[v]

            _index_sweep_y = index_sweep_y
            _index_static_x = index_static_x + nx
            _index_static_y = index_static_y + ny
            index_edge_last = (v + 2) % 4

            break

    else:
        _index_sweep_y = index_sweep_y + 1
        _index_static_x = index_static_x
        _index_static_y = index_static_y
        index_edge_last = sys.maxsize

    _calc_and_save_weights(
        p1=(x1, y1),
        p2=(x2, y2),
        index_sweep=(index_sweep_x, index_sweep_y),
        index_static=(index_static_x, index_static_y),
        grid_sweep=grid_sweep,
        volume_input=volume_input,
        shape_cells_input=shape_cells_input,
        shape_cells_output=shape_cells_output,
        weights_input=weights_input,
        weights_output=weights_output,
        sweep_input=sweep_input,
        axis_sweep=axis_sweep,
    )

    return (
        x1,
        y1,
        x2,
        y2,
        _index_sweep_y,
        _index_static_x,
        _index_static_y,
        index_edge_last,
    )


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def _calc_and_save_weights(
    p1: tuple[float, float],
    p2: tuple[float, float],
    index_sweep: tuple[int, int],
    index_static: tuple[int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int],
    shape_cells_output: tuple[int, int],
    weights_input: None | np.ndarray,
    weights_output: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
    axis_sweep: int,
):

    x_sweep, y_sweep = grid_sweep

    shape_sweep = x_sweep.shape

    i, j = index_sweep

    i_left = i - 1
    i_right = i

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

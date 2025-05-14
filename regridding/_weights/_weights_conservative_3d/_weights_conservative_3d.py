import sys
import numpy as np
import numba
from numba import literal_unroll
import regridding as rg
from . import (
    _arrays,
    _grids,
    _intercepts,
)
from ._arrays import axis_x, axis_y, axis_z

axes = (
    (axis_x, ),
    (axis_y, ),
    (axis_z, ),
    (axis_x, axis_y),
    (axis_y, axis_z),
    (axis_z, axis_x),
)

__all__ = [
    "weights_conservative_3d",
]


@numba.njit(cache=True)
def weights_conservative_3d(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> numba.typed.List[tuple[int, int, float]]:
    """
    For each cell of `grid_output`,
    compute the fraction of volume shared with each cell of `grid_input`
    and save as a list of weights.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
        Every component must have the same 3D shape.
    grid_output
        The vertices of the new grid.
        Every component must have the same 3D shape.
    """

    x_input, y_input, z_input = grid_input
    x_output, y_output, z_output = grid_output

    shape_input = x_input.shape
    shape_output = x_output.shape

    weights = numba.typed.List()
    for x in range(0):
        weights.append((0, 0, 0.0))

    volume_input = _grids.grid_volume(grid_input)


    intercepts = _intercepts.empty(shape_input, shape_output)

    for sweep_input in (False, True):
        _sweep_grid(
            grid_input=grid_input,
            grid_output=grid_output,
            volume_input=volume_input,
            weights=weights,
            intercepts=intercepts,
            sweep_input=sweep_input,
        )

    _intercepts.sweep(
        intercepts=intercepts,
        weights=weights,
        grid_input=grid_input,
        grid_output=grid_output,
        volume_input=volume_input,
    )

    return weights


@numba.njit(cache=True)
def _sweep_grid(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    weights: numba.typed.List[tuple[int, int, float]],
    intercepts: numba.typed.List,
    sweep_input: bool,
) -> None:
    """
    Sweep along each of the three axes and each of the three diagonals
    of either `grid_input` or `grid_output` and compute the weights and
    intercepts.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
    grid_output
        The vertices of the new grid.
    volume_input
        The area of the input grid which will used to compute the weight.
    weights
        The current list of weights.
        New weights will be appended to this list.
    intercepts
        A sorted list of intercepts to be traversed later.
        As new intercepts are found, they are inserted into the list.
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

    x_sweep, y_sweep, z_sweep = grid_sweep
    x_static, y_static, z_static = grid_static

    bbox_static = (
        (x_static.min(), y_static.min(), z_static.min()),
        (x_static.max(), y_static.max(), z_static.max()),
    )

    _, boundary_static = _grids.grid_boundary(grid_static)

    for axis in axes:
    # for axis in literal_unroll(axes):

        axis_last = axis[~0]

        _sweep_along_axis(
            grid_sweep=(
                _arrays.align_axis_right(x_sweep, axis_last),
                _arrays.align_axis_right(y_sweep, axis_last),
                _arrays.align_axis_right(z_sweep, axis_last),
            ),
            grid_static=grid_static,
            bbox_static=bbox_static,
            boundary_static=boundary_static,
            volume_input=volume_input,
            shape_cells_input=shape_cells_input,
            shape_cells_output=shape_cells_output,
            weights=weights,
            intercepts=intercepts,
            sweep_input=sweep_input,
            axis_sweep=axis,
        )


@numba.njit(cache=True, parallel=True)
def _sweep_along_axis(
    grid_sweep: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox_static: tuple[tuple[float, float, float], tuple[float, float, float]],
    boundary_static: numba.typed.List[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    weights: numba.typed.List[tuple[int, int, float]],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
) -> None:
    """
    Sweep along one logical axis of either `grid_input` or `grid_output`.
    At each vertex, compute and save the weights and intercepts.

    Parameters
    ----------
    grid_sweep
        The vertices of the sweep grid.
        The last axis of this grid must be the sweep axis.
    grid_static
        Coordinates of the static grid.
    bbox_static
        Two points defining the bounding box of the static grid.
    boundary_static
        A sequence of triangles defining the outer surface of the static grid.
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
    weights
        The current list of weights.
        New weights will be appended to this list.
    intercepts
        A sorted list of intercepts to be traversed later.
        As new intercepts are found, they are inserted into the list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the sweep grid to iterate along.
    """

    grid_sweep, grid_static = _grid_sweep_static(
        grid_input=grid_input,
        grid_output=grid_output,
        sweep_input=sweep_input,
        axis_sweep=axis_sweep,
    )

    x_sweep, y_sweep, z_sweep = grid_sweep
    x_static, y_static, z_static = grid_static

    shape_sweep = x_sweep.shape
    shape_static = x_static.shape

    shape_cells_sweep = _grids.shape_centers(shape_sweep)
    shape_cells_static = _grids.shape_centers(shape_static)

    shape_sweep_x, shape_sweep_y, shape_sweep_z = shape_sweep
    shape_static_x, shape_static_y, shape_static_z = shape_static

    weight = numba.typed.List()

    for i in range(shape_sweep_x):
        weight_i = numba.typed.List()
        for _ in range(0):
            weight_i.append((0, 0, 0.))
        weight.append(weight_i)

    if len(axis_sweep) == 2:
        d_start = -(shape_sweep_z - 2)
        d_end = shape_sweep_y - 1
        step_index = (0, 1, 1)
    else:
        d_start = 0
        d_end = shape_sweep_y
        step_index = (0, 0, 1)

    j_end = d_end
    k_end = shape_cells_sweep[2]

    for i in range(shape_sweep_x):

        i = numba.types.int64(i)

        for d in range(d_start, d_end):

            if d < 0:
                j = 0
                k = -d
            else:
                j = d
                k = 0

            index_sweep = i, j, k

            index_static = sys.maxsize, sys.maxsize, sys.maxsize

            sweep_is_outside_static = True

            point_1 = (
                x_sweep[index_sweep],
                y_sweep[index_sweep],
                z_sweep[index_sweep],
            )

            if rg.geometry.point_is_inside_box_3d(
                point=point_1,
                box=bbox_static,
            ):
                if rg.geometry.point_is_inside_polyhedron(
                    point=point_1,
                    polyhedron=boundary_static,
                ):
                    index_static = _grids.index_of_point_brute(
                        point=point_1,
                        grid=grid_static,
                    )
                    sweep_is_outside_static = False

            while (index_sweep[1] < j_end) and (index_sweep[2] < k_end):

                index_sweep_new = rg.math.sum_3d(index_sweep, step_index)

                point_2 = (
                    x_sweep[index_sweep_new],
                    y_sweep[index_sweep_new],
                    z_sweep[index_sweep_new],
                )

                line = point_1, point_2
                if sweep_is_outside_static:

                    line, index_sweep, index_static = _step_outside_static(
                        line=line,
                        index_sweep=index_sweep,
                        index_static=index_static,
                        step_index=step_index,
                        grid_static=grid_static,
                        bbox_static=bbox_static,
                        shape_cells_input=shape_cells_input,
                        shape_cells_output=shape_cells_output,
                        intercepts=intercepts,
                        sweep_input=sweep_input,
                        axis_sweep=axis_sweep,
                    )

                    sweep_is_outside_static = index_static[0] == sys.maxsize

                else:

                    line, index_sweep, index_static = _step_inside_static(
                        line=line,
                        index_sweep=index_sweep,
                        index_static=index_static,
                        step_index=step_index,
                        grid_sweep=grid_sweep,
                        grid_static=grid_static,
                        volume_input=volume_input,
                        shape_cells_input=shape_cells_input,
                        shape_cells_output=shape_cells_output,
                        weights=weights,
                        intercepts=intercepts,
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
        weight_i = weight[i]
        for w in range(len(weight_i)):
            weights.append(weight_i[w])


@numba.njit(cache=True)
def _step_outside_static(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    step_index: tuple[int, int, int],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox_static: tuple[tuple[float, float, float], tuple[float, float, float]],
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
) -> tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    tuple[int, int, int],
    tuple[int, int, int],
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
    step_index
        A step size in index space on the sweep grid.
    grid_static
        The vertices of the static grid.
    bbox_static
        Two points defining the bounding box of the static grid.
    shape_cells_input
        The number of cells along each axis of the input grid.
        This could be calculated from the other arguments,
        but we don't to save computation time.
    shape_cells_output
        The number of cells along each axis of the output grid.
        Like `shape_cells_input`,
        this could be calculated from the other arguments,
        but we don't to save computation time.
    intercepts
        A sorted list of intercepts to be traversed later.
        As new intercepts are found, they are inserted into the list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the sweep grid to iterate along.
    """

    p1, p2 = line

    x, y, z = grid_static

    shape_cells_static = _grids.shape_centers(x.shape)

    if rg.geometry.point_is_inside_box_3d(p2, bbox_static):

        for axis in _arrays.axes:

            x_axis = _arrays.align_axis_right(x, axis)
            y_axis = _arrays.align_axis_right(y, axis)
            z_axis = _arrays.align_axis_right(z, axis)

            for direction_face in (-1, 1):

                if direction_face > 0:
                    k_face = ~0
                else:
                    k_face = 0

                x_face = x_axis[..., k_face]
                y_face = y_axis[..., k_face]
                z_face = z_axis[..., k_face]

                shape_face = x_face.shape

                num_i, num_j = shape_face

                for i0 in range(num_i - 1):
                    for j0 in range(num_j - 1):

                        i1 = i0 + 1
                        j1 = j0 + 1

                        for t in (-1, 1):

                            if t > 0:
                                ind = (
                                    (i0, j0),
                                    (i1, j0),
                                    (i1, j1),
                                )
                            else:
                                ind = (
                                    (i1, j1),
                                    (i0, j1),
                                    (i0, j0),
                                )

                            if direction_face < 0:
                                ind = ind[::-1]

                            v0, v1, v2 = ind

                            triangle = (
                                (x_face[v0], y_face[v0], z_face[v0]),
                                (x_face[v1], y_face[v1], z_face[v1]),
                                (x_face[v2], y_face[v2], z_face[v2]),
                            )

                            tuv = rg.geometry.line_triangle_intersection_parameters(
                                line=line,
                                triangle=triangle
                            )

                            if rg.geometry.line_intersects_triangle(tuv):

                                if direction_face > 0:
                                    index_static = i0, j0, shape_cells_static[axis] - 1
                                else:
                                    index_static = i0, j0, 0

                                i, j, k = index_static

                                if axis == axis_x:
                                    index_static = k, i, j
                                elif axis == axis_y:
                                    index_static = j, k, i

                                normal = [1 if ax == axis else 0 for ax in _arrays.axes]
                                normal = rg.math.multiply_3d(direction_face, normal)

                                _, p2 = _calc_and_save_intercept(
                                    line=line,
                                    tuv=tuv,
                                    index_sweep=index_sweep,
                                    index_static=index_static,
                                    shape_cells_input=shape_cells_input,
                                    shape_cells_output=shape_cells_output,
                                    intercepts=intercepts,
                                    sweep_input=sweep_input,
                                    axis_sweep=axis_sweep,
                                    axis_static=axis,
                                    normal_static=normal,
                                )

                                return (p1, p2), index_sweep, index_static

    index_sweep = rg.math.sum_3d(index_sweep, step_index)

    return (p1, p2), index_sweep, index_static


@numba.njit(cache=True)
def _step_inside_static(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    step_index: tuple[int, int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    weights: numba.typed.List[tuple[int, int, float]],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
) -> tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    """
    Check if the current line segment crosses any face of the current cell
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
        The current index in the static grid.
        This is assumed to be a valid, positive index.
    step_index
        A step size in index space on the sweep grid.
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
    weights
        The current list of weights.
        New weights will be appended to this list.
    intercepts
        A sorted list of intercepts to be traversed later.
        As new intercepts are found, they are inserted into the list.
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis_sweep
        The logical axis of the sweep grid to iterate along.
    """

    p1, p2 = line

    polyhedron_cell = _grids.cell_boundary(
        index=index_static,
        grid=grid_static,
    )

    for t in range(len(polyhedron_cell)):

        triangle = polyhedron_cell[t]

        tuv = rg.geometry.line_triangle_intersection_parameters(
            line=line,
            triangle=triangle,
        )

        if rg.geometry.line_intersects_triangle(tuv):

            if tuv[0] > 1e-8:

                index_sweep_new = index_sweep

                index_static_new, p2 = _calc_and_save_intercept(
                    line=line,
                    tuv=tuv,
                    index_sweep=index_sweep,
                    index_static=index_static,
                    shape_cells_input=shape_cells_input,
                    shape_cells_output=shape_cells_output,
                    intercepts=intercepts,
                    sweep_input=sweep_input,
                    axis_sweep=axis_sweep,
                    axis_static=_grids.cell_axes[t],
                    normal_static=_grids.cell_normals[t],
                )

                break

    else:
        index_sweep_new = rg.math.sum_3d(index_sweep, step_index)
        index_static_new = index_static

    line = p1, p2

    if len(axis_sweep) == 1:
        _calc_and_save_weights(
            line=line,
            index_sweep=index_sweep,
            index_static=index_static,
            grid_sweep=grid_sweep,
            volume_input=volume_input,
            shape_cells_input=shape_cells_input,
            shape_cells_output=shape_cells_output,
            weights=weights,
            sweep_input=sweep_input,
            axis_sweep=axis_sweep,
        )

    return line, index_sweep_new, index_static_new


@numba.njit(cache=True)
def _calc_and_save_intercept(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    tuv: tuple[float, float, float],
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
    axis_static: int,
    normal_static: tuple[int, int, int],
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
]:

    intercept = rg.geometry.line_triangle_intersection(
        line=line,
        tuv=tuv,
    )

    shape_input = [n + 1 for n in shape_cells_input]
    shape_output = [n + 1 for n in shape_cells_output]

    index_static_new = rg.math.sum_3d(index_static, normal_static)
    # print(f"{index_static=}")
    # print(f"{index_static_new=}")

    index_static_intercept = index_static
    if index_static_new[axis_static] > index_static[axis_static]:
        index_static_intercept = index_static_new

    axes_orthogonal = [ax for ax in _arrays.axes if ax not in axis_sweep]

    for axis_orthogonal in axes_orthogonal:

        if sweep_input:
            axis_input = axis_orthogonal
            axis_output = axis_static
        else:
            axis_input = axis_static
            axis_output = axis_orthogonal

        index_input, index_output = _index_input_output(
            index_sweep=index_sweep,
            index_static=index_static_intercept,
            sweep_input=sweep_input,
            axis_sweep=axis_sweep,
        )

        i_input = index_input[axis_input]
        i_output = index_output[axis_output]

        _intercepts.insert_intercept(
            intercepts=intercepts[axis_input][axis_output][i_input][i_output],
            intercept_new=(
                _arrays.index_flat(index_input, shape_input),
                _arrays.index_flat(index_output, shape_output),
                intercept,
            ),
        )

    return index_static_new, intercept


@numba.njit(cache=True)
def _calc_and_save_weights(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    weights: numba.typed.List[tuple[int, int, float]],
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
):

    x_sweep, y_sweep, z_sweep = grid_sweep

    shape_sweep = x_sweep.shape

    i, j, k = index_sweep

    i_left = i - 1
    i_right = i

    j_lower = j - 1
    j_upper = j

    index_sweep_left = i_left, j, k
    index_sweep_lower = i, j_lower, k

    p0_left = (
        x_sweep[index_sweep_left],
        y_sweep[index_sweep_left],
        z_sweep[index_sweep_left],
    )
    p0_lower = (
        x_sweep[index_sweep_lower],
        y_sweep[index_sweep_lower],
        z_sweep[index_sweep_lower],
    )

    p1, p2 = line

    volume_left = rg.geometry.volume_tetrahedron(p0_left, p1, p2)
    volume_lower = -rg.geometry.volume_tetrahedron(p0_lower, p1, p1)

    if i_left >= 0:
        if j_lower >= 0:

            index_sweep_00 = i_left, j_lower, k

            index_input_00, index_output_00 = _index_input_output(
                index_sweep=index_sweep_00,
                index_static=index_static,
                sweep_input=sweep_input,
                axis_sweep=axis_sweep,
            )

            volume_input_lower_left = volume_input[index_input_00]

            index_input_00 = _arrays.index_flat(index_input_00, shape_cells_input)
            index_output_00 = _arrays.index_flat(index_output_00, shape_cells_output)

            weights.append((index_input_00, index_output_00, volume_lower_left))
            volume_lower_left = (-volume_lower - volume_left) / volume_input_lower_left

        if j_upper < (shape_sweep[1] - 1):

            index_sweep_01 = i_left, j_upper, k

            index_input_01, index_output_01 = _index_input_output(
                index_sweep=index_sweep_01,
                index_static=index_static,
                sweep_input=sweep_input,
                axis_sweep=axis_sweep,
            )

            volume_input_upper_left = volume_input[index_input_01]

            index_input_01 = _arrays.index_flat(index_input_01, shape_cells_input)
            index_output_01 = _arrays.index_flat(index_output_01, shape_cells_output)

            weights.append((index_input_01, index_output_01, volume_upper_left))
            volume_upper_left = volume_left / volume_input_upper_left

    if i_right < (shape_sweep[0] - 1):
        if j_lower >= 0:

            index_sweep_10 = i_right, j_lower, k

            index_input_10, index_output_10 = _index_input_output(
                index_sweep=index_sweep_10,
                index_static=index_static,
                sweep_input=sweep_input,
                axis_sweep=axis_sweep,
            )

            volume_input_lower_right = volume_input[index_input_10]

            index_input_10 = _arrays.index_flat(index_input_10, shape_cells_input)
            index_output_10 = _arrays.index_flat(index_output_10, shape_cells_output)

            volume_lower_right = volume_lower / volume_input_lower_right


@numba.njit(cache=True)
def _grid_sweep_static(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    sweep_input: bool,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
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

    # axis = axis_sweep[~0]
    # grid_sweep_x = _arrays.align_axis_right(grid_sweep_x, axis)
    # grid_sweep_y = _arrays.align_axis_right(grid_sweep_y, axis)
    # grid_sweep_z = _arrays.align_axis_right(grid_sweep_z, axis)
    #
    # grid_sweep = grid_sweep_x, grid_sweep_y, grid_sweep_z

    return grid_sweep, grid_static


@numba.njit(cache=True)
def _index_input_output(
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    sweep_input: bool,
    axis_sweep: tuple[int] | tuple[int, int],
):
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

    axis = axis_sweep[~0]

    i_sweep, j_sweep, k_sweep = index_sweep

    if axis == axis_x:
        index_sweep = k_sweep, i_sweep, j_sweep
    elif axis == axis_y:
        index_sweep = j_sweep, k_sweep, i_sweep

    if sweep_input:
        index_input = index_sweep
        index_output = index_static
    else:
        index_input = index_static
        index_output = index_sweep

    return index_input, index_output


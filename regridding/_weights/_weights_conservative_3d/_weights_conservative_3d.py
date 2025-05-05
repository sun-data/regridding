import sys
import numpy as np
import numpy.typing as npt
import numba
import regridding as rg
from . import  (
    _arrays,
    _volumes,
    _grids,
    _intercepts,
)
from _arrays import axis_x, axis_y, axis_z

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

    volume_input = _volumes.volume_grid(grid_input)

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

    # _sweep_intercepts(
    #     grid_input=grid_input,
    #     grid_output=grid_output,
    #     volume_input=volume_input,
    #     intercepts=intercepts,
    #     weights=weights,
    # )

    return weights, intercepts


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

    grid_sweep, grid_static = _grid_sweep_static(grid_input, grid_output, sweep_input)

    grid_static_x, grid_static_y, grid_static_z = grid_static

    bbox_boundary = (
        (grid_static_x.min(), grid_static_y.min(), grid_static_z.min()),
        (grid_static_x.max(), grid_static_y.max(), grid_static_z.max()),
    )

    indices_boundary, triangles_boundary = _grids.boundary(grid_static)

    axes_parallel = (
        axis_x,
        axis_y,
        axis_z,
    )

    axes_diagonal = (
        (axis_x, axis_y),
        (axis_y, axis_z),
        (axis_z, axis_x),
    )

    for axis in axes_parallel:
        _sweep_along_axis(
            grid_input=grid_input,
            grid_output=grid_output,
            volume_input=volume_input,
            bbox_boundary=bbox_boundary,
            indices_boundary=indices_boundary,
            triangles_boundary=triangles_boundary,
            weights=weights,
            intercepts=intercepts,
            sweep_input=sweep_input,
            axes=axis,
        )

    for axis in axes_diagonal:
        _sweep_along_diagonal()


@numba.njit(cache=True, parallel=True)
def _sweep_along_axis(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    bbox_boundary: tuple[tuple[float, float, float], tuple[float, float, float]],
    indices_boundary: tuple[np.ndarray, np.ndarray, np.ndarray],
    coord_boundary: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: numba.typed.List[tuple[int, int, float]],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis: int,
) -> None:
    """
    Sweep along one logical axis of either `grid_input` or `grid_output`.
    At each vertex, compute and save the weights and intercepts.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.
    grid_output
        The vertices of the new grid.
    volume_input
        The area of the input grid which will used to compute the weight.
    bbox_boundary
        Two points defining the bounding box of the static grid.
    coord_boundary
        A sequence of triangles defining the outer surface of the static grid.
    index_boundary
        The index of the static grid corresponding to each triangle in
        `coord_boundary`.
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
    axis
        The logical axis of the sweep grid to iterate along.
    """

    grid_sweep, grid_static = _grid_sweep_static(
        grid_input=grid_input,
        grid_output=grid_output,
        sweep_input=sweep_input,
        axis_x=axis,
    )

    x_sweep, y_sweep, z_sweep = grid_sweep

    shape_sweep_x, shape_sweep_y, shape_sweep_z = grid_sweep.shape
    shape_static_x, shape_static_y, shape_static_z = grid_static.shape

    weight = numba.typed.List()

    for i in range(shape_sweep_x):
        weight_i = numba.typed.List()
        for _ in range(0):
            weight_i.append((0, 0, 0.))
        weight.append(weight_i)

    for i in numba.prange(shape_sweep_x):

        i = numba.types.int64(i)

        for j in range(shape_sweep_y):

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
                box=bbox_boundary,
            ):
                if rg.geometry.point_is_inside_polyhedron(
                    point=point_1,
                    polyhedron=coord_boundary,
                ):
                    index_static = _grids.index_of_point_brute(
                        point=point_1,
                        grid=grid_static,
                    )
                    sweep_is_outside_static = False

            while k < (shape_sweep_z - 1):

                point_2 = (
                    x_sweep[i, j, k + 1],
                    y_sweep[i, j, k + 1],
                    z_sweep[i, j, k + 1],
                )

                if sweep_is_outside_static:

                    point_2, k, index_static = _step_outside_static(
                        point_1=point_1,
                        point_2=point_2,
                        bbox_boundary=bbox_boundary,
                        index_boundary=index_boundary,
                        coord_boundary=coord_boundary,
                        intercepts=intercepts,
                    )

                    sweep_is_outside_static = index_static[0] == sys.maxsize

                else:

                    point_2, k, index_static = _step_inside_static(
                        point_1=point_1,
                        point_2=point_2,
                        index_static=index_static,
                        grid_static=grid_static,
                        weights=weights,
                        intercepts=intercepts,
                    )

                    if not 0 <= index_static[axis_x] < (shape_static_x - 1):
                        break
                    if not 0 <= index_static[axis_y] < (shape_static_y - 1):
                        break
                    if not 0 <= index_static[axis_z] < (shape_static_z - 1):
                        break

    for i in range(shape_sweep_x):
        weight_i = weight[i]
        for w in range(len(weight_i)):
            weights.append(weight_i[w])


@numba.njit(cache=True)
def _step_outside_static(
    point_1: tuple[float, float, float],
    point_2: tuple[float, float, float],
    index_sweep: tuple[int, int, int],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox_boundary: tuple[tuple[float, float, float], tuple[float, float, float]],
    indices_boundary: numba.typed.List[
        tuple[int, int, int],
    ],
    triangles_boundary: numba.typed.List[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis: int,
) -> tuple[tuple[float, float, float], int, tuple[int, int, int]]:
    """
    Check if the current line segment crosses the boundary.
    If it does, compute the intersection and return the intersection point
    and the indices of the `grid_static` where the intersection occurs.

    Parameters
    ----------
    point_1
        The first point of the line segment.
    point_2
        The second point of the line segment.
    bbox_boundary
        Two points defining the bounding box of the static grid.
    indices_boundary
        A sequence of triangle indices defining the outer boundary of the
        static grid.
    intercepts
        A sorted list of intercepts to be traversed later.
        As new intercepts are found, they are inserted into the list.
    """

    line = (point_1, point_2)

    x, y, z = grid_static

    if rg.geometry.point_is_inside_box_3d(point_2, bbox_boundary):

        for t in range(len(triangles_boundary)):

            triangle = triangles_boundary[t]

            tuv = rg.geometry.line_triangle_intersection_parameters(
                line=line,
                triangle=triangle
            )

            if rg.geometry.line_intersects_triangle(tuv):

                index_static = indices_boundary[t]

                intercept = rg.geometry.line_triangle_intersection(
                    line=line,
                    tuv=tuv,
                )

                index_input, index_output = _index_input_output(
                    index_sweep=index_sweep,
                    index_static=index_static,
                    sweep_input=sweep_input,
                    axis=axis,
                )

                if axis == axis_x:

                    pass


@numba.njit(cache=True)
def _step_inside_static(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    grid_sweep: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape_cells_input: tuple[int, int, int],
    shape_cells_output: tuple[int, int, int],
    weights: numba.typed.List[tuple[int, int, float]],
    intercepts: numba.typed.List,
    sweep_input: bool,
    axis: int,
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
        The current index in the sweep grid.
        This is assumed to be a valid, positive index.
    index_static
        The current index in the static grid.
        This is assumed to be a valid, positive index.
    grid_sweep
        The vertices of the sweep grid.
        The last axis of this grid must be the sweep axis.
    grid_static
        Coordinates of the static grid.
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
    axis
        The logical axis of the sweep grid to iterate along.
    """

    p1, p2 = line

    i, j, k = index_sweep
    l, m, n = index_static

    x_sweep, y_sweep, z_sweep = grid_sweep
    x_static, y_static, z_static = grid_static

    shape_sweep = x_sweep.shape

    i_000 = l + 0, m + 0, n + 0
    i_001 = l + 0, m + 0, n + 1
    i_010 = l + 0, m + 1, n + 0
    i_011 = l + 0, m + 1, n + 1
    i_100 = l + 1, m + 0, n + 0
    i_101 = l + 1, m + 0, n + 1
    i_110 = l + 1, m + 1, n + 0
    i_111 = l + 1, m + 1, n + 1

    indices_triangles = (
        (i_000, i_010, i_110),
        (i_110, i_100, i_000),
        (i_001, i_101, i_111),
        (i_111, i_011, i_001),
        (i_000, i_001, i_011),
        (i_011, i_010, i_000),
        (i_100, i_110, i_111),
        (i_111, i_101, i_100),
        (i_000, i_100, i_101),
        (i_101, i_001, i_000),
        (i_010, i_011, i_111),
        (i_111, i_110, i_010),
    )

    axis_index = (
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    )

    normals_index = (
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, +1),
        (0, 0, +1),
        (-1, 0, 0),
        (-1, 0, 0),
        (+1, 0, 0),
        (+1, 0, 0),
        (0, -1, 0),
        (0, -1, 0),
        (0, +1, 0),
        (0, +1, 0),
    )

    for t in range(len(indices_triangles)):

        i0, i1, i2 = indices_triangles[t]

        v0 = (x_static[i0], y_static[i0], z_static[i0])
        v1 = (x_static[i1], y_static[i1], z_static[i1])
        v2 = (x_static[i2], y_static[i2], z_static[i2])

        triangle = (v0, v1, v2)

        tuv = rg.geometry.line_triangle_intersection_parameters(
            line=line,
            triangle=triangle,
        )

        if rg.geometry.line_intersects_triangle(tuv):

            intercept = rg.geometry.line_triangle_intersection(
                line=line,
                tuv=tuv,
            )

            p2 = intercept

            normal_index = normals_index[t]

            index_static_new = rg.math.sum_3d(index_static, normal_index)

            axis_static = axis_index[t]

            if index_static_new[axis_static] > index_static[axis_static]:
                index_static_intercept = index_static_new
            else:
                index_static_intercept = index_static

            axes_orthogonal = [ax for ax in _arrays.axes if ax != axis]

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
                    axis=axis,
                )

                i_input = index_input[axis_input]
                i_output = index_output[axis_output]

                _intercepts.insert_intercept(
                    intercepts=intercepts[axis_input][axis_output][i_input][i_output],
                    intercept_new=(
                        _arrays.index_flat(index_input, shape_cells_input),
                        _arrays.index_flat(index_output, shape_cells_output),
                        intercept,
                    ),
                )

                break

    else:
        index_sweep = i, j, k + 1

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

    volume_left = rg.geometry.volume_tetrahedron(p0_left, p1, p2)
    volume_lower = -rg.geometry.volume_tetrahedron(p0_lower, p1, p1)

    if i_left >= 0:
        if j_lower >= 0:

            index_sweep_00 = i_left, j_lower, k

            index_input_00, index_output_00 = _index_input_output(
                index_sweep=index_sweep_00,
                index_static=index_static,
                sweep_input=sweep_input,
                axis=axis,
            )

            index_input_00 = _arrays.index_flat(index_input_00, shape_cells_input)
            index_output_00 = _arrays.index_flat(index_output_00, shape_cells_output)

            volume_lower_left = -volume_lower - volume_left
            weights.append((index_input_00, index_output_00, volume_lower_left))

        if j_upper < (shape_sweep[1] - 1):

            index_sweep_01 = i_left, j_upper, k

            index_input_01, index_output_01 = _index_input_output(
                index_sweep=index_sweep_01,
                index_static=index_static,
                sweep_input=sweep_input,
                axis=axis,
            )

            index_input_01 = _arrays.index_flat(index_input_01, shape_cells_input)
            index_output_01 = _arrays.index_flat(index_output_01, shape_cells_output)

            volume_upper_left = volume_left
            weights.append((index_input_01, index_output_01, volume_upper_left))

    if i_right < (shape_sweep[0] - 1):
        if j_lower >= 0:

            index_sweep_10 = i_right, j_lower, k

            index_input_10, index_output_10 = _index_input_output(
                index_sweep=index_sweep_10,
                index_static=index_static,
                sweep_input=sweep_input,
                axis=axis,
            )

            index_input_10 = _arrays.index_flat(index_input_10, shape_cells_input)
            index_output_10 = _arrays.index_flat(index_output_10, shape_cells_output)

            volume_lower_right = volume_lower
            weights.append((index_input_10, index_output_10, volume_lower_right))

    line = p1, p2

    return line, index_sweep, index_static


@numba.njit(cache=True)
def _grid_sweep_static(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    sweep_input: bool,
    axis: int,
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
    axis
        The logical axis of the `grid_sweep` to iterate along.
    """
    if sweep_input:
        grid_sweep = grid_input
        grid_static = grid_output
    else:
        grid_sweep = grid_output
        grid_static = grid_input

    grid_sweep_x, grid_sweep_y, grid_sweep_z = grid_sweep

    grid_sweep_x = _arrays.align_axis_right(grid_sweep_x, axis)
    grid_sweep_y = _arrays.align_axis_right(grid_sweep_y, axis)
    grid_sweep_z = _arrays.align_axis_right(grid_sweep_z, axis)

    grid_sweep = grid_sweep_x, grid_sweep_y, grid_sweep_z

    return grid_sweep, grid_static


@numba.njit(cache=True)
def _index_input_output(
    index_sweep: tuple[int, int, int],
    index_static: tuple[int, int, int],
    sweep_input: bool,
    axis: int,
):

    i_sweep, j_sweep, k_sweep = index_sweep

    if axis == axis_x:
        index_sweep = j_sweep, k_sweep, i_sweep
    elif axis == axis_y:
        index_sweep = k_sweep, i_sweep, j_sweep

    if sweep_input:
        index_input = index_sweep
        index_output = index_static
    else:
        index_input = index_static
        index_output = index_sweep

    return index_input, index_output


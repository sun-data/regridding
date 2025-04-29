import sys
import numpy as np
import numpy.typing as npt
import numba
import regridding

__all__ = [
    "weights_conservative_3d",
]

axis_x = 0
axis_y = 1
axis_z = 2


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
        weights.append((0.0, 0.0, 0.0))

    volume_input = _cell_volume(grid_input)

    intercepts = _empty_intercepts(shape_input, shape_output)

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

    index_boundary, coord_boundary = _calc_boundary(grid_static)

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
            index_boundary=index_boundary,
            coord_boundary=coord_boundary,
            weights=weights,
            intercepts=intercepts,
            sweep_input=sweep_input,
            axes=axis,
        )

    for axis in axes_diagonal:
        _sweep_along_diagonal(

        )


@numba.njit(cache=True, parallel=True)
def _sweep_along_axis(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
    bbox_boundary: tuple[tuple[float, float, float], tuple[float, float, float]],
    index_boundary: tuple[np.ndarray, np.ndarray, np.ndarray],
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
    sweep_input
        A boolean flag indicating whether to iterate along `grid_input`
        or `grid_output`.
        If :obj:`True`, this function sweeps along `grid_input`.
    axis
        The logical axis of the `grid_sweep` to iterate along.
    """

    grid_sweep, grid_static = _grid_sweep_static(grid_input, grid_output, sweep_input)

    grid_sweep_x, grid_sweep_y, grid_sweep_z = grid_sweep
    grid_static_x, grid_static_y, grid_static_z = grid_static

    grid_sweep_x = _align_axis_right(grid_sweep_x, axis)
    grid_sweep_y = _align_axis_right(grid_sweep_y, axis)
    grid_sweep_z = _align_axis_right(grid_sweep_z, axis)

    shape_sweep_x, shape_sweep_y, shape_sweep_z = grid_sweep.shape
    shape_static_x, shape_static_y, shape_static_z = grid_static.shape

    for i in numba.prange(shape_sweep_x):

        i = numba.types.int64(i)

        for j in range(shape_sweep_y):

            k = 0

            index_sweep = i, j, k

            index_static = sys.maxsize, sys.maxsize, sys.maxsize

            sweep_is_outside_static = True

            point_1 = (
                grid_sweep_x[index_sweep],
                grid_sweep_y[index_sweep],
                grid_sweep_z[index_sweep],
            )

            if regridding.geometry.point_is_inside_box_3d(
                point=point_1,
                box=bbox_boundary,
            ):
                if regridding.geometry.point_is_inside_polyhedron(
                    point=point_1,
                    polyhedron=coord_boundary,
                ):
                    index_static = _index_of_point_brute(
                        point=point_1,
                        grid=grid_static,
                    )

            while True:

                point_2 = (
                    grid_sweep_x[i, j, k + 1],
                    grid_sweep_y[i, j, k + 1],
                    grid_sweep_z[i, j, k + 1],
                )

                sweep_is_outside_static = index_static[0] == sys.maxsize

                if sweep_is_outside_static:

                    point_2, k, index_static = _step_outside_static(
                        point_1=point_1,
                        point_2=point_2,
                        bbox_boundary=bbox_boundary,
                        index_boundary=index_boundary,
                        coord_boundary=coord_boundary,
                        intercepts=intercepts,
                    )

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

                if k >= (shape_sweep_z - 1):
                    break








@numba.njit(cache=True)
def _step_outside_static(
    point_1: tuple[float, float, float],
    point_2: tuple[float, float, float],
    index_static: tuple[int, int, int],
    index_sweep: tuple[int, int, int],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    bbox_boundary: tuple[tuple[float, float, float], tuple[float, float, float]],
    coord_boundary: tuple[np.ndarray, np.ndarray, np.ndarray],
    index_boundary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[
    tuple[float, float, float],
]:
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
    index_static
        The current 3D index in the static grid.
    index_sweep
        The current 3D index in the sweep grid.
    grid_static
        Coordinates of the static grid.
    bbox_boundary
        Two points defining the bounding box of the static grid.
    coord_boundary
        A sequence of triangles defining the outer surface of the static grid.
    index_boundary
        The index of the static grid corresponding to each triangle in
        `coord_boundary`.
    """


@numba.njit(cache=True)
def _step_inside_static(
    point_1: tuple[float, float, float],
    point_2: tuple[float, float, float],
    index_static: tuple[int, int, int],
    index_sweep: tuple[int, int, int],
    grid_static: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: numba.typed.List[tuple[int, int, float]],
) -> None:
    """
    Check if the current line segment crosses any face of the current cell
    in the static grid.
    If it does, compute the intersection point and indices of the new
    cell in the static grid.

    Parameters
    ----------
    point_1
        The first point of the line segment.
    point_2
        The second point of the line segment.
    index_static
        The current index in the static grid.
    index_sweep
        The current index in the sweep grid.
    grid_static
        Coordinates of the static grid.
    weights
        The current list of weights.
        New weights will be appended to this list.
    """


@numba.njit(cache=True)
def _grid_sweep_static(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    sweep_input: bool,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    if sweep_input:
        grid_sweep = grid_input
        grid_static = grid_output
    else:
        grid_sweep = grid_output
        grid_static = grid_input

    return grid_sweep, grid_static


@numba.njit(cache=True)
def _cell_volume(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Compute the volume of each cell in a logically-rectangular grid.

    Parameters
    ----------
    grid
        A 3D grid of cell vertices.
    """

    x, y, z = grid

    num_i, num_j, num_k = x.shape

    result = np.zeros((num_i - 1, num_j - 1, num_k - 1))

    for axis in (0, 1, 2):
        _cell_volume_sweep(
            grid=grid,
            out=result,
            axis=axis,
        )

    return result


@numba.njit(cache=True)
def _cell_volume_sweep(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    out: np.ndarray,
    axis: int,
) -> None:
    """
    Compute the volume contribution of this axis.

    Parameters
    ----------
    grid
        A 3D grid of cell vertices.
    out
        An output to array to store the result.
    axis
        The axis along which to iterate.
    """

    x, y, z = grid

    x = _align_axis_right(x, axis)
    y = _align_axis_right(y, axis)
    z = _align_axis_right(z, axis)

    out = _align_axis_right(out, axis)

    num_i, num_j, num_k = x.shape

    for i in numba.prange(num_i):

        i = numba.types.int64(i)

        for j in range(num_j):

            for k in range(num_k - 1):

                i_left = i - 1
                i_right = i

                j_lower = j - 1
                j_upper = j

                k1 = k
                k2 = k + 1

                v1 = x[i, j, k1], y[i, j, k1], z[i, j, k1]
                v2 = x[i, j, k2], y[i, j, k2], z[i, j, k2]

                volume_left = 0
                volume_lower = 0

                if i_left >= 0:
                    v0_left = x[i_left, j, k], y[i_left, j, k], z[i_left, j, k]
                    volume_left = _volume(v0_left, v1, v2)

                if j_lower >= 0:
                    v0_lower = x[i, j_lower, k], y[i, j_lower, k], z[i, j_lower, k]
                    volume_lower = -_volume(v0_lower, v1, v2)

                if i_left >= 0:
                    if j_lower >= 0:
                        out[i_left, j_lower, k] -= volume_left
                    if j_upper < (num_j - 1):
                        out[i_left, j_upper, k] += volume_left

                if j_lower >= 0:
                    if i_left >= 0:
                        out[i_left, j_lower, k] -= volume_lower
                    if i_right < (num_i - 1):
                        out[i_right, j_lower, k] += volume_lower


@numba.njit(cache=True)
def _volume(
    v0: tuple[float, float, float],
    v1: tuple[float, float, float],
    v2: tuple[float, float, float],
) -> float:
    """
    Compute the volume of a tetrahedron constructed from three
    vertices and the origin.

    Parameters
    ----------
    v0
        First vertex of the tetrahedron.
    v1
        Second vertex of the tetrahedron.
    v2
        Third vertex of the tetrahedron.
    """
    triple_product = regridding.math.dot_3d(
        a=v0,
        b=regridding.math.cross_3d(v1, v2),
    )

    return triple_product / 6


@numba.njit(cache=True)
def _empty_intercepts(
    shape_input: tuple[int, int, int],
    shape_output: tuple[int, int, int],
) -> numba.typed.List:
    """
    Create an empty list of intercepts between each plane in the
    input grid and each plane in the output grid.

    Parameters
    ----------
    shape_input
        The shape of the input grid.
    shape_output
        The shape of the output grid.
    """
    intercepts = numba.typed.List()
    for a in range(len(shape_input)):
        intercepts_a = numba.typed.List()
        for b in range(len(shape_output)):
            intercepts_ab = numba.typed.List()
            for i in range(shape_input[a]):
                intercepts_abi = numba.typed.List()
                for j in range(shape_output[b]):
                    intercepts_abij = numba.typed.List()
                    for x in range(0):
                        intercepts_abij.append((0.0, 0.0, 0.0))
                    intercepts_abi.append(intercepts_abij)
                intercepts_ab.append(intercepts_abi)
            intercepts_a.append(intercepts_ab)
        intercepts.append(intercepts_a)
    return intercepts


@numba.njit(cache=True)
def _align_axis_right(
    a: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Roll all the axes of a 3D array to the right until `axis` is the last axis..

    This function is needed to permute the axes in such a way to retain
    their right-handedness.

    Parameters
    ----------
    a
        The array to modify the axes of.
    axis
        The axis to set as the last axis.
    """
    axis = axis % a.ndim

    if axis == axis_x:
        result = a.transpose((axis_y, axis_z, axis_x))
    elif axis == axis_y:
        result = a.transpose((axis_z, axis_x, axis_y))
    else:
        result = a

    return result


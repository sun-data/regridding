"""
Utilities for inspecting and searching logically-rectangular grids of
coordinates.
"""

import numpy as np
import numba
import regridding as rg
from . import _arrays

__all__ = [
    "shape_centers",
    "cell_axes",
    "cell_normals",
    "cell_boundary",
    "grid_boundary",
    "index_of_point_brute",
]


@numba.njit(cache=True)
def shape_centers(
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Given the shape of the grid of cell vertices,
    compute the shape of the grid of cell centers.

    Parameters
    ----------
    shape
        The shape of the grid of cell vertices.
    """

    nx, ny, nz = shape

    return nx - 1, ny - 1, nz - 1


@numba.njit(cache=True)
def grid_volume(
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
        _grid_volume_sweep(
            grid=grid,
            out=result,
            axis=axis,
        )

    return result


@numba.njit(cache=True)
def _grid_volume_sweep(
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

    x = _arrays.align_axis_right(x, axis)
    y = _arrays.align_axis_right(y, axis)
    z = _arrays.align_axis_right(z, axis)

    out = _arrays.align_axis_right(out, axis)

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

                if i_left >= 0:

                    v0_left = x[i_left, j, k], y[i_left, j, k], z[i_left, j, k]
                    volume_left = rg.geometry.volume_tetrahedron(v0_left, v1, v2)

                    if j_lower >= 0:
                        out[i_left, j_lower, k] -= volume_left
                    if j_upper < (num_j - 1):
                        out[i_left, j_upper, k] += volume_left

                if j_lower >= 0:

                    v0_lower = x[i, j_lower, k], y[i, j_lower, k], z[i, j_lower, k]
                    volume_lower = -rg.geometry.volume_tetrahedron(v0_lower, v1, v2)

                    if i_left >= 0:
                        out[i_left, j_lower, k] -= volume_lower
                    if i_right < (num_i - 1):
                        out[i_right, j_lower, k] += volume_lower


cell_axes = (
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
"""
The index of the axis normal to each face in :func:`cell_boundary`.
"""

cell_normals = (
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
"""
Vectors normal to each face in :func:`cell_boundary`.
"""


@numba.njit(cache=True)
def cell_boundary(
    index: tuple[int, int, int],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> numba.typed.List[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
]:
    """
    For a given cell in a grid of cell vertices,
    compute the 12 triangles composing the boundary.

    Parameters
    ----------
    index
        The index of the cell to compute the boundary of.
    grid
        A grid of cell vertices.

    """
    i, j, k = index

    x, y, z = grid

    i_000 = i + 0, j + 0, k + 0
    i_001 = i + 0, j + 0, k + 1
    i_010 = i + 0, j + 1, k + 0
    i_011 = i + 0, j + 1, k + 1
    i_100 = i + 1, j + 0, k + 0
    i_101 = i + 1, j + 0, k + 1
    i_110 = i + 1, j + 1, k + 0
    i_111 = i + 1, j + 1, k + 1

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

    polyhedron = numba.typed.List()

    for t in range(len(indices_triangles)):

        i0, i1, i2 = indices_triangles[t]

        v0 = x[i0], y[i0], z[i0]
        v1 = x[i1], y[i1], z[i1]
        v2 = x[i2], y[i2], z[i2]

        polyhedron.append((v0, v1, v2))

    return polyhedron


@numba.njit(cache=True)
def grid_boundary(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[
    numba.typed.List[
        tuple[int, int, int],
    ],
    numba.typed.List[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ],
]:
    """
    For a given grid, find the indices of the boundary cells
    and the vertices of all the triangles on the boundary.

    find the set of indices that expresses the boundary
    as a sequence of triangles.

    Parameters
    ----------
    grid
        A logically-rectangular grid of cell vertices.
    """

    x, y, z = grid

    shape_x, shape_y, shape_z = x.shape

    indices = numba.typed.List()

    triangles = numba.typed.List()

    for i0 in range(shape_x - 1):
        for j0 in range(shape_y - 1):

            k0 = 0

            i1 = i0 + 1
            j1 = j0 + 1

            i_cell = i0, j0, k0

            i_000 = i0, j0, k0
            i_010 = i0, j1, k0
            i_110 = i1, j1, k0
            i_100 = i1, j0, k0

            v_000 = x[i_000], y[i_000], z[i_000]
            v_010 = x[i_010], y[i_010], z[i_010]
            v_110 = x[i_110], y[i_110], z[i_110]
            v_100 = x[i_100], y[i_100], z[i_100]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_000, v_010, v_110))
            triangles.append((v_110, v_100, v_000))

    for i0 in range(shape_x - 1):
        for j0 in range(shape_y - 1):

            k1 = shape_z

            i1 = i0 + 1
            j1 = j0 + 1

            i_cell = i0, j0, k1 - 1

            i_001 = i0, j0, k1
            i_101 = i1, j0, k1
            i_111 = i1, j1, k1
            i_011 = i0, j1, k1

            v_001 = x[i_001], y[i_001], z[i_001]
            v_101 = x[i_101], y[i_101], z[i_101]
            v_111 = x[i_111], y[i_111], z[i_111]
            v_011 = x[i_011], y[i_011], z[i_011]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_001, v_101, v_111))
            triangles.append((v_111, v_011, v_001))

    for j0 in range(shape_y - 1):
        for k0 in range(shape_z - 1):

            i0 = 0

            j1 = j0 + 1
            k1 = k0 + 1

            i_cell = i0, j0, k0

            i_000 = i0, j0, k0
            i_001 = i0, j0, k1
            i_011 = i0, j1, k1
            i_010 = i0, j1, k0

            v_000 = x[i_000], y[i_000], z[i_000]
            v_001 = x[i_001], y[i_001], z[i_001]
            v_011 = x[i_011], y[i_011], z[i_011]
            v_010 = x[i_010], y[i_010], z[i_010]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_000, v_001, v_011))
            triangles.append((v_011, v_010, v_000))

    for j0 in range(shape_y - 1):
        for k0 in range(shape_z - 1):

            i1 = shape_x

            j1 = j0 + 1
            k1 = k0 + 1

            i_cell = i1 - 1, j0, k0

            i_100 = i1, j0, k0
            i_110 = i1, j1, k0
            i_111 = i1, j1, k1
            i_101 = i1, j0, k1

            v_100 = x[i_100], y[i_100], z[i_100]
            v_110 = x[i_110], y[i_110], z[i_110]
            v_111 = x[i_111], y[i_111], z[i_111]
            v_101 = x[i_101], y[i_101], z[i_101]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_100, v_110, v_111))
            triangles.append((v_111, v_101, v_100))

    for i0 in range(shape_x - 1):
        for k0 in range(shape_z - 1):

            j0 = 0

            i1 = i0 + 1
            k1 = k0 + 1

            i_cell = i0, j0, k0

            i_000 = i0, j0, k0
            i_100 = i1, j0, k0
            i_101 = i1, j0, k1
            i_001 = i0, j0, k1

            v_000 = x[i_000], y[i_000], z[i_000]
            v_100 = x[i_100], y[i_100], z[i_100]
            v_101 = x[i_101], y[i_101], z[i_101]
            v_001 = x[i_001], y[i_001], z[i_001]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_000, v_100, v_101))
            triangles.append((v_101, v_001, v_000))

    for i0 in range(shape_x - 1):
        for k0 in range(shape_z - 1):

            j1 = shape_y

            i1 = i0 + 1
            k1 = k0 + 1

            i_cell = i0, j1 - 1, k0

            i_010 = i0, j1, k0
            i_011 = i0, j1, k1
            i_111 = i1, j1, k1
            i_110 = i1, j1, k0

            v_010 = x[i_010], y[i_010], z[i_010]
            v_011 = x[i_011], y[i_011], z[i_011]
            v_111 = x[i_111], y[i_111], z[i_111]
            v_110 = x[i_110], y[i_110], z[i_110]

            indices.append(i_cell)
            indices.append(i_cell)

            triangles.append((v_010, v_011, v_111))
            triangles.append((v_111, v_110, v_010))

    return indices, triangles


@numba.njit(cache=True)
def index_of_point_brute(
    point: tuple[float, float, float],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, int, int]:
    """
    Find the index of the cell in the grid which contains the given point.

    This function uses brute force to search,
    but this could be improved significantly by using the secant method
    or possibly the bisection method.

    Parameters
    ----------
    point
        The query point.
    grid
        A logically-rectangular grid of cell vertices.
    """

    x, y, z = grid

    shape_x, shape_y, shape_z = x.shape

    for i in range(shape_x - 1):
        for j in range(shape_y - 1):
            for k in range(shape_z - 1):

                index = i, j, k

                polyhedron = cell_boundary(
                    index=index,
                    grid=grid,
                )

                if rg.geometry.point_is_inside_polyhedron(
                    point=point,
                    polyhedron=polyhedron,
                ):
                    return index

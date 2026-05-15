"""
Utilities for inspecting and searching logically-rectangular grids of
coordinates.
"""

import sys
import numpy as np
import numba
import regridding as rg
from . import _arrays

__all__ = [
    "shape_centers",
    "cell_normals",
    "grid_boundary",
    "index_of_point_brute",
]


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def shape_centers(
    shape: tuple[int, int],
) -> tuple[int, int]:
    """
    Given the shape of the grid of cell vertices,
    compute the shape of the grid of cell centers.

    Parameters
    ----------
    shape
        The shape of the grid of cell vertices.
    """

    nx, ny = shape

    return nx - 1, ny - 1


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def grid_volume(
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Compute the volume of each cell in a logically-rectangular grid.

    Parameters
    ----------
    grid
        A 2D grid of cell vertices.
    """

    x, y = grid

    num_i, num_j = x.shape

    result = np.zeros(shape=(num_i - 1, num_j - 1))

    for axis in _arrays.axes:
        _grid_volume_sweep(
            grid=grid,
            out=result,
            axis=axis,
        )

    return result


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _grid_volume_sweep(
    grid: tuple[np.ndarray, np.ndarray],
    out: np.ndarray,
    axis: int,
) -> None:
    """
    Compute the volume contribution of this axis.

    Parameters
    ----------
    grid
        A 2D grid of cell vertices.
    out
        An output to array to store the result.
    axis
        The axis along which to iterate.
    """

    x, y = grid

    x = _arrays.align_axis_right(x, axis)
    y = _arrays.align_axis_right(y, axis)

    if axis == 0:
        x, y = y, x

    out = _arrays.align_axis_right(out, axis)

    num_i, num_j = x.shape

    for j in numba.prange(num_j - 1):

        for i in range(num_i):

            i_left = i - 1
            i_right = i

            j1 = j
            j2 = j + 1

            x1 = x[i, j1]
            y1 = y[i, j1]
            x2 = x[i, j2]
            y2 = y[i, j2]

            vertex_1 = (x1, y1)
            vertex_2 = (x2, y2)

            area = rg.geometry.area_triangle(vertex_1, vertex_2)

            if i_left >= 0:

                out[i_left, j] += area

            if i_right < (num_i - 1):

                out[i_right, j] -= area


cell_normals = (
    (-1, 0),
    (0, -1),
    (+1, 0),
    (0, +1),
)
"""
Vectors normal to each face in :func:`cell_boundary`.
"""

indices_cell_vertex = (
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
)
"""The indices of each vertex in a cell"""


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def grid_boundary(
    grid: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    For a given grid of cell vertices,
    return the vertices of the boundary.

    Parameters
    ----------
    grid
        A logically-rectangular grid of cell vertices.
    """

    x, y = grid

    shape_x, shape_y = x.shape

    num_vertices = 2 * (shape_x - 1) + 2 * (shape_y - 1)

    x_vertices = np.empty(num_vertices)
    y_vertices = np.empty(num_vertices)

    n = 0

    j = 0
    for i in range(shape_x - 1):
        x_vertices[n] = x[i, j]
        y_vertices[n] = y[i, j]
        n = n + 1

    i = ~0
    for j in range(shape_y - 1):
        x_vertices[n] = x[i, j]
        y_vertices[n] = y[i, j]
        n = n + 1

    j = ~0
    for i in range(shape_x - 1):
        x_vertices[n] = x[~i, j]
        y_vertices[n] = y[~i, j]
        n = n + 1

    i = 0
    for j in range(shape_y - 1):
        x_vertices[n] = x[i, ~j]
        y_vertices[n] = y[i, ~j]
        n = n + 1

    return x_vertices, y_vertices


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def index_of_point_brute(
    point: tuple[float, float],
    grid: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
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

    px, py = point

    x, y = grid

    shape_x, shape_y = x.shape

    vertices_x = np.empty(4)
    vertices_y = np.empty(4)

    for i in range(shape_x - 1):
        for j in range(shape_y - 1):

            index = i, j

            v1 = i + 0, j + 0
            v2 = i + 1, j + 0
            v3 = i + 1, j + 1
            v4 = i + 0, j + 1

            vertices_x[0] = x[v1]
            vertices_x[1] = x[v2]
            vertices_x[2] = x[v3]
            vertices_x[3] = x[v4]

            vertices_y[0] = y[v1]
            vertices_y[1] = y[v2]
            vertices_y[2] = y[v3]
            vertices_y[3] = y[v4]

            if rg.geometry.point_is_inside_polygon(
                x=px,
                y=py,
                vertices_x=vertices_x,
                vertices_y=vertices_y,
            ):
                return index

    return sys.maxsize, sys.maxsize

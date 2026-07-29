"""
Utilities for inspecting and searching logically-rectangular grids of
coordinates.
"""

import sys
import math
import numpy as np
import numba
import regridding as rg
from ..._interp_ndarray import _bilinear_interpolation
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


@numba.njit(
    cache=True,
    inline="always",
)
def _index_of_point_local(
    point: tuple[float, float],
    grid: tuple[np.ndarray, np.ndarray],
    i0: int,
    j0: int,
) -> tuple[int, int]:
    """
    Find the lowest-index cell containing ``point`` among the cells adjacent to
    ``(i0, j0)``.

    The secant iteration locates the cell ``(i0, j0)`` whose fractional index the
    converged iterate floors into. A point on a shared cell face or vertex also
    lies in a lower-index neighbour; this ascending search over the (at most)
    nine surrounding cells returns the same cell as
    :func:`index_of_point_brute`, which is required for the conservative
    resampling to conserve mass. Returns ``sys.maxsize, sys.maxsize`` if no
    adjacent cell contains the point (i.e. it lies outside the grid).

    Parameters
    ----------
    point
        The query point.
    grid
        A logically-rectangular grid of cell vertices.
    i0
        The first index of the cell the converged iterate floors into.
    j0
        The second index of the cell the converged iterate floors into.
    """

    px, py = point

    x, y = grid

    shape_x, shape_y = x.shape
    shape_cells = shape_x - 1, shape_y - 1

    vertices_x = np.empty(4)
    vertices_y = np.empty(4)

    for ii in range(max(i0 - 1, 0), min(i0 + 2, shape_cells[0])):
        for jj in range(max(j0 - 1, 0), min(j0 + 2, shape_cells[1])):
            i1 = ii + 1
            j1 = jj + 1

            vertices_x[0] = x[ii, jj]
            vertices_x[1] = x[i1, jj]
            vertices_x[2] = x[i1, j1]
            vertices_x[3] = x[ii, j1]

            vertices_y[0] = y[ii, jj]
            vertices_y[1] = y[i1, jj]
            vertices_y[2] = y[i1, j1]
            vertices_y[3] = y[ii, j1]

            if rg.geometry.point_is_inside_polygon(
                x=px,
                y=py,
                vertices_x=vertices_x,
                vertices_y=vertices_y,
            ):
                return ii, jj

    return sys.maxsize, sys.maxsize


@numba.njit(
    cache=True,
    inline="always",
)
def index_of_point_secant(
    point: tuple[float, float],
    grid: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
    """
    Find the index of the cell in the grid which contains the given point.

    Parameters
    ----------
    point
        The query point.
    grid
        A logically-rectangular grid of cell vertices.
    """

    h = 1e-3

    px, py = point

    x, y = grid

    shape_x, shape_y = x.shape

    shape_cells = shape_x - 1, shape_y - 1

    i = shape_x / 2
    j = shape_y / 2

    vertices_x = np.empty(4)
    vertices_y = np.empty(4)

    for _ in range(100):

        _x = _bilinear_interpolation(x, i, j)
        _y = _bilinear_interpolation(y, i, j)

        i0 = math.floor(i)
        j0 = math.floor(j)

        error_x = _x - px
        error_y = _y - py

        if _arrays.index_in_bounds(
            index=(i0, j0),
            shape=shape_cells,
        ):
            i1 = i0 + 1
            j1 = j0 + 1

            index_00 = i0, j0
            index_01 = i0, j1
            index_10 = i1, j0
            index_11 = i1, j1

            v1 = index_00
            v2 = index_10
            v3 = index_11
            v4 = index_01

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
                # the query point lies in cell (i0, j0). If it is on a shared
                # lower/left face a lower-index neighbour contains it too, so
                # resolve it with a *local* ascending search and return the
                # lowest-index containing cell, exactly matching
                # `index_of_point_brute` (required for conservation).
                return _index_of_point_local(point, grid, i0, j0)

        # the iterate has converged on the query point but it was not strictly
        # inside a cell, so it lies on a boundary face (or outside the grid).
        # Resolve the cell with the same local ascending search.
        if (abs(error_x) < 1e-10) and (abs(error_y) < 1e-10):
            return _index_of_point_local(point, grid, i0, j0)

        dx_di = (_bilinear_interpolation(x, i + h, j) - _x) / h
        dx_dj = (_bilinear_interpolation(x, i, j + h) - _x) / h
        dy_di = (_bilinear_interpolation(y, i + h, j) - _y) / h
        dy_dj = (_bilinear_interpolation(y, i, j + h) - _y) / h

        det = dx_di * dy_dj - dx_dj * dy_di

        # a singular Jacobian means the Newton step is undefined; defer to the
        # exhaustive search (rare).
        if det == 0:  # pragma: no cover
            return index_of_point_brute(point, grid)

        di = (+dy_dj * error_x - dx_dj * error_y) / det
        dj = (-dy_di * error_x + dx_di * error_y) / det

        i -= di
        j -= dj

    # the iteration did not converge; fall back to the exhaustive search (rare).
    return index_of_point_brute(point, grid)  # pragma: no cover

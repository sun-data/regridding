import numpy as np
import numba
import regridding as rg
from . import _arrays

__all__ = [
    "volume_tetrahedron",
    "volume_grid",
]


@numba.njit(cache=True)
def volume_tetrahedron(
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
    triple_product = rg.math.dot_3d(
        a=v0,
        b=rg.math.cross_3d(v1, v2),
    )

    return triple_product / 6


@numba.njit(cache=True)
def volume_grid(
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
        _volume_grid_sweep(
            grid=grid,
            out=result,
            axis=axis,
        )

    return result


@numba.njit(cache=True)
def _volume_grid_sweep(
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

                volume_left = 0
                volume_lower = 0

                if i_left >= 0:
                    v0_left = x[i_left, j, k], y[i_left, j, k], z[i_left, j, k]
                    volume_left = volume_tetrahedron(v0_left, v1, v2)

                if j_lower >= 0:
                    v0_lower = x[i, j_lower, k], y[i, j_lower, k], z[i, j_lower, k]
                    volume_lower = -volume_tetrahedron(v0_lower, v1, v2)

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

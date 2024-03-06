from typing import Sequence
import numpy as np
import numba
import regridding._util

__all__ = [
    "fill_gauss_seidel",
]


def fill_gauss_seidel(
    a: np.ndarray,
    where: np.ndarray,
    axis: None | int | Sequence[int],
    num_iterations: int = 100,
) -> np.ndarray:

    a = a.copy()

    a, where = np.broadcast_arrays(a, where, subok=True)

    a[where] = 0

    axis = regridding._util._normalize_axis(axis=axis, ndim=a.ndim)
    axis_numba = ~np.arange(len(axis))[::-1]

    shape = a.shape
    shape_numba = tuple(shape[ax] for ax in axis)

    a = np.moveaxis(a, axis, axis_numba)
    where = np.moveaxis(where, axis, axis_numba)

    shape_moved = a.shape

    a = a.reshape(-1, *shape_numba)
    where = where.reshape(-1, *shape_numba)

    if len(axis) == 2:
        result = _fill_gauss_seidel_2d(
            a=a,
            where=where,
            num_iterations=num_iterations,
        )
    else:  # pragma: nocover
        raise ValueError(
            f"The number of interpolation axes, {len(axis)}," f"is not supported"
        )

    result = result.reshape(shape_moved)
    result = np.moveaxis(result, axis_numba, axis)

    return result


@numba.njit(parallel=True)
def _fill_gauss_seidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    num_iterations: int,
) -> np.ndarray:

    num_t, num_y, num_x = a.shape

    for t in numba.prange(num_t):
        for k in range(num_iterations):
            for is_odd in [False, True]:
                _iteration_gauss_seidel_2d(
                    a=a,
                    where=where,
                    t=t,
                    num_x=num_x,
                    num_y=num_y,
                    is_odd=is_odd,
                )

    return a


@numba.njit(fastmath=True)
def _iteration_gauss_seidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    t: int,
    num_x: int,
    num_y: int,
    is_odd: bool,
) -> None:

    xmin, xmax = -1, 1
    ymin, ymax = -1, 1

    dx = (xmax - xmin) / (num_x - 1)
    dy = (ymax - ymin) / (num_y - 1)

    dxxinv = 1 / (dx * dx)
    dyyinv = 1 / (dy * dy)

    dcent = 1 / (2 * (dxxinv + dyyinv))

    for j in range(num_y):
        for i in range(num_x):
            if (i + j) & 1 == is_odd:
                if where[t, j, i]:
                    i9 = (i - 1) % num_x
                    i1 = (i + 1) % num_x
                    j9 = (j - 1) % num_y
                    j1 = (j + 1) % num_y

                    xterm = dxxinv * (a[t, j, i9] + a[t, j, i1])
                    yterm = dyyinv * (a[t, j9, i] + a[t, j1, i])
                    a[t, j, i] = (xterm + yterm) * dcent

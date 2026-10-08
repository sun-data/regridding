import warnings
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
    guess: None | float | np.ndarray = None,
    num_iterations: int = 100,
) -> np.ndarray:

    shape = np.broadcast_shapes(a.shape, where.shape)

    axis = regridding._util._normalize_axis(axis=axis, ndim=len(shape))

    if len(axis) != 2:
        raise ValueError(
            f"The number of interpolation axes, {len(axis)}, is not supported."
        )

    # `a` is copied after it is broadcast, so that each slice has its own
    # memory to be filled, and `where` is a read-only view, which, unlike the
    # views made by `np.broadcast_arrays`, does not warn when numba inspects it.
    a = np.broadcast_to(a, shape, subok=True).copy()
    where = np.broadcast_to(where, shape)

    if guess is None:
        guess = _guess_median(a=a, where=where, axis=axis)

    a[where] = np.broadcast_to(guess, a.shape)[where]

    # Non-finite elements are unknown, so they are not used as neighbors of
    # the elements being filled.
    # This includes the elements being filled whose guess is not finite,
    # which become known once they are given a value by their neighbors.
    valid = np.isfinite(a)

    axis_numba = ~np.arange(len(axis))[::-1]

    shape_numba = tuple(shape[ax] for ax in axis)

    a = np.moveaxis(a, axis, axis_numba)
    where = np.moveaxis(where, axis, axis_numba)
    valid = np.moveaxis(valid, axis, axis_numba)

    shape_moved = a.shape

    a = a.reshape(-1, *shape_numba)
    where = where.reshape(-1, *shape_numba)
    valid = valid.reshape(-1, *shape_numba)

    result = _fill_gauss_seidel_2d(
        a=a,
        where=where,
        valid=valid,
        num_iterations=num_iterations,
    )

    result = result.reshape(shape_moved)
    result = np.moveaxis(result, axis_numba, axis)

    return result


def _guess_median(
    a: np.ndarray,
    where: np.ndarray,
    axis: tuple[int, ...],
) -> np.ndarray:
    """
    The median of the known elements of `a` along `axis`,
    the finite elements which are not being filled.

    Used as the default starting point of the relaxation.
    Slices with no known elements fall back to zero.
    """

    a = np.where(where | ~np.isfinite(a), np.nan, a)

    with warnings.catch_warnings():
        # slices with no valid elements are expected, and handled below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(a, axis=axis, keepdims=True)

    return np.where(np.isnan(result), 0, result)


@numba.njit(cache=True, parallel=True)
def _fill_gauss_seidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    valid: np.ndarray,
    num_iterations: int,
) -> np.ndarray:

    num_t, num_y, num_x = a.shape

    for t in numba.prange(num_t):
        for k in range(num_iterations):
            for is_odd in [False, True]:
                _iteration_gauss_seidel_2d(
                    a=a,
                    where=where,
                    valid=valid,
                    t=t,
                    num_x=num_x,
                    num_y=num_y,
                    is_odd=is_odd,
                )

    return a


@numba.njit(cache=True, fastmath=True)
def _iteration_gauss_seidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    valid: np.ndarray,
    t: int,
    num_x: int,
    num_y: int,
    is_odd: bool,
) -> None:
    """
    One half of a red-black Gauss-Seidel iteration.

    Each element being filled is replaced by the mean of its valid nearest
    neighbors, and is then valid itself.
    The elements are assumed to be equally spaced along both axes,
    and neighbors which are outside the array or not valid are left out,
    which is a zero-gradient (Neumann) boundary condition.
    """

    for j in range(num_y):
        for i in range(num_x):
            if (i + j) & 1 == is_odd:
                if where[t, j, i]:
                    total = 0.0
                    num = 0
                    if i > 0 and valid[t, j, i - 1]:
                        total += a[t, j, i - 1]
                        num += 1
                    if i < num_x - 1 and valid[t, j, i + 1]:
                        total += a[t, j, i + 1]
                        num += 1
                    if j > 0 and valid[t, j - 1, i]:
                        total += a[t, j - 1, i]
                        num += 1
                    if j < num_y - 1 and valid[t, j + 1, i]:
                        total += a[t, j + 1, i]
                        num += 1
                    if num > 0:
                        a[t, j, i] = total / num
                        valid[t, j, i] = True

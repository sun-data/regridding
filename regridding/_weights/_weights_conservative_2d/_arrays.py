"""
3-dimensional array manipulation routines.
"""

import numpy as np
import numba

__all__ = [
    "axis_x",
    "axis_y",
    "axes",
    "index_in_bounds",
    "index_flat",
    "index_2d",
    "align_axis_right",
]

axis_x = 0
axis_y = 1

axes = (axis_x, axis_y)

vector_unit = (
    (1, 0),
    (0, 1),
)


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def index_in_bounds(
    index: tuple[int, int],
    shape: tuple[int, int],
):
    """
    Check if a 2D index is within array bounds specified by `shape`.

    Parameters
    ----------
    index
        The 2D index to check.
    shape
        The shape of the array to use as the bounds.
    """
    i, j = index

    if i < 0:
        return False
    if j < 0:
        return False

    nx, ny = shape

    if i >= nx:
        return False
    if j >= ny:
        return False

    return True


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def index_flat(
    index: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    """
    Convert a 2D index to a flat index.

    Parameters
    ----------
    index
        A 2D index to convert.
    shape
        The sizes of each axis of the array.
    """

    i, j = index

    num_x, num_y = shape

    result = i * num_y + j

    return result


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def index_2d(
    index: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    """
    Convert a flat index to a 2D index.

    Parameters
    ----------
    index
        A flat index to convert.
    shape
        The sizes of each axis of the array.
    """

    num_x, num_y = shape

    j = index % num_y

    i = (index // num_y) % num_x

    return i, j


@numba.njit(
    cache=True,
    fastmath=True,
    inline="always",
)
def align_axis_right(
    a: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Roll all the axes of a 2D array to the right until `axis` is the last axis.

    Parameters
    ----------
    a
        The array to modify the axes of.
    axis
        The axis to set as the last axis.
    """
    axis = axis % a.ndim

    if axis == axis_x:
        result = a.transpose()
    else:
        result = a

    return result

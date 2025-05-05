"""
3-dimensional array manipulation routines.
"""

import numpy as np
import numba

__all__ = [
    "axis_x",
    "axis_y",
    "axis_z",
    "axes",
    "index_in_bounds",
    "index_flat",
    "index_3d",
    "align_axis_right",

]

axis_x = 0
axis_y = 1
axis_z = 2

axes = (axis_x, axis_y, axis_z)


@numba.njit(cache=True)
def index_in_bounds(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
):
    """
    Check if a 3D index is within array bounds specified by `shape`.

    Parameters
    ----------
    index
        The 3D index to check.
    shape
        The shape of the array to use as the bounds.
    """
    i, j, k = index

    if i < 0:
        return False
    if j < 0:
        return False
    if k < 0:
        return False

    nx, ny, nz = shape

    if i >= nx:
        return False
    if j >= ny:
        return False
    if k >= nz:
        return False

    return True


@numba.njit(cache=True)
def index_flat(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> int:
    """
    Convert a 3D index to a flat index.

    Parameters
    ----------
    index
        A 3D index to convert.
    shape
        The sizes of each axis of the array.
    """

    i, j, k = index

    num_x, num_y, num_z = shape

    result = i * num_y * num_z + j * num_z + k

    return result


@numba.njit(cache=True)
def index_3d(
    index: int,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Convert a flat index to a 3D index.

    Parameters
    ----------
    index
        A flat index to convert.
    shape
        The sizes of each axis of the array.
    """

    num_x, num_y, num_z = shape

    k = index % num_z

    j = (index // num_z) % num_y

    i = index // (num_z * num_y)

    return i, j, k


@numba.njit(cache=True)
def align_axis_right(
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

from __future__ import annotations
import math
import numpy as np
import numba

__all__ = [
    "ndarray_linear_interpolation",
]


def ndarray_linear_interpolation(
        a: np.ndarray,
        indices: tuple[np.ndarray, ...],
        axis: None | int | tuple[int] = None,
):

    if axis is None:
        axis = tuple(range(a.ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=a.ndim)

    if len(indices) != len(axis):
        raise ValueError(
            f"The number of coordinates, {len(indices)}, must match the number of elements in axis, {len(axis)}"
        )

    axis_orthogonal = tuple(ax for ax in range(a.ndim) if ax not in axis)
    shape_orthogonal = tuple(a.shape[ax] if ax in axis_orthogonal else 1 for ax in range(a.ndim))

    shape_result = np.broadcast_shapes(shape_orthogonal, *[ind.shape for ind in indices])

    indices = tuple(np.broadcast_to(ind, shape=shape_result) for ind in indices)

    result = np.empty(shape_result)
    for index in np.ndindex(*shape_orthogonal):

        index = list(index)
        for ax in axis:
            index[ax] = slice(None)
        index = tuple(index)

        if len(axis) == 1:

            x, = indices

            result[index] = _ndarray_linear_interpolation_1d(
                a=a[index],
                x=x[index],
            )

        elif len(axis) == 2:

            x, y = indices

            result[index] = _ndarray_linear_interpolation_2d(
                a=a[index],
                x=x[index],
                y=y[index],
            )

        else:
            raise NotImplementedError

    if a_unit is not None:
        result = result << a_unit

    return result


@numba.jit(nopython=True, parallel=True)
def _ndarray_linear_interpolation_1d(
        a: np.ndarray,
        x: np.ndarray,
) -> np.ndarray:

    shape_output = x.shape
    shape_output_x, = shape_output

    result = np.empty(shape_output)

    for i in numba.prange(shape_output_x):
        result[i] = _linear_interpolation(a, x, i)

    return result


@numba.jit(nopython=True, parallel=True)
def _ndarray_linear_interpolation_2d(
        a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:

    shape_output = x.shape
    shape_output_x, shape_output_y = shape_output

    result = np.empty(shape_output)

    for i in numba.prange(shape_output_x):
        for j in numba.prange(shape_output_y):
            result[i, j] = _bilinear_interpolation(a, x, y, i, j)

    return result


@numba.njit
def _linear_interpolation(
        a: np.ndarray,
        x: np.ndarray,
        i: int,
) -> float:

    shape_input_x, = a.shape

    x_i = x[i]
    x_0 = int(math.floor(x_i))

    if x_0 < 0:
        x_0 = 0
    elif x_0 > shape_input_x - 2:
        x_0 = shape_input_x - 2

    x_1 = x_0 + 1

    y_0 = a[x_0]
    y_1 = a[x_1]

    return y_0 + (x_i - x_0) * (y_1 - y_0)


@numba.njit
def _bilinear_interpolation(
        a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        i: int,
        j: int,
) -> float:

    shape_input_x, shape_input_y = a.shape

    x_ij = x[i, j]
    y_ij = y[i, j]

    x_00 = int(math.floor(x_ij))
    y_00 = int(math.floor(y_ij))

    if x_00 < 0:
        x_00 = 0
    elif x_00 > shape_input_x - 2:
        x_00 = shape_input_x - 2

    if y_00 < 0:
        y_00 = 0
    elif y_00 > shape_input_y - 2:
        y_00 = shape_input_y - 2

    x_01 = x_00
    x_10 = x_11 = x_00 + 1

    y_01 = y_11 = y_00 + 1
    y_10 = y_00

    a_00 = a[x_00, y_00]
    a_01 = a[x_01, y_01]
    a_10 = a[x_10, y_10]
    a_11 = a[x_11, y_11]

    dx_ij = x_ij - x_00
    dy_ij = y_ij - y_00

    w_00 = (1 - dx_ij) * (1 - dy_ij)
    w_01 = (1 - dx_ij) * dy_ij
    w_10 = dx_ij * (1 - dy_ij)
    w_11 = dx_ij * dy_ij

    return (a_00 * w_00) + (a_01 * w_01) + (a_10 * w_10) + (a_11 * w_11)




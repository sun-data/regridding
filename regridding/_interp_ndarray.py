from __future__ import annotations
import math
import numpy as np
import astropy.units as u
import numba

__all__ = [
    "ndarray_linear_interpolation",
]


def ndarray_linear_interpolation(
        a: np.ndarray,
        coordinates: tuple[np.ndarray],
        axis: None | int | tuple[int] = None,
):

    if isinstance(a, u.Quantity):
        a = a.value
        a_unit = a.unit
    else:
        a_unit = None

    if axis is None:
        axis = tuple(range(a.ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=a.ndim)
    print("axis", axis)

    if len(coordinates) != len(axis):
        raise ValueError(
            f"The number of coordinates, {len(coordinates)}, must match the number of elements in axis, {len(axis)}"
        )

    axis_orthogonal = tuple(ax for ax in range(a.ndim) if ax not in axis)
    print("axis_orthogonal", axis_orthogonal)
    shape_orthogonal = tuple(a.shape[ax] if ax in axis_orthogonal else 1 for ax in range(a.ndim))
    print("shape_orthogonal", shape_orthogonal)

    shape_result = np.broadcast_shapes(shape_orthogonal, *[coord.shape for coord in coordinates])
    print("shape_result", shape_result)

    coordinates = tuple(np.broadcast_to(coord, shape=shape_result) for coord in coordinates)

    result = np.empty(shape_result)
    for index in np.ndindex(*shape_orthogonal):

        index = list(index)
        for ax in axis:
            index[ax] = slice(None)
        index = tuple(index)

        if len(axis) == 1:

            x, = coordinates

            result[index] = _ndarray_linear_interpolation_1d(
                a=a[index],
                x=x[index],
            )

        elif len(axis) == 2:

            x, y = coordinates

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

    shape_x, = x.shape

    result = np.empty(x.shape)

    for i in numba.prange(shape_x):

        x_i = x[i]
        x_0 = int(math.floor(x_i))

        if x_0 < 0:
            x_0 = 0
        elif x_0 > shape_x - 2:
            x_0 = shape_x - 2

        x_1 = x_0 + 1

        y_0 = a[x_0]
        y_1 = a[x_1]

        result[i] = y_0 + (x_i - x_0) * (y_1 - y_0)

    return result


@numba.jit(nopython=True, parallel=True)
def _ndarray_linear_interpolation_2d(
        a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:

    shape_input_x, shape_input_y = a.shape
    shape_output_x, shape_output_y = x.shape

    result = np.empty((shape_output_x, shape_output_y))

    for i in numba.prange(shape_output_x):
        for j in numba.prange(shape_output_y):

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

            result[i, j] = (a_00 * w_00) + (a_01 * w_01) + (a_10 * w_10) + (a_11 * w_11)

    return result







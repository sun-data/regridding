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
        axis: None | int | tuple[int, ...] = None,
        axis_indices: None | int | tuple[int, ...] = None,
):
    """
    Interpolate a :class:`numpy.ndarray` onto a new grid.

    Similar to :func:`scipy.ndimage.map_coordinates`, but allows for interpolation along only some of the
    axes of ``a``.

    Parameters
    ----------
    a
        The input array to be interpolated. Should have at least as many axes as ``len(indices)``
    indices
        The new indices where ``a`` will be evaluated. The number of indices, ``len(indices)`` should be equal
        to the number of interpolation axes, ``len(axis)``
    axis
        The axes of ``a`` to interpolate. If :class:`None`, interpolate over all the axes of ``a``.
    axis_indices
        The axes of ``indices`` that are complementary to the axes in ``axis``.
        If :class:`None`, all axes of indices are complementary to the axes in ``axis``.

    Returns
    -------

        The interpolated array

    Examples
    --------

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        x = np.arange(5)
        y = np.arange(6)

    Interpolate a 1D array

    .. jupyter-execute::

        a = x * x

        x_new = np.linspace(0, x.size - 1, num=20)
        a_new = regridding.ndarray_linear_interpolation(a, indices=(x_new, ))

        plt.figure(figsize=(6, 3));
        plt.scatter(x, a, s=100, label="original", zorder=1);
        plt.scatter(x_new, a_new, label="interpolated", zorder=0);
        plt.legend();

    """

    shape_a = a.shape
    ndim_a = a.ndim

    indices_broadcasted = np.broadcast(*indices)
    shape_indices = indices_broadcasted.shape
    ndim_indices = indices_broadcasted.ndim

    if axis is None:
        axis = tuple(range(ndim_a))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=ndim_a)
    axis = tuple(~(~np.array(axis) % ndim_a))

    if axis_indices is None:
        axis_indices = tuple(range(ndim_indices))
    axis_indices = np.core.numeric.normalize_axis_tuple(axis_indices, ndim=ndim_indices)
    axis_indices = tuple(~(~np.array(axis_indices) % ndim_indices))

    if len(indices) != len(axis):
        raise ValueError(
            f"The number of indices, {len(indices)}, must match the number of elements in axis, {len(axis)}"
        )

    axis_orthogonal_a = tuple(ax for ax in range(-ndim_a, 0) if ax not in axis)
    axis_orthogonal_indices = tuple(ax for ax in range(-ndim_indices, 0) if ax not in axis_indices)

    shape_orthogonal_a = tuple(shape_a[ax] for ax in axis_orthogonal_a)
    shape_orthogonal_indices = tuple(shape_indices[ax] for ax in axis_orthogonal_indices)

    shape_orthogonal = np.broadcast_shapes(shape_orthogonal_a, shape_orthogonal_indices)

    ndim_broadcasted_a = len(shape_orthogonal) + len(axis)
    ndim_broadcasted_indices = len(shape_orthogonal) + len(axis_indices)

    shape_broadcasted_a = tuple(
        shape_a[ax] if ax in axis else shape_orthogonal[axis_orthogonal_a.index(ax)]
        for ax in range(-ndim_broadcasted_a, 0)
    )
    shape_broadcasted_indices = tuple(
        shape_indices[ax] if ax in axis_indices else shape_orthogonal[axis_orthogonal_indices.index(ax)]
        for ax in range(-ndim_broadcasted_indices, 0)
    )

    a = np.broadcast_to(a, shape_broadcasted_a)
    indices = tuple(np.broadcast_to(ind, shape_broadcasted_indices) for ind in indices)

    result = np.empty(shape_broadcasted_indices)

    def index_a_indices(index: tuple[int, ...]) -> tuple[tuple, tuple]:
        index_a = tuple(
            slice(None) if ax in axis else index[axis_orthogonal_a.index(ax)]
            for ax in range(-ndim_broadcasted_a, 0)
        )
        index_indices = tuple(
            slice(None) if ax in axis_indices else index[axis_orthogonal_indices.index(ax)]
            for ax in range(-ndim_broadcasted_indices, 0)
        )
        return index_a, index_indices

    if len(axis) == 1:

        x, = indices

        for index in np.ndindex(*shape_orthogonal):
            index_a, index_indices = index_a_indices(index)
            result[index_indices] = _ndarray_linear_interpolation_1d(
                a=a[index_a],
                x=x[index_indices].reshape(-1),
            ).reshape(x[index_indices].shape)

    elif len(axis) == 2:

        x, y = indices

        for index in np.ndindex(*shape_orthogonal):
            index_a, index_indices = index_a_indices(index)
            result[index_indices] = _ndarray_linear_interpolation_2d(
                a=a[index_a],
                x=x[index_indices].reshape(-1),
                y=y[index_indices].reshape(-1),
            ).reshape(x[index_indices].shape)

    else:
        raise NotImplementedError

    return result


@numba.jit(nopython=True, parallel=True)
def _ndarray_linear_interpolation_1d(
        a: np.ndarray,
        x: np.ndarray,
) -> np.ndarray:

    shape_output = x.shape
    size_output, = shape_output

    result = np.empty(size_output)

    for i in numba.prange(size_output):
        result[i] = _linear_interpolation(a, x[i])

    return result


@numba.jit(nopython=True, parallel=True)
def _ndarray_linear_interpolation_2d(
        a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
) -> np.ndarray:

    shape_output = x.shape
    size_output, = shape_output

    result = np.empty(size_output)

    for i in numba.prange(size_output):
        result[i] = _bilinear_interpolation(a, x[i], y[i])

    return result


@numba.njit
def _linear_interpolation(
        a: np.ndarray,
        x: float,
) -> float:

    shape_input_x, = a.shape

    x_0 = int(math.floor(x))

    if x_0 < 0:
        x_0 = 0
    elif x_0 > shape_input_x - 2:
        x_0 = shape_input_x - 2

    x_1 = x_0 + 1

    a_0 = a[x_0]
    a_1 = a[x_1]

    dx = x - x_0

    return a_0 * (1 - dx) + a_1 * dx


@numba.njit
def _bilinear_interpolation(
        a: np.ndarray,
        x: float,
        y: float,
) -> float:

    shape_input_x, shape_input_y = a.shape

    x_00 = int(math.floor(x))
    y_00 = int(math.floor(y))

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

    dx = x - x_00
    dy = y - y_00

    w_00 = (1 - dx) * (1 - dy)
    w_01 = (1 - dx) * dy
    w_10 = dx * (1 - dy)
    w_11 = dx * dy

    return (a_00 * w_00) + (a_01 * w_01) + (a_10 * w_10) + (a_11 * w_11)




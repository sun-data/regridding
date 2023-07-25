from __future__ import annotations

import dataclasses
from typing import Sequence
import numpy as np
import numba
import regridding._conservative_ramshaw

__all__ = [
    "regrid",
    "calc_weights",
    "regrid_from_weights",
]


def _normalize_axis(axis: None | int | tuple[int, ...], ndim: int):
    if axis is None:
        axis = tuple(range(ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=ndim)
    axis = tuple(~(~np.array(axis) % ndim))
    return axis


def regrid(
        vertices_input: tuple[np.ndarray, ...],
        vertices_output: tuple[np.ndarray, ...],
        values_input: np.ndarray,
        values_output: None | np.ndarray = None,
        axis_input: None | int | tuple[int, ...] = None,
        axis_output: None | int | tuple[int, ...] = None,
        method: str = "conservative",
        order: int = 1,
) -> np.ndarray:
    """
    Transfer a histogram defined on any curvilinear grid onto a new curvilinear grid.

    Parameters
    ----------
    vertices_input
        The vertices of each bin in the input histogram.
        The number of elements in ``vertices``, ``len(vertices)``,
        should match the number of regridding axes, ``len(axis)``.
        All elements of ``vertices`` should be broadcastable with the shape :math:`(...,M,...,N,...)`,
        where :math:`M` and :math:`N` are the number of elements along each regridding axis.
    vertices_output
        The vertices of each new bin in the output histogram.
        The number of elements in ``vertices``, ``len(vertices)``,
        should match the number of regridding axes, ``len(axis)``.
    values_input
        The value of each bin in the input histogram.
        Should be broadcastable with :math:`(...,M-1,...,N-1,...)`.
    values_output
        An alternative output array to place the result.
        It must have the same shape as the expected output.
    axis_input
        The axes of the input histogram to regrid.
        If :class:`None`, regrid over all the axes of the input histogram.
    axis_output
        The axes of ``vertices_new`` corresponding to the axes in ``axis``.
    method
        The type of regridding to use. Currently, the only valid option is ``conservative``.
    order
        The order of the regridding operation. Currently, only first-order regridding (``order=1``) is supported

    Returns
    -------
        The regridded histogram

    Examples
    --------
    Define an input curvilinear grid

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        x = np.linspace(-5, 5, num=66)
        y = np.linspace(-5, 5, num=66)

        x, y = np.meshgrid(x, y, indexing="ij")

        angle = 0.4
        x_input = x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x
        y_input = x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y

    Define a test patten

    .. jupyter-execute::

        pitch = 16
        a_input = 0 * x[:~0,:~0]
        a_input[::pitch, :] = 1
        a_input[:, ::pitch] = 1
        a_input[pitch//2::pitch, pitch//2::pitch] = 1

        plt.figure();
        plt.pcolormesh(x_input, y_input, a_input);

    Define a new grid

    .. jupyter-execute::

        x_output = 2 * x
        y_output = 2 * y

    Regrid the test pattern onto the new grid

    .. jupyter-execute::

        a_output = regridding.regrid(
            vertices_input=(x_input, y_input),
            vertices_output=(x_output, y_output),
            values_input=a_input,
        )

        plt.figure();
        plt.pcolormesh(x_output, y_output, a_output);
    """
    weights = calc_weights(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
        order=order,
    )
    result = regrid_from_weights(
        weights=weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    return result


def calc_weights(
        vertices_input: tuple[np.ndarray, ...],
        vertices_output: tuple[np.ndarray, ...],
        # values_input: np.ndarray,
        # values_output: None | np.ndarray = None,
        axis_input: None | int | tuple[int, ...] = None,
        axis_output: None | int | tuple[int, ...] = None,
        method: str = "conservative",
        order: int = 1,
) -> tuple[np.ndarray, tuple[int, ...]]:

    # shape_values_input = values_input.shape
    shape_vertices_input = np.broadcast(*vertices_input).shape
    shape_vertices_output = np.broadcast(*vertices_output).shape

    # ndim_values_input = len(shape_values_input)
    # ndim_vertices_input = len(shape_vertices_input)
    ndim_input = len(shape_vertices_input)
    ndim_output = len(shape_vertices_output)

    axis_input = _normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _normalize_axis(axis_output, ndim=ndim_output)

    axis_input = sorted(axis_input, reverse=True)
    axis_output = sorted(axis_output, reverse=True)

    if len(axis_output) != len(axis_input):
        raise ValueError(
            f"The number of axes in `axis_output`, {axis_output}, "
            f"must match the number of axes in `axis_input`, {axis_input}"
        )

    if len(vertices_input) != len(axis_input):
        raise ValueError(
            f"The number of elements in `vertices_input`, {len(vertices_input)}, "
            f"should match the number of axes in `axis_input`, {axis_input}"
        )

    if len(vertices_output) != len(vertices_input):

        raise ValueError(
            f"The number of elements in `vertices_output`, {len(vertices_output)}, "
            f"should match the number of elements in `vertices_input`, {len(vertices_input)}"
        )

    # shape_input = np.broadcast_shapes(
    #     tuple(1 if ax in axis_input else shape_values_input[ax] for ax in _normalize_axis(None, ndim_values_input)),
    #     shape_vertices_input,
    # )

    axis_input_orthogonal = tuple(ax for ax in _normalize_axis(None, ndim_input) if ax not in axis_input)
    axis_output_orthogonal = tuple(ax for ax in _normalize_axis(None, ndim_output) if ax not in axis_output)

    shape_input_orthogonal = tuple(shape_vertices_input[ax] for ax in axis_input_orthogonal)
    shape_output_orthogonal = tuple(shape_vertices_output[ax] for ax in axis_output_orthogonal)

    shape_orthogonal = np.broadcast_shapes(shape_input_orthogonal, shape_output_orthogonal)

    shape_input = list(shape_orthogonal)
    for ax in axis_input:
        shape_input.insert(ax, shape_vertices_input[ax])
    shape_input = tuple(shape_input)

    shape_output = list(shape_orthogonal)
    for ax in axis_output:
        shape_output.insert(ax, shape_vertices_output[ax])
    shape_output = tuple(shape_output)

    shape_centers_output = list(shape_output)
    for ax in axis_output:
        shape_centers_output[ax] -= 1
    shape_centers_output = tuple(shape_centers_output)

    # bshape_values_input = list(shape_orthogonal)
    # for ax in axis_input:
    #     bshape_values_input.insert(ax, shape_values_input[ax])
    # bshape_values_input = tuple(bshape_values_input)

    # if values_output is None:
    #     bshape_values_output = list(shape_orthogonal)
    #     for ax in axis_output:
    #         bshape_values_output.insert(ax, shape_output[ax] - 1)
    #     bshape_values_output = tuple(bshape_values_output)
    #
    #     values_output = np.zeros(bshape_values_output, dtype=values_input.dtype)

    vertices_input = tuple(np.broadcast_to(component, shape_input) for component in vertices_input)
    vertices_output = tuple(np.broadcast_to(component, shape_output) for component in vertices_output)
    # values_input = np.broadcast_to(values_input, bshape_values_input)

    weights = np.empty(shape_orthogonal, dtype=numba.typed.List)

    for index in np.ndindex(*shape_orthogonal):

        index_vertices_input = list(index)
        for ax in axis_input:
            index_vertices_input.insert(ax, slice(None))
        index_vertices_input = tuple(index_vertices_input)

        index_vertices_output = list(index)
        for ax in axis_output:
            index_vertices_output.insert(ax, slice(None))
        index_vertices_output = tuple(index_vertices_output)

        # index_values_input = list(index)
        # for ax in axis_input:
        #     index_values_input.insert(ax, slice(None))
        # index_values_input = tuple(index_values_input)

        # index_values_output = list(index)
        # for ax in axis_output:
        #     index_values_output.insert(ax, slice(None))
        # index_values_output = tuple(index_values_output)

        if len(axis_input) == 1:
            raise NotImplementedError("1D regridding not supported")

        elif len(axis_input) == 2:

            vertices_input_x, vertices_input_y = vertices_input
            vertices_output_x, vertices_output_y = vertices_output

            if method == "conservative":
                if order == 1:
                    weights[index] = regridding._conservative_ramshaw._conservative_ramshaw(
                        # values_input=values_input[index_values_input],
                        # values_output=values_output[index_values_output],
                        grid_input=(
                            vertices_input_x[index_vertices_input],
                            vertices_input_y[index_vertices_input],
                        ),
                        grid_output=(
                            vertices_output_x[index_vertices_output],
                            vertices_output_y[index_vertices_output],
                        ),
                    )
                else:
                    raise NotImplementedError(f"order {order} not supported")
            else:
                raise NotImplementedError(f"method {method} not supported")

        else:
            raise NotImplementedError("Regridding operations greater than 2D are not supported")

    return weights, shape_centers_output


def regrid_from_weights(
        weights: tuple[np.ndarray, tuple[int, ...]],
        values_input: np.ndarray,
        values_output: None | np.ndarray = None,
        axis_input: None | int | tuple[int, ...] = None,
        axis_output: None | int | tuple[int, ...] = None,
):
    weights, shape_centers_output = weights

    shape_weights = weights.shape

    shape_values_input = values_input.shape
    if values_output is not None:
        shape_values_output = values_output.shape
    else:
        shape_values_output = shape_centers_output

    ndim_input = len(shape_values_input)
    ndim_output = len(shape_values_output)

    axis_input = _normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _normalize_axis(axis_output, ndim=ndim_output)

    axis_input = sorted(axis_input, reverse=True)
    axis_output = sorted(axis_output, reverse=True)

    if len(axis_output) != len(axis_input):
        raise ValueError(
            f"The number of axes in `axis_output`, {axis_output}, "
            f"must match the number of axes in `axis_input`, {axis_input}"
        )

    axis_input_orthogonal = tuple(ax for ax in _normalize_axis(None, ndim_input) if ax not in axis_input)
    axis_output_orthogonal = tuple(ax for ax in _normalize_axis(None, ndim_output) if ax not in axis_output)

    shape_input_orthogonal = tuple(shape_values_input[ax] for ax in axis_input_orthogonal)
    shape_output_orthogonal = tuple(shape_values_output[ax] for ax in axis_output_orthogonal)

    shape_orthogonal = np.broadcast_shapes(shape_input_orthogonal, shape_output_orthogonal, shape_weights)

    weights = np.broadcast_to(weights, shape_orthogonal)

    shape_input = list(shape_orthogonal)
    for ax in axis_input:
        shape_input.insert(ax, shape_values_input[ax])
    shape_input = tuple(shape_input)

    values_input = np.broadcast_to(values_input, shape_input)

    shape_output = list(shape_orthogonal)
    for ax in axis_output:
        shape_output.insert(ax, shape_values_output[ax])
    shape_output = tuple(shape_output)

    if values_output is None:
        values_output = np.zeros(shape_output)
    else:
        if values_output.shape != shape_output:
            raise ValueError(
                f"provided output array has the wrong shape. Expected {shape_output}, got {values_output.shape}"
            )

    for index in np.ndindex(*shape_orthogonal):

        index_values_input = list(index)
        for ax in axis_input:
            index_values_input.insert(ax, slice(None))
        index_values_input = tuple(index_values_input)

        index_values_output = list(index)
        for ax in axis_output:
            index_values_output.insert(ax, slice(None))
        index_values_output = tuple(index_values_output)

        _regrid_from_weights_2d(
            weights=weights[index],
            values_input=values_input[index_values_input],
            values_output=values_output[index_values_output],
        )

    return values_output


@numba.njit(error_model="numpy")
def _regrid_from_weights_2d(
        weights: np.ndarray,
        values_input: np.ndarray,
        values_output: np.ndarray,
) -> None:

    values_input = values_input.reshape(-1)
    values_output = values_output.reshape(-1)

    for i in range(len(weights)):
        index_input, index_output, weight = weights[i]
        values_output[int(index_output)] += weight * values_input[int(index_input)]


@dataclasses.dataclass
class Regridder:

    index_input: np.ndarray
    index_output: np.ndarray
    shape_input: tuple[int, ...]
    shape_output: tuple[int, ...]
    weights: np.ndarray





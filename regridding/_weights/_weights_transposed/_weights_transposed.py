from typing import Any
import numpy as np
import numba
from ... import _util
from ..._cuda import on_device
from .._weights_conservative_1d._grids import cell_length
from .._weights_conservative_2d._grids import grid_volume as cell_area
from ._cuda import transpose_weights_conservative_cuda

__all__ = [
    "transpose_weights",
    "transpose_weights_conservative",
]


def transpose_weights(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Transpose the sparse matrix of weights calculated by :func:`regridding.weights`.

    This function works by swapping the index arrays of each element,
    :math:`(i, j, w) \rightarrow (j, i, w)`, without copying the values.

    Transposed weights can be used with :func:`regridding.regrid_from_weights`
    to perform a transform in the opposite direction.

    Note that this transpose is not conservative:
    use :func:`regridding.transpose_weights_conservative` if the reverse
    transform needs to conserve the total of the resampled array.

    Weights in device memory are swapped the same way and stay there.  Their
    empty slots, which carry an index of ``-1`` on the input side, carry it
    on the output side once transposed, where
    :func:`regridding.regrid_from_weights` skips them too.

    Parameters
    ----------
    weights
        Array of weights computed by :func:`regridding.weights`.

    See Also
    --------
    :func:`regridding.transpose_weights_conservative`
    :func:`regridding.regrid_from_weights`
    """

    weights_array, shape_input, shape_output = weights

    shape = weights_array.shape
    flat = weights_array.reshape(-1)

    result = np.empty(flat.size, dtype=object)
    for d in range(flat.size):
        indices_input, indices_output, values = flat[d]
        result[d] = (indices_output, indices_input, values)

    return result.reshape(shape), shape_output, shape_input


def transpose_weights_conservative(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    coordinates_input: np.ndarray | tuple[np.ndarray, ...],
    coordinates_output: np.ndarray | tuple[np.ndarray, ...],
    axis_input: None | int | tuple[int, ...] = None,
    axis_output: None | int | tuple[int, ...] = None,
    weights_input: None | np.ndarray = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Transpose matrix of weights and normalize to be conservative.

    Similar to :func:`transpose_weights`,
    this function transposes the matrix of weights calculated by :func:`regridding.weights`.
    However, this function also normalizes the transposed weights by the volume
    of each cell, so that they conserve flux when used with
    :func:`regridding.regrid_from_weights` to perform an inverse transform.

    If `weights_input` was given to :func:`regridding.weights`, the transposed
    weights additionally *invert* that weighting, so that a round trip is free
    of any residual factor of `weights_input`. See the Notes below: inverting a
    spatially-varying `weights_input` and conserving flux are not the same
    operation.

    Parameters
    ----------
    weights
        Array of weights computed by :func:`regridding.weights`.
    coordinates_input
        Vertices of each cell in the input grid provided to :func:`~regridding.weights`.
        Each transposed weight will be `multiplied` by the volume
        of the corresponding cell in the input grid.
    coordinates_output
        Vertices of each cell in the output grid.
        Each transposed weight will be `divided` by the volume
        of the corresponding cell in the output grid.
    axis_input
        Logical axes of the input grid to resample.
        If :obj:`None`, resample all the axes of the input grid.
        The number of axes should be equal to the number of
        coordinates in the input grid.
    axis_output
        Logical axes of the output grid corresponding to the resampled axes
        of the input grid.
        If :obj:`None`, all the axes of the output grid correspond to resampled
        axes in the input grid.
        The number of axes should be equal to the number of
        coordinates in the output grid.
    weights_input
        An optional array of weights that were applied to the input values
        by :func:`regridding.weights`.
        Each transposed weight will be `divided` by the square of its
        corresponding input weight: once to remove the factor that the forward
        transform applied, and once more so that the transpose inverts that
        weighting.

    Notes
    -----
    Conserving flux and inverting `weights_input` coincide only where
    `weights_input` is constant. Inverting a spatially-varying `weights_input`
    divides each cell by its own weight, which does not preserve the total, so
    the round trip conserves flux exactly only when `weights_input` is
    :obj:`None` or constant.

    Independently of `weights_input`, a round trip reproduces the input values
    exactly only for a constant field, since resampling onto a grid whose cells
    do not align with the input grid mixes neighboring cells. The total is
    still conserved. Where `weights_input` varies, that mixing is amplified by
    the subsequent division, by an amount that grows with how sharply
    `weights_input` changes from cell to cell.

    Weights built with ``device="cuda"`` are transposed on the device, and
    the result is left there.  The volumes of the cells are computed on the
    host from the grids, which are small next to the weights, and divided
    there by the square of `weights_input`.  Each distinct row of them is
    sent to the device once.  The index arrays are reused rather than
    copied, as on the host, and the values are stored as
    :class:`numpy.float64`, as on the host.  Empty slots keep their index of
    ``-1``, now on the output side, where
    :func:`regridding.regrid_from_weights` skips it as well.

    Examples
    --------

    Regrid array of values onto new grid with precalculated weights,
    and then transform back with transposed weights.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define input grid
        x_input = np.linspace(-4, 4, num=11)
        y_input = np.linspace(-4, 4, num=11)
        x_input, y_input = np.meshgrid(x_input, y_input, indexing="ij")

        # Define rotated output grid
        angle = 0.2
        x_output = x_input * np.cos(angle) - y_input * np.sin(angle)
        y_output = x_input * np.sin(angle) + y_input * np.cos(angle)

        # Define arrays of values defined on the same grid
        values_input = np.zeros((10, 10))
        values_input[4, 4] = 1

        # Save regridding weights relating the input and output grids
        weights = regridding.weights(
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
            method="conservative",
        )

        # Regrid the first array of values using the saved weights
        values_output = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )

        # Transpose calculated weights
        weights_transposed = regridding.transpose_weights_conservative(
            weights,
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
        )

        # Regrid the regridded values back onto original grid using transposed weights.
        values_transposed = regridding.regrid_from_weights(
            *weights_transposed,
            values_input=values_output,
        )

        # Plot the original and regridded arrays of values
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs[0].pcolormesh(x_input, y_input, values_input, vmin=0, vmax=1);
        axs[0].set_title(r"original");
        axs[1].pcolormesh(x_output, y_output, values_output, vmin=0, vmax=1);
        axs[1].set_title(r"rotated");
        axs[2].pcolormesh(x_input, y_input, values_transposed, vmin=0, vmax=1);
        axs[2].set_title(r"rotated and tranposed");
        axs[0].set_aspect("equal");
        axs[1].set_aspect("equal");
        axs[2].set_aspect("equal");
    """

    weights_array, shape_input, shape_output = weights

    (
        coordinates_input,
        coordinates_output,
        axis_input,
        axis_output,
        shape_coordinates_input,
        shape_coordinates_output,
        shape_orthogonal,
    ) = _util._normalize_input_output_coordinates(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    # `_normalize_input_output_coordinates` returns the axes in descending
    # order, but the flat cell indices stored in `weights` (and used by
    # `regrid_from_weights`) are built in ascending-axis order. Sort ascending
    # so that `weights_input` and the cell volumes are flattened in the same
    # layout the stored indices address; otherwise a spatially-varying
    # `weights_input` (or non-uniform grid) is applied to the wrong cells.
    axis_input = tuple(sorted(axis_input))
    axis_output = tuple(sorted(axis_output))

    # the weights can have more orthogonal elements than the grids do, when
    # `convolve_weights` folds in a kernel which varies along an orthogonal
    # axis the grids are broadcast along, so the grids are broadcast to the
    # weights rather than the other way around
    shape_orthogonal = np.broadcast_shapes(shape_orthogonal, weights_array.shape)
    weights_array = np.broadcast_to(weights_array, shape_orthogonal)

    # the two grids have been broadcast against each other, so the volumes of
    # each are computed only along the orthogonal axes it varies along, and a
    # grid shared by every element of the other is computed once
    volume_input = _volume(coordinates_input, shape_coordinates_input, axis_input)
    volume_output = _volume(coordinates_output, shape_coordinates_output, axis_output)

    # Divide by the input weight twice: once to remove the weight that the
    # forward transform multiplied into the values, and again so the
    # transpose *inverts* that weighting (retaining a factor of
    # ``1 / weights_input``). This makes the round trip recover the original
    # input values. Both apply to a whole input cell, so they are folded into
    # its volume once rather than applied to every weight which touches it.
    factor_input = volume_input
    if weights_input is not None:
        weights_input = np.asarray(_number(weights_input), dtype=np.float64)
        weights_input = _cells(
            _unbroadcast(
                np.broadcast_to(weights_input, shape_input),
                shape=weights_input.shape,
                axis=axis_input,
            ),
            axis=axis_input,
        )
        factor_input = factor_input / np.square(weights_input)

    # broadcast rather than copied, so that rows which are the same row of a
    # grid share their memory
    factor_input = np.broadcast_to(
        factor_input,
        shape_orthogonal + factor_input.shape[~0:],
    )
    volume_output = np.broadcast_to(
        volume_output,
        shape_orthogonal + volume_output.shape[~0:],
    )

    if on_device(weights_array):
        result, outside = transpose_weights_conservative_cuda(
            weights=weights_array,
            factor_input=factor_input,
            volume_output=volume_output,
        )
        if outside:
            raise ValueError(_message_outside(shape_input, shape_output))
        return result, shape_output, shape_input

    result = np.empty(shape_orthogonal, dtype=object)
    for index in np.ndindex(*shape_orthogonal):
        indices_input, indices_output, values = weights_array[index]

        factor_input_index = factor_input[index]
        volume_output_index = volume_output[index]

        # an index past the end of a grid means the grids are not the ones
        # the weights were built on, which `numpy` would report only as an
        # `IndexError`
        if np.size(indices_input):
            if (
                np.max(indices_input) >= factor_input_index.size
                or np.max(indices_output) >= volume_output_index.size
            ):
                raise ValueError(_message_outside(shape_input, shape_output))

        values = np.array(_number(values), dtype=np.float64)
        values = (
            values
            * factor_input_index[indices_input]
            / volume_output_index[indices_output]
        )

        result[index] = (indices_output, indices_input, values)

    return result, shape_output, shape_input


def _message_outside(
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
) -> str:
    """
    Explain a weight which addresses a cell outside the grids.

    Parameters
    ----------
    shape_input
        The shape of the input grid the weights were built for.
    shape_output
        The shape of the output grid the weights were built for.
    """
    return (
        f"the weights address cells outside the grids given; they were built "
        f"for an input of {shape_input} and an output of {shape_output}, and "
        f"`coordinates_input` and `coordinates_output` should be the grids "
        f"given to `regridding.weights()`"
    )


def _number(a: Any) -> Any:
    """
    Reduce a quantity to the number it stands for if it is dimensionless,
    or else to its value.

    :func:`regridding.weights` keeps the unit of `weights_input` on the
    weights it builds on the host, and reduces a dimensionless one to its
    number on a device.  Reducing both `weights_input` and the weights here
    makes a percentage, say, cancel in either case.  A unit with dimensions
    is dropped, as it always has been.

    Parameters
    ----------
    a
        An array, which may be an :class:`astropy.units.Quantity`.
    """
    unit = getattr(a, "unit", None)
    if unit is None:
        return a
    value = getattr(a, "value")
    try:
        return value * unit.to("")
    except (TypeError, ValueError):
        return value


def _unbroadcast(
    a: np.ndarray,
    shape: tuple[int, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    """
    Undo broadcasting an array along its orthogonal axes.

    Every orthogonal axis which the array was broadcast along is reduced to
    a length of one, as a view.  The resampled axes are left whole, since
    an array defined on the cells is needed on all of them.

    Parameters
    ----------
    a
        An array which has been broadcast.
    shape
        The shape of the array before it was broadcast, matched to the
        shape of `a` from the right.
    axis
        The resampled axes of `a`.
    """
    shape = (1,) * (a.ndim - len(shape)) + tuple(shape)
    resampled = [ax % a.ndim for ax in axis]
    index = tuple(
        slice(0, 1) if num == 1 and i not in resampled else slice(None)
        for i, num in enumerate(shape)
    )
    return a[index]


def _cells(
    a: np.ndarray,
    axis: tuple[int, ...],
) -> np.ndarray:
    """
    Arrange an array defined on the cells of a grid with the orthogonal axes
    first, followed by one axis of the cells.

    The cells are in the order the flat cell indices of a set of weights
    address them.

    Parameters
    ----------
    a
        An array defined on the cells of a grid.
    axis
        The resampled axes of the grid, in ascending order.
    """
    axis_numba = ~np.arange(len(axis))[::-1]
    a = np.moveaxis(a, axis, axis_numba)
    shape_orthogonal = a.shape[: a.ndim - len(axis)]
    return a.reshape(shape_orthogonal + (-1,))


def _volume(
    grid: tuple[np.ndarray, ...],
    shape: tuple[int, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    """
    Compute the volume of each cell of a grid which has been broadcast
    against another, arranged by :func:`_cells`.

    The volumes are computed only along the orthogonal axes the grid varies
    along, and have a length of one along the others.

    Parameters
    ----------
    grid
        The vertices of the grid, broadcast against the other grid.
    shape
        The shape of the grid before it was broadcast.
    axis
        The resampled axes of the grid, in ascending order.
    """
    grid = tuple(_unbroadcast(c, shape=shape, axis=axis) for c in grid)
    return _cells(_cell_volume(grid, axis), axis=axis)


def _cell_volume(
    grid: tuple[np.ndarray, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    """
    Compute the :math:`n`-dimensional volume of each cell in a grid.

    Parameters
    ----------
    grid
        1- or 2-dimensional grid.
    axis
        The resampled axes of the grid.
    """

    shape = grid[0].shape

    axis_numba = ~np.arange(len(axis))[::-1]

    shape_numba = tuple(shape[ax] for ax in axis)

    if len(grid) == 1:
        (x,) = grid
        x = np.moveaxis(x, axis, axis_numba)
        x_ = np.reshape(x, (-1,) + shape_numba)
        result = _cell_volume_1d(grid=(x_,))
        result = np.reshape(result, x.shape[:-1] + result.shape[-1:])
        result = np.moveaxis(result, axis_numba, axis)

    elif len(grid) == 2:
        x, y = grid
        x = np.moveaxis(x, axis, axis_numba)
        y = np.moveaxis(y, axis, axis_numba)
        x_ = np.reshape(x, (-1,) + shape_numba)
        y_ = np.reshape(y, (-1,) + shape_numba)
        if x_.shape[0] < _num_grids_across:
            result = _cell_volume_2d(grid=(x_, y_))
        else:
            result = _cell_volume_2d_across(grid=(x_, y_))
        result = np.reshape(result, x.shape[:-2] + result.shape[-2:])
        result = np.moveaxis(result, axis_numba, axis)

    else:  # pragma: nocover
        raise ValueError("Grids greater than 2D not supported.")

    return result


_num_grids_across = 8
"""
The number of 2D grids from which their areas are computed in parallel
across the grids, rather than one grid at a time in parallel within each.

Within a grid, every thread is used however few grids there are, but each
grid costs about 0.1 ms to start the threads on.  Across grids, a thread
does a whole grid at a time, so there is nothing to start for each one, but
only as many threads are used as there are grids.  On 48 threads, four grids
of a million cells take 29 ms within against 35 ms across, eight take 61 ms
against 49 ms, and 500 grids of four thousand cells take 71 ms against 5 ms.
"""


@numba.njit(
    cache=True,
    fastmath=True,
)
def _cell_volume_1d(
    grid: tuple[np.ndarray],
) -> np.ndarray:
    (x,) = grid

    shape_t, shape_x = x.shape

    result = np.empty((shape_t, shape_x - 1))

    for t in range(shape_t):
        result[t] = cell_length(x[t])

    return result


# `cell_area` is inlined, so its own parallel loop over the cells of one grid
# runs in parallel only in a caller which is compiled with `parallel=True`
@numba.njit(
    cache=True,
    fastmath=True,
    parallel=True,
)
def _cell_volume_2d(
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Compute the area of each cell of a stack of 2D grids, one grid at a
    time, in parallel within each.

    Parameters
    ----------
    grid
        The vertices of the grids, stacked along the first axis.
    """
    x, y = grid

    shape_t, shape_x, shape_y = x.shape

    result = np.empty((shape_t, shape_x - 1, shape_y - 1))

    for t in range(shape_t):
        result[t] = cell_area(grid=(x[t], y[t]))

    return result


# only the outermost parallel loop runs in parallel, so the one inside the
# inlined `cell_area` runs on one thread here, which is what is wanted
@numba.njit(
    cache=True,
    fastmath=True,
    parallel=True,
)
def _cell_volume_2d_across(
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Compute the area of each cell of a stack of 2D grids, in parallel across
    the grids.

    Parameters
    ----------
    grid
        The vertices of the grids, stacked along the first axis.
    """
    x, y = grid

    shape_t, shape_x, shape_y = x.shape

    result = np.empty((shape_t, shape_x - 1, shape_y - 1))

    for t in numba.prange(shape_t):
        result[t] = cell_area(grid=(x[t], y[t]))

    return result

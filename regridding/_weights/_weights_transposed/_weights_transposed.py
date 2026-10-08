import numpy as np
import numba
from ... import _util
from ..._cuda import on_device
from .._weights_conservative_1d._grids import cell_length
from ...geometry import area_triangle
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
    there by the square of `weights_input`.  They are sent to the device
    once, before they are broadcast along the orthogonal axes.  The index
    arrays are reused rather than copied, as on the host, and the values are
    stored as :class:`numpy.float64`, as on the host.  Empty slots keep their
    index of ``-1``, now on the output side, where
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
        _,
        _,
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

    # a grid which does not have the cells the weights were built for would
    # be read at the wrong cells, or past its end, without anything failing
    _check_grids(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=shape_input,
        shape_output=shape_output,
    )

    # the weights can have more orthogonal elements than the grids do, when
    # `convolve_weights` folds in a kernel which varies along an orthogonal
    # axis the grids are broadcast along, so the grids are broadcast to the
    # weights rather than the other way around
    try:
        shape_orthogonal = np.broadcast_shapes(shape_orthogonal, weights_array.shape)
    except ValueError as error:
        raise ValueError(
            f"the orthogonal axes of the grids, {shape_orthogonal}, cannot be "
            f"broadcast against those of the weights, {weights_array.shape}"
        ) from error
    weights_array = np.broadcast_to(weights_array, shape_orthogonal)

    factor_input, volume_output = _factors(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=shape_input,
        weights_input=weights_input,
    )
    factor_input = _cells(factor_input, axis_input)
    volume_output = _cells(volume_output, axis_output)

    if on_device(weights_array):
        result, outside = transpose_weights_conservative_cuda(
            weights=weights_array,
            factor_input=factor_input,
            volume_output=volume_output,
        )
        if outside:
            raise ValueError(
                f"the weights address cells outside the input of "
                f"{shape_input} and the output of {shape_output} they were "
                f"built for"
            )
        return result, shape_output, shape_input

    factor_input = np.broadcast_to(
        factor_input,
        shape_orthogonal + factor_input.shape[~0:],
    )
    volume_output = np.broadcast_to(
        volume_output,
        shape_orthogonal + volume_output.shape[~0:],
    )

    result = np.empty(shape_orthogonal, dtype=object)
    for index in np.ndindex(*shape_orthogonal):
        indices_input, indices_output, values = weights_array[index]

        values = _util._dimensionless(values, strict=False)
        values = (
            values
            * factor_input[index][indices_input]
            / volume_output[index][indices_output]
        )

        result[index] = (indices_output, indices_input, values)

    return result, shape_output, shape_input


def _factors(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: tuple[int, ...],
    axis_output: tuple[int, ...],
    shape_input: tuple[int, ...],
    weights_input: None | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute what the conservative transpose multiplies and divides each
    weight by: the volume of its input cell over the square of that cell's
    weight, and the volume of its output cell.

    Both are computed only along the orthogonal axes their grid varies
    along, so a grid shared by every element of the other is computed once,
    and are returned laid out as the grids are, with a length of one along
    the other orthogonal axes.

    Parameters
    ----------
    coordinates_input
        The vertices of the input grid, as normalized by
        :func:`regridding._util._normalize_input_output_coordinates`.
    coordinates_output
        The vertices of the output grid, normalized the same way.
    axis_input
        The resampled axes of the input grid, in ascending order.
    axis_output
        The resampled axes of the output grid, in ascending order.
    shape_input
        The shape of the cells of the input grid, which `weights_input` is
        broadcast to.
    weights_input
        The weights which were applied to the input values by
        :func:`regridding.weights`, if any.
    """
    factor_input = _cell_volume(
        _util._unbroadcast(coordinates_input, axis_input),
        axis_input,
    )
    volume_output = _cell_volume(
        _util._unbroadcast(coordinates_output, axis_output),
        axis_output,
    )

    # Divide by the input weight twice: once to remove the weight that the
    # forward transform multiplied into the values, and again so the
    # transpose *inverts* that weighting (retaining a factor of
    # ``1 / weights_input``). This makes the round trip recover the original
    # input values. Both apply to a whole input cell, so they are folded into
    # its volume once rather than applied to every weight which touches it.
    if weights_input is not None:
        weights_input = np.broadcast_to(weights_input, shape_input, subok=True)
        (weights_input,) = _util._unbroadcast((weights_input,), axis_input)
        weights_input = _util._dimensionless(weights_input, strict=False)
        # the factor is computed for every cell, but read only for those which
        # a weight touches, so a cell which none does may have a weight of
        # zero without anything being wrong.  A cell which one does still
        # warns, when its weights are scaled by the factor.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            factor_input = factor_input / np.square(weights_input)

    return factor_input, volume_output


def _check_grids(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: tuple[int, ...],
    axis_output: tuple[int, ...],
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
) -> None:
    """
    Check that the grids have the cells a set of weights was built for.

    Parameters
    ----------
    coordinates_input
        The vertices of the input grid.
    coordinates_output
        The vertices of the output grid.
    axis_input
        The resampled axes of the input grid.
    axis_output
        The resampled axes of the output grid.
    shape_input
        The shape of the input the weights were built for.
    shape_output
        The shape of the output the weights were built for.

    Raises
    ------
    ValueError
        If either grid has more resampled axes than the weights have axes,
        or a different number of cells along any of its resampled axes than
        the weights address, which is also the case for weights which
        address the vertices of the grids rather than their cells, such as
        multilinear ones.
    """
    hint = (
        "`coordinates_input`, `coordinates_output`, `axis_input` and "
        "`axis_output` should be the grids and axes which were given to "
        '`regridding.weights()`, with `method="conservative"`, for these '
        "weights, with input and output swapped if the weights have been "
        "transposed since"
    )

    if len(axis_input) > len(shape_input) or len(axis_output) > len(shape_output):
        raise ValueError(
            f"the grids have {len(axis_input)} input and {len(axis_output)} "
            f"output resampled axes, more than the input of {shape_input} and "
            f"the output of {shape_output} the weights were built for have; "
            f"{hint}"
        )

    cells_input = tuple(coordinates_input[0].shape[ax] - 1 for ax in axis_input)
    cells_output = tuple(coordinates_output[0].shape[ax] - 1 for ax in axis_output)
    expected_input = tuple(shape_input[ax] for ax in axis_input)
    expected_output = tuple(shape_output[ax] for ax in axis_output)
    if cells_input != expected_input or cells_output != expected_output:
        raise ValueError(
            f"the weights address {expected_input} input and {expected_output} "
            f"output elements along their resampled axes, but the grids have "
            f"{cells_input} and {cells_output} cells; {hint}"
        )


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
    num_cells = int(np.prod(a.shape[a.ndim - len(axis) :], dtype=int))
    return a.reshape(shape_orthogonal + (num_cells,))


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
        result = _cell_volume_2d(grid=(x_, y_))
        result = np.reshape(result, x.shape[:-2] + result.shape[-2:])
        result = np.moveaxis(result, axis_numba, axis)

    else:  # pragma: nocover
        raise ValueError("Grids greater than 2D not supported.")

    return result


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


@numba.njit(
    cache=True,
    fastmath=True,
    parallel=True,
)
def _cell_volume_2d(
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Compute the area of each cell of a stack of 2D grids.

    The area of a cell is the sum of the signed areas of the triangles which
    its edges form with the origin, as
    :func:`~regridding._weights._weights_conservative_2d._grids.grid_volume`
    computes it for one grid, and in the same order.  The edges along each
    axis are swept in turn, and a line of edges does not touch a cell of any
    other line, in the same grid or another, so each sweep is one parallel
    loop over every line of every grid.  Many small grids then cost no more
    to start than one large one, and a few large grids still use every
    thread.

    Parameters
    ----------
    grid
        The vertices of the grids, stacked along the first axis.
    """
    x, y = grid

    num_t, num_x, num_y = x.shape

    result = np.zeros((num_t, num_x - 1, num_y - 1))

    # the edges which run along the first axis, one line of them for each
    # row of cells
    for k in numba.prange(num_t * (num_x - 1)):
        t = k // (num_x - 1)
        i = k - t * (num_x - 1)
        for j in range(num_y):
            area = area_triangle(
                (y[t, i, j], x[t, i, j]),
                (y[t, i + 1, j], x[t, i + 1, j]),
            )
            if j >= 1:
                result[t, i, j - 1] += area
            if j < num_y - 1:
                result[t, i, j] -= area

    # the edges which run along the second axis, one line of them for each
    # column of cells
    for k in numba.prange(num_t * (num_y - 1)):
        t = k // (num_y - 1)
        j = k - t * (num_y - 1)
        for i in range(num_x):
            area = area_triangle(
                (x[t, i, j], y[t, i, j]),
                (x[t, i, j + 1], y[t, i, j + 1]),
            )
            if i >= 1:
                result[t, i - 1, j] += area
            if i < num_x - 1:
                result[t, i, j] -= area

    return result

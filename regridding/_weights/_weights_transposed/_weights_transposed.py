import numpy as np
import numba
from ... import _util
from .._weights_conservative_1d._grids import cell_length
from .._weights_conservative_2d._grids import grid_volume as cell_area

__all__ = [
    "transpose_weights",
    "transpose_weights_conservative",
]


def transpose_weights(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Transpose the sparse matrix of weights calculated by :func:`regridding.weights`.

    This function works by swapping the indices, :math:`(i, j, w) \rightarrow (j, i, w)`.

    Transposed weights can be used with :func:`regridding.regrid_from_weights`
    to perform a transform in the opposite direction.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`regridding.weights`.
    """

    weights, shape_input, shape_output = weights

    shape = weights.shape

    weights = numba.typed.List(weights.reshape(-1))

    result = _transpose_weights_numba(weights)

    result = np.fromiter(result, dtype=object)

    result = result.reshape(shape)

    return result, shape_output, shape_input


@numba.njit(
    cache=True,
    parallel=True,
)
def _transpose_weights_numba(
    weights: numba.typed.List,
) -> numba.typed.List:

    result = numba.typed.List()

    for d in numba.prange(len(weights)):
        d = numba.types.int64(d)
        weights_d = weights[d]
        result_d = numba.typed.List()
        for w in range(len(weights_d)):
            i_input, i_output, weight = weights_d[w]
            result_d.append((i_output, i_input, weight))
        result.append(result_d)

    return result


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
    However, this function also applies the appropriate normalization to the
    transposed weights such that they conserve flux when used with
    :func:`regridding.regrid_from_weights` to perform an inverse transform.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`regridding.weights`.
    coordinates_input
        Vertices of each cell in the input grid provided to :func:`~regridding.weights`.
        If provided, each transposed weight will be `multiplied` by the volume
        of the corresponding cell in the input grid.
    coordinates_output
        Vertices of each cell in the output grid.
        If provided, each transposed weight will be `divided` by the volume
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
        If provided, each transposed weight will be `divided` by its corresponding
        input weight.

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
    """

    weights, shape_input, shape_output = weights

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

    axis_numba_input = ~np.arange(len(axis_input))[::-1]
    axis_numba_output = ~np.arange(len(axis_output))[::-1]

    size_orthogonal = int(np.prod(shape_orthogonal))

    if weights_input is not None:
        weights_input = np.broadcast_to(weights_input, shape_input)
        weights_input = np.moveaxis(weights_input, axis_input, axis_numba_input)
        weights_input = weights_input.reshape(size_orthogonal, -1)

    if len(coordinates_input) == 1:
        volume_input = cell_length(coordinates_input[0])
    elif len(coordinates_input) == 2:
        volume_input = cell_area(coordinates_input)
    else:  # pragma: nocover
        raise ValueError("Coordinates greater than 2D not supported.")
    volume_input = np.moveaxis(volume_input, axis_input, axis_numba_input)
    volume_input = volume_input.reshape(size_orthogonal, -1)

    if len(coordinates_output) == 1:
        volume_output = cell_length(coordinates_output[0])
    elif len(coordinates_output) == 2:
        volume_output = cell_area(coordinates_output)
    else:  # pragma: nocover
        raise ValueError("Coordinates greater than 2D not supported.")
    volume_output = np.moveaxis(volume_output, axis_output, axis_numba_output)
    volume_output = volume_output.reshape(size_orthogonal, -1)

    shape = weights.shape

    weights = numba.typed.List(weights.reshape(-1))

    result = _transpose_weights_conservative_numba(
        weights,
        weights_input=weights_input,
        volume_input=volume_input,
        volume_output=volume_output,
    )

    result = np.fromiter(result, dtype=object)

    result = result.reshape(shape)

    return result, shape_output, shape_input


@numba.njit(
    cache=True,
    fastmath=True,
    parallel=True,
)
def _transpose_weights_conservative_numba(
    weights: numba.typed.List,
    volume_input: np.ndarray,
    volume_output: np.ndarray,
    weights_input: None | np.ndarray,
) -> numba.typed.List:
    result = numba.typed.List()

    for d in numba.prange(len(weights)):
        d = numba.types.int64(d)
        weights_d = weights[d]
        result_d = numba.typed.List()
        for w in range(len(weights_d)):
            i_input, i_output, weight = weights_d[w]

            if weights_input is not None:
                weight = weight / weights_input[d][i_input]

            weight = weight * volume_input[d][i_input] / volume_output[d][i_output]

            result_d.append((i_output, i_input, weight))

        result.append(result_d)

    return result

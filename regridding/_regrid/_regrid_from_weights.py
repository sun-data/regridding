from typing import Sequence
import numpy as np
import numba
from regridding import _util

__all__ = [
    "regrid_from_weights",
]


def regrid_from_weights(
    weights: np.ndarray,
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray = None,
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
) -> np.ndarray:
    """
    Regrid an array of values using weights computed by
    :func:`regridding.weights`.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`regridding.weights`.
    shape_input
        Broadcasted shape of the input coordinates computed by :func:`regridding.weights`.
    shape_output
        Broadcasted shape of the output coordinates computed by :func:`regridding.weights`.
    values_input
        Input array of values to be resampled.
    values_output
        Optional array in which to place the output.
    axis_input
        Logical axes of the input array to resample.
        If :obj:`None`, resample all the axes of the input array.
        The number of axes should be equal to the number of
        coordinates in the original input grid passed to :func:`regridding.weights`.
    axis_output
        Logical axes of the output array corresponding to the resampled axes
        of the input array.
        If :obj:`None`, all the axes of the output array correspond to resampled
        axes in the input grid.
        The number of axes should be equal to the original number of
        coordinates in the output grid passed to :func:`regridding.weights`.

    See Also
    --------
    :func:`regridding.regrid`
    :func:`regridding.regrid_from_weights`
    """

    shape_input = np.broadcast_shapes(shape_input, values_input.shape)
    values_input = np.broadcast_to(values_input, shape=shape_input, subok=True)
    ndim_input = len(shape_input)
    axis_input = _util._normalize_axis(axis_input, ndim=ndim_input)

    if values_output is None:
        shape_output = np.broadcast_shapes(
            shape_output,
            tuple(
                shape_input[ax] if ax not in axis_input else 1
                for ax in _util._normalize_axis(None, ndim_input)
            ),
        )
        values_output = np.zeros_like(values_input, shape=shape_output)
    else:
        if values_output.shape != shape_output:
            raise ValueError(f"")
        values_output.fill(0)

    ndim_output = len(shape_output)
    axis_output = _util._normalize_axis(axis_output, ndim=ndim_output)

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    values_input = np.moveaxis(values_input, axis_input, axis_input_numba)
    values_output = np.moveaxis(values_output, axis_output, axis_output_numba)

    shape_output_tmp = values_output.shape

    weights = numba.typed.List(weights.reshape(-1))
    values_input = values_input.reshape(-1, *shape_input_numba)
    values_output = values_output.reshape(-1, *shape_output_numba)

    values_input = np.ascontiguousarray(values_input)
    values_output = np.ascontiguousarray(values_output)

    _regrid_from_weights(
        weights=weights,
        values_input=values_input,
        values_output=values_output,
    )

    values_output = values_output.reshape(*shape_output_tmp)

    values_output = np.moveaxis(values_output, axis_output_numba, axis_output)

    return values_output


@numba.njit()
def _regrid_from_weights(
    weights: numba.typed.List,
    values_input: np.ndarray,
    values_output: np.ndarray,
) -> None:
    for d in numba.prange(len(weights)):
        weights_d = weights[d]
        values_input_d = values_input[d].reshape(-1)
        values_output_d = values_output[d].reshape(-1)

        for w in range(len(weights_d)):
            i_input, i_output, weight = weights_d[w]
            values_output_d[int(i_output)] += weight * values_input_d[int(i_input)]

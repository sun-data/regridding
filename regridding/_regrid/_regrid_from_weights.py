from typing import Sequence
import numpy as np
import numba
from numba.typed.typedlist import List as TypedList
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

    Since building the weights is much more expensive than applying them,
    this is the efficient way to resample many arrays defined on the same grid.
    See :func:`regridding.weights` for a complete example.

    Parameters
    ----------
    weights
        Array of weights computed by :func:`regridding.weights`, whose
        elements are ``(indices_input, indices_output, values)`` tuples of
        flat arrays.
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
    :func:`regridding.weights`
    """

    unit = getattr(values_input, "unit", None)

    ndim_input = len(shape_input)
    ndim_output = len(shape_output)

    axis_input = _util._normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _util._normalize_axis(axis_output, ndim=ndim_output)

    shape_input_orthogonal = tuple(
        shape_input[i]
        for i in _util._normalize_axis(None, ndim=len(shape_input))
        if i not in axis_input
    )
    shape_output_orthogonal = tuple(
        shape_output[i]
        for i in _util._normalize_axis(None, ndim=len(shape_output))
        if i not in axis_output
    )
    if np.ndim(values_input) > 0:
        shape_values_orthogonal = tuple(
            values_input.shape[i]
            for i in _util._normalize_axis(None, ndim=values_input.ndim)
            if i not in axis_input
        )
    else:
        shape_values_orthogonal = ()

    shape_orthogonal = np.broadcast_shapes(
        shape_input_orthogonal,
        shape_output_orthogonal,
        shape_values_orthogonal,
    )

    axis_input = tuple(sorted(axis_input))
    axis_output = tuple(sorted(axis_output))

    shape_input_new = list(reversed(shape_orthogonal))
    for ax in reversed(axis_input):
        shape_input_new.insert(~ax, shape_input[ax])
    shape_input = tuple(reversed(shape_input_new))

    shape_output_new = list(reversed(shape_orthogonal))
    for ax in reversed(axis_output):
        shape_output_new.insert(~ax, shape_output[ax])
    shape_output = tuple(reversed(shape_output_new))

    weights = np.broadcast_to(np.array(weights), shape_orthogonal, subok=True)
    values_input = np.broadcast_to(values_input, shape_input, subok=True)

    if values_output is None:
        values_output = np.zeros_like(values_input, shape=shape_output, dtype=float)
    else:
        if values_output.shape != shape_output:  # pragma: nocover
            raise ValueError(
                f"{values_output.shape=} should be equal to {shape_output}"
            )
        values_output.fill(0)

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    values_input = np.moveaxis(values_input, axis_input, axis_input_numba)
    values_output = np.moveaxis(values_output, axis_output, axis_output_numba)

    shape_output_tmp = values_output.shape

    values_input = values_input.reshape(-1, *shape_input_numba)
    values_output = values_output.reshape(-1, *shape_output_numba)

    flat = weights.reshape(-1)
    unit_weights = getattr(flat[0][2], "unit", None) if flat.size else None

    # `numba.typed.List()` is declared to return a plain `list` when Numba's
    # JIT is disabled, a mode this library is not usable in.
    weights_numba: TypedList = TypedList()  # type: ignore[assignment]
    for indices_input, indices_output, values in flat:
        weights_numba.append((indices_input, indices_output, np.asarray(values)))

    values_input = np.ascontiguousarray(values_input)
    values_output = np.ascontiguousarray(values_output)

    _regrid_from_weights(
        weights=weights_numba,
        values_input=values_input,
        values_output=values_output,
    )

    values_output = values_output.reshape(*shape_output_tmp)

    values_output = np.moveaxis(values_output, axis_output_numba, axis_output)

    if unit_weights is not None:
        unit = unit_weights if unit is None else unit * unit_weights

    if unit is None:
        return values_output

    return values_output << unit


@numba.njit(cache=True, parallel=True)
def _regrid_from_weights(
    weights: TypedList,
    values_input: np.ndarray,
    values_output: np.ndarray,
) -> None:

    for d_prange in numba.prange(len(weights)):
        # Calling a Numba type is a compile-time cast, but to a static checker
        # `numba.types.int64(...)` looks like the construction of a signature.
        d: int = numba.types.int64(d_prange)  # type: ignore[assignment]
        indices_input, indices_output, values = weights[d]
        values_input_d = values_input[d].reshape(-1)
        values_output_d = values_output[d].reshape(-1)
        for w in range(values.shape[0]):
            values_output_d[indices_output[w]] += (
                values[w] * values_input_d[indices_input[w]]
            )

from typing import NamedTuple, Sequence
import numpy as np
import numba
from numba import cuda
from numba.typed.typedlist import List as TypedList
from regridding import _util
from ._regrid_from_weights_cuda import regrid_from_weights_cuda

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

    Notes
    -----
    Weights built with ``device="cuda"`` are applied on the device and the
    result is left there, as a
    :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`.  Nothing has to
    be passed to ask for that: the weights are what say where the work
    happens.  A device array cannot have its axes moved or be broadcast the
    way the host path moves and broadcasts them, so the kernel addresses it
    by its strides instead, which comes to the same thing; it does have to
    be contiguous.

    Resampling many arrays onto one grid is a single call, since the axes
    the weights do not touch are broadcast over.  Giving it somewhere to
    put the answer is what makes that worth doing, since otherwise it
    allocates the whole result and clears it on every call::

        cube = numba.cuda.device_array((num, *shape_output), float)
        regridding.regrid_from_weights(
            *weights,
            values_input=scenes,      # (num, *shape_input), on the device
            values_output=cube,       # filled in place, allocated once
        )

    Twelve 1000 by 2000 images take 7.3 ms resampled one at a time into a
    fresh array each, 4.1 ms one at a time into slices of a cube, and 2.8 ms
    as the one call above.

    The scatter adds into the output atomically, so the order the
    contributions arrive in is not fixed and a result is reproducible only
    to rounding, around ``1e-16`` relative.  Two runs of the same call do
    not compare equal bit for bit.

    See Also
    --------
    :func:`regridding.regrid`
    :func:`regridding.weights`
    """

    unit = getattr(values_input, "unit", None)

    normalized = _normalize(
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    axis_input, axis_output, shape_orthogonal, shape_input, shape_output = normalized

    if _on_device(weights):
        return regrid_from_weights_cuda(
            weights=weights,
            normalized=normalized,
            values_input=values_input,
            values_output=values_output,
        )

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


class _Normalized(NamedTuple):
    """
    The axes and shapes a resampling operates on.

    These are worked out together and used together, by whichever path runs.

    Parameters
    ----------
    axis_input
        The resampled axes of the input, counted from the end and sorted.
    axis_output
        The resampled axes of the output, counted from the end and sorted.
    shape_orthogonal
        The shape of the axes which are not resampled.
    shape_input
        The shape of the input values, including the orthogonal axes.
    shape_output
        The shape of the output values, including the orthogonal axes.
    """

    axis_input: tuple[int, ...]
    axis_output: tuple[int, ...]
    shape_orthogonal: tuple[int, ...]
    shape_input: tuple[int, ...]
    shape_output: tuple[int, ...]


def _normalize(
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    values_input: np.ndarray,
    axis_input: None | int | Sequence[int],
    axis_output: None | int | Sequence[int],
) -> _Normalized:
    """
    Work out the axes and shapes the resampling operates on.

    This is only bookkeeping over tuples, so the host and the device paths
    share it, and each then arranges the values in whichever way its arrays
    allow.

    Parameters
    ----------
    shape_input
        The broadcasted shape of the input coordinates.
    shape_output
        The broadcasted shape of the output coordinates.
    values_input
        The values to resample.
    axis_input
        The axes of the input to resample.
    axis_output
        The axes of the output to resample into.

    Returns
    -------
    The normalized axes and shapes, where the two shapes carry the orthogonal
    axes as well as the resampled ones.
    """

    axis_input = _util._normalize_axis(axis_input, ndim=len(shape_input))
    axis_output = _util._normalize_axis(axis_output, ndim=len(shape_output))

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
    ndim_values = getattr(values_input, "ndim", 0)
    if ndim_values > 0:
        shape_values_orthogonal = tuple(
            values_input.shape[i]
            for i in _util._normalize_axis(None, ndim=ndim_values)
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

    return _Normalized(
        axis_input=axis_input,
        axis_output=axis_output,
        shape_orthogonal=shape_orthogonal,
        shape_input=shape_input,
        shape_output=shape_output,
    )


def _on_device(weights: np.ndarray) -> bool:
    """
    Test whether a set of weights lives in device memory.

    Parameters
    ----------
    weights
        Weights built by :func:`regridding.weights`.
    """
    flat = np.asarray(weights).reshape(-1)
    if not flat.size:  # pragma: nocover
        return False
    return cuda.is_cuda_array(flat[0][2])


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

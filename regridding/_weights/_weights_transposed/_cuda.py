"""
Transpose weights which are already in device memory, without bringing them
back.

The conservative transpose swaps the two index arrays of each weight and
scales its value by the volumes of the two cells it joins.  The swap needs
no work at all, and the scaling is a loop over independent slots, so the
values are computed on the device and the indices are reused where they are.

The volumes, and any `weights_input`, are defined on the cells of the grids
rather than on the weights, so they are computed on the host, where the
grids are, and sent to the device once for each distinct element of the
orthogonal axes.

This is reached by calling :func:`regridding.transpose_weights_conservative`
with weights built by :func:`regridding.weights` with ``device="cuda"``;
there is no separate function to call.
"""

from typing import Any
import numpy as np
from numba import cuda
from regridding import _cuda

__all__ = [
    "transpose_weights_conservative_cuda",
]


# the kernel below runs on the device, where `coverage` cannot follow it, so
# it reports its body as missed even when it does the work.  Its arrays are
# device arrays, which `numba` ships no type for, so they are annotated as
# `Any`
@cuda.jit
def _normalize(  # pragma: nocover
    indices_input: Any,
    indices_output: Any,
    values: Any,
    volume_input: Any,
    volume_output: Any,
    weights_input: Any,
    result: Any,
) -> None:
    """
    Scale each weight by the volumes of the cells it joins.

    The weight is multiplied by the volume of its input cell and divided by
    the volume of its output cell.  If `weights_input` is not empty, it is
    also divided by the square of its input cell's weight, in the same order
    as on the host so that the two agree to the last bit.

    Slots which saw no overlap carry an index of ``-1`` on one side or the
    other, and are given a weight of zero rather than being read.
    """
    w = cuda.grid(1)  # type: ignore[call-arg]
    if w >= values.size:
        return

    index_input = indices_input[w]
    index_output = indices_output[w]
    if index_input < 0 or index_output < 0:
        result[w] = 0
        return

    value = values[w]
    if weights_input.size:
        weight = weights_input[index_input]
        value = value / (weight * weight)

    result[w] = value * volume_input[index_input] / volume_output[index_output]


def _to_device(
    row: np.ndarray,
    cache: dict[int, Any],
) -> Any:
    """
    Send one row of an array defined on the cells of a grid to the device,
    unless an identical row has already been sent.

    The rows are views into an array broadcast along the orthogonal axes, so
    rows which are the same row of the grid share their memory, and that is
    what identifies them.

    Parameters
    ----------
    row
        The values on the cells of the grid, for one element of the
        orthogonal axes.
    cache
        The rows already sent, keyed by where their memory begins.
    """
    key = row.__array_interface__["data"][0]
    if key not in cache:
        cache[key] = cuda.to_device(np.ascontiguousarray(row, dtype=np.float64))
    return cache[key]


def transpose_weights_conservative_cuda(
    weights: np.ndarray,
    volume_input: np.ndarray,
    volume_output: np.ndarray,
    weights_input: None | np.ndarray = None,
    threads: int = _cuda.threads,
) -> np.ndarray:
    """
    Transpose weights which live in device memory, and normalize them to be
    conservative.

    The result is left on the device.  Its index arrays are those of
    `weights`, swapped rather than copied, as the host swaps them, and its
    values are new arrays of :class:`numpy.float64`, as the host computes
    them.  Empty slots keep their index of ``-1``, which is now on the output
    side, where :func:`regridding.regrid_from_weights` skips it as well.

    Parameters
    ----------
    weights
        Weights built by :func:`regridding.weights` with ``device="cuda"``,
        already broadcast to their orthogonal shape.
    volume_input
        The volume of each cell of the input grid, with the orthogonal axes
        of `weights` followed by one axis of the cells.
    volume_output
        The volume of each cell of the output grid, arranged as
        `volume_input`.
    weights_input
        The weights which were applied to the input values, arranged as
        `volume_input`, if any were.
    threads
        The number of threads in each block.
    """
    cache_input: dict[int, Any] = dict()
    cache_output: dict[int, Any] = dict()
    cache_weights: dict[int, Any] = dict()

    # an empty array stands for no `weights_input`, since a kernel is
    # compiled for the types of its arguments and `None` is a different one
    empty = _cuda.allocate(0, np.float64)

    result = np.empty(weights.shape, dtype=object)

    for index in np.ndindex(*weights.shape):

        indices_input, indices_output, values = weights[index]

        values_result = _cuda.allocate(values.size, np.float64)

        if values.size:
            blocks = (values.size + threads - 1) // threads
            _normalize[blocks, threads](  # type: ignore[index]
                indices_input,
                indices_output,
                values,
                _to_device(volume_input[index], cache_input),
                _to_device(volume_output[index], cache_output),
                (
                    empty
                    if weights_input is None
                    else _to_device(weights_input[index], cache_weights)
                ),
                values_result,
            )

        result[index] = (indices_output, indices_input, values_result)

    return result

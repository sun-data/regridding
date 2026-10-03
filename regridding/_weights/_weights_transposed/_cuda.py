"""
Transpose weights which are already in device memory, without bringing them
back.

The conservative transpose swaps the two index arrays of each weight and
scales its value by a factor for each of the two cells it joins.  The swap
needs no work at all, and the scaling is a loop over independent slots, so
the values are computed on the device and the indices are reused where they
are.

The factors are defined on the cells of the grids rather than on the
weights, so they are computed on the host, where the grids are, and sent to
the device once for each distinct row.

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
    factor_input: Any,
    volume_output: Any,
    outside: Any,
    result: Any,
) -> None:
    """
    Scale each weight by the factors of the cells it joins.

    The weight is multiplied by the factor of its input cell and divided by
    the volume of its output cell, in the same order as on the host, so
    that the two agree to the last bit.

    Slots which saw no overlap carry an index of ``-1`` on one side or the
    other, and are given a weight of zero rather than being read.  An index
    past the end of either grid means the grids are not the ones the
    weights were built on: it is flagged in `outside` rather than read.
    """
    w = cuda.grid(1)  # type: ignore[call-arg]
    if w >= values.size:
        return

    index_input = indices_input[w]
    index_output = indices_output[w]
    if index_input < 0 or index_output < 0:
        result[w] = 0
        return

    if index_input >= factor_input.size or index_output >= volume_output.size:
        outside[0] = 1
        result[w] = 0
        return

    result[w] = values[w] * factor_input[index_input] / volume_output[index_output]


def transpose_weights_conservative_cuda(
    weights: np.ndarray,
    factor_input: np.ndarray,
    volume_output: np.ndarray,
    threads: int = _cuda.threads,
) -> tuple[np.ndarray, bool]:
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
    factor_input
        The factor to multiply each weight by, for its input cell, with the
        orthogonal axes of `weights` followed by one axis of the cells.
    volume_output
        The volume of each cell of the output grid, arranged as
        `factor_input`.
    threads
        The number of threads in each block.

    Returns
    -------
    The transposed weights, and whether any of them addressed a cell
    outside the grids, which leaves them meaningless.
    """
    # the rows sent are kept until every element is done, since any later
    # element may share them.  That is at most one row of each grid for each
    # distinct element of the orthogonal axes it varies along, which is the
    # size of the grid rather than of the weights.
    cache_input: dict[Any, Any] = dict()
    cache_output: dict[Any, Any] = dict()

    # set on the device if any weight addresses a cell outside the grids,
    # and read once all the elements are done
    outside = _cuda.zeros(1, np.int64)

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
                _cuda.to_device_cached(factor_input[index], cache_input),
                _cuda.to_device_cached(volume_output[index], cache_output),
                outside,
                values_result,
            )

        result[index] = (indices_output, indices_input, values_result)

    return result, bool(outside.copy_to_host()[0])

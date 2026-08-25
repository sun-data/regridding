"""
Apply weights which are already in device memory, without bringing them back.

:func:`~regridding._regrid._regrid_from_weights.regrid_from_weights` gathers
each weight's input value and scatters it into the output, which is a loop
over independent slots and so suits a device as well as the building of the
weights does.  What it needs is an atomic add, since several weights land on
the same output cell.

This is reached by calling
:func:`regridding.regrid_from_weights` with weights built by
:func:`regridding.weights` with ``device="cuda"``; there is no separate
function to call.
"""

from typing import Any, Sequence
import numpy as np
from numba import cuda
from regridding import _util

__all__ = [
    "regrid_from_weights_cuda",
]


# the two kernels below run on the device, where `coverage` cannot follow
# them, so it reports their bodies as missed even when they do the work
@cuda.jit
def _scatter(  # pragma: nocover
    indices_input,
    indices_output,
    values,
    values_input,
    values_output,
):
    """
    Scatter each weight's contribution into the output.

    Slots which saw no overlap carry an index of ``-1`` and are skipped, so
    the weights do not have to be compacted first.

    Parameters
    ----------
    indices_input
        The flattened index of the input cell each weight gathers from.
    indices_output
        The flattened index of the output cell each weight scatters to.
    values
        The weights.
    values_input
        The flattened input values.
    values_output
        The flattened output values, added to in place.
    """
    w = cuda.grid(1)  # type: ignore[call-arg]
    if w >= values.size:
        return
    index_input = indices_input[w]
    if index_input < 0:
        return
    cuda.atomic.add(
        values_output,
        indices_output[w],
        values[w] * values_input[index_input],
    )


@cuda.jit
def _zero(a):  # pragma: nocover
    """Zero a device array, which is cheaper than sending one from the host."""
    i = cuda.grid(1)  # type: ignore[call-arg]
    if i < a.size:
        a[i] = 0


def _zeros(shape: tuple[int, ...], dtype: np.typing.DTypeLike, threads: int) -> Any:
    """
    Allocate a device array of zeros.

    Filling it with a kernel is cheaper than sending one from the host.  The
    annotations `numba` gives these two describe how the compiler calls them
    rather than how a caller does, so both are waived here rather than at
    each use.

    Parameters
    ----------
    shape
        The shape of the array.
    dtype
        The type of the array's elements.
    threads
        The number of threads in each block.
    """
    result = cuda.device_array(shape, dtype)  # type: ignore[arg-type]
    flat = result.reshape(-1)
    _zero[(flat.size + threads - 1) // threads, threads](flat)  # type: ignore[index]
    return result


def _is_trailing(axis: None | int | Sequence[int], ndim: int, num: int) -> bool:
    """
    Test whether `axis` names the last `num` axes of an array.

    The device path does not move axes around, since a device array cannot be
    transposed the way :func:`numpy.moveaxis` transposes a host one, so the
    resampled axes have to be the ones which are already contiguous.

    Parameters
    ----------
    axis
        The axes to test, in the form given to
        :func:`regridding.regrid_from_weights`.
    ndim
        The number of dimensions of the array.
    num
        The number of resampled axes.
    """
    normalized = _util._normalize_axis(axis, ndim=ndim)
    # `_normalize_axis` counts from the end, so the trailing axes are the
    # last `num` negative indices
    return tuple(sorted(int(a) for a in normalized)) == tuple(range(-num, 0))


def regrid_from_weights_cuda(
    weights: np.ndarray,
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    values_input: Any,
    values_output: None | Any = None,
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    threads: int = 256,
) -> Any:
    """
    Apply weights which live in device memory.

    The result is left on the device, as a
    :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`.  It exposes
    ``__cuda_array_interface__``, so :func:`torch.as_tensor` wraps it
    without copying.

    Parameters
    ----------
    weights
        Weights built by :func:`regridding.weights` with ``device="cuda"``.
    shape_input
        The broadcasted shape of the input coordinates.
    shape_output
        The broadcasted shape of the output coordinates.
    values_input
        The values to resample.  Anything exposing
        ``__cuda_array_interface__`` is used where it is; a host array is
        sent to the device first.
    values_output
        An optional device array to place the output in.
    axis_input
        The axes of the input array to resample, which have to be its last
        axes.
    axis_output
        The axes of the output array to resample into, which have to be its
        last axes.
    threads
        The number of threads in each block.

    Raises
    ------
    ValueError
        If the resampled axes are not the trailing axes, or if the shape of
        `values_input` is not the shape the weights were built for.
    """

    if getattr(values_input, "unit", None) is not None:
        raise ValueError(
            "an `astropy.units.Quantity` cannot be resampled on a device; "
            "pass its `.value` and reapply the unit to the result"
        )

    num_input = len(shape_input)
    num_output = len(shape_output)

    shape_orthogonal = np.asarray(weights).shape

    ndim_values_input = len(shape_orthogonal) + num_input
    if not _is_trailing(axis_input, ndim_values_input, num_input):
        raise ValueError(
            f"{axis_input=} has to name the last {num_input} axes of the "
            f"input values to resample on a device, since a device array "
            f"cannot have its axes moved"
        )
    if not _is_trailing(axis_output, len(shape_orthogonal) + num_output, num_output):
        raise ValueError(
            f"{axis_output=} has to name the last {num_output} axes of the "
            f"output values to resample on a device, since a device array "
            f"cannot have its axes moved"
        )

    expected = shape_orthogonal + shape_input
    if tuple(values_input.shape) != expected:
        raise ValueError(
            f"{values_input.shape=} has to be {expected} to resample on a "
            f"device, since a device array cannot be broadcast"
        )

    if not cuda.is_cuda_array(values_input):
        values_input = cuda.to_device(np.ascontiguousarray(values_input))

    flat = np.asarray(weights).reshape(-1)
    num_orthogonal = flat.size

    size_input = int(np.prod(shape_input, dtype=int))
    size_output = int(np.prod(shape_output, dtype=int))

    dtype = np.promote_types(
        np.dtype(values_input.dtype),
        np.dtype(flat[0][2].dtype),
    )

    if values_output is None:
        result = _zeros(shape_orthogonal + shape_output, dtype, threads)
    else:
        if tuple(values_output.shape) != shape_orthogonal + shape_output:
            raise ValueError(
                f"{values_output.shape=} should be equal to "
                f"{shape_orthogonal + shape_output}"
            )
        result = values_output

    values_input_flat = values_input.reshape(num_orthogonal, size_input)
    values_output_flat = result.reshape(num_orthogonal, size_output)

    for d in range(num_orthogonal):
        indices_input, indices_output, values = flat[d]
        blocks = (values.size + threads - 1) // threads
        _scatter[blocks, threads](  # type: ignore[index]
            indices_input,
            indices_output,
            values,
            values_input_flat[d],
            values_output_flat[d],
        )

    return result

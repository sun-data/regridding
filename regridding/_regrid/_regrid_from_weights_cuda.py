"""
Apply weights which are already in device memory, without bringing them back.

:func:`~regridding._regrid._regrid_from_weights.regrid_from_weights` gathers
each weight's input value and scatters it into the output, which is a loop
over independent slots and so suits a device as well as the building of the
weights does.  What it needs is an atomic add, since several weights land on
the same output cell.

A device array cannot be transposed or broadcast the way the host path
transposes and broadcasts its values, so the axes are not moved here: the
kernel addresses both sides through strides which the caller works out from
the shapes.  An axis being broadcast is one whose stride is zero, so that
falls out of the same arithmetic rather than needing a copy.  Addressing
this way rather than assuming the resampled axes are contiguous costs about
four percent, since a device resampling is always the 2D conservative one
and the index arithmetic is therefore a fixed pair of divisions.

This is reached by calling :func:`regridding.regrid_from_weights` with
weights built by :func:`regridding.weights` with ``device="cuda"``; there is
no separate function to call.
"""

from typing import Any
import numpy as np
from numba import cuda

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
    base_input,
    num_input,
    stride_input_0,
    stride_input_1,
    base_output,
    num_output,
    stride_output_0,
    stride_output_1,
):
    """
    Scatter each weight's contribution into the output.

    The indices address a flattened grid of the resampled axes, so each is
    split back into a pair and applied to the strides of those axes.  Slots
    which saw no overlap carry an index of ``-1`` and are skipped, so the
    weights do not have to be compacted first.
    """
    w = cuda.grid(1)  # type: ignore[call-arg]
    if w >= values.size:
        return

    index_input = indices_input[w]
    if index_input < 0:
        return

    i0 = index_input // num_input
    i1 = index_input - i0 * num_input

    index_output = indices_output[w]
    o0 = index_output // num_output
    o1 = index_output - o0 * num_output

    cuda.atomic.add(
        values_output,
        base_output + o0 * stride_output_0 + o1 * stride_output_1,
        values[w]
        * values_input[base_input + i0 * stride_input_0 + i1 * stride_input_1],
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
    annotations :mod:`numba` gives its allocation and its kernels describe
    how the compiler calls them rather than how a caller does, so they are
    waived here rather than at each use.

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


def _strides(
    shape: tuple[int, ...],
    shape_broadcast: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Compute the strides of a contiguous array broadcast to a larger shape.

    The strides are in elements rather than in bytes, since that is how the
    kernel indexes.  An axis which is being broadcast is given a stride of
    zero, which is what lets the kernel broadcast without a copy.

    Parameters
    ----------
    shape
        The shape of the array.
    shape_broadcast
        The shape it is being broadcast to, aligned with `shape` from the
        right as :func:`numpy.broadcast_to` aligns it.
    """

    strides_shape = []
    stride = 1
    for num in reversed(shape):
        strides_shape.append(stride)
        stride = stride * num
    strides_shape = list(reversed(strides_shape))

    offset = len(shape_broadcast) - len(shape)

    result = []
    for index, num in enumerate(shape_broadcast):
        index_source = index - offset
        if index_source < 0:
            result.append(0)
        elif shape[index_source] != num:
            result.append(0)
        else:
            result.append(strides_shape[index_source])

    return tuple(result)


def _split(
    strides: tuple[int, ...],
    shape: tuple[int, ...],
    axis: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Separate the resampled axes from the orthogonal ones.

    Parameters
    ----------
    strides
        The strides of an array of `shape`.
    shape
        The shape the strides belong to.
    axis
        The resampled axes, counted from the end as
        :func:`regridding._util._normalize_axis` counts them.

    Returns
    -------
    The shape and the strides of the resampled axes, and the strides of the
    orthogonal ones.
    """
    ndim = len(shape)
    resampled = tuple(a % ndim for a in axis)
    orthogonal = tuple(i for i in range(ndim) if i not in resampled)
    return (
        tuple(shape[i] for i in resampled),
        tuple(strides[i] for i in resampled),
        tuple(strides[i] for i in orthogonal),
    )


def regrid_from_weights_cuda(
    weights: np.ndarray,
    shape_orthogonal: tuple[int, ...],
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    values_input: Any,
    values_output: None | Any = None,
    axis_input: tuple[int, ...] = (),
    axis_output: tuple[int, ...] = (),
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
    shape_orthogonal
        The shape of the axes which are not resampled.
    shape_input
        The shape of the input values, including the orthogonal axes.
    shape_output
        The shape of the output values, including the orthogonal axes.
    values_input
        The values to resample.  Anything exposing
        ``__cuda_array_interface__`` is used where it is; a host array is
        broadcast and sent to the device first.
    values_output
        An optional device array to place the output in.
    axis_input
        The resampled axes of the input, normalized and sorted.
    axis_output
        The resampled axes of the output, normalized and sorted.
    threads
        The number of threads in each block.

    Raises
    ------
    ValueError
        If the values are an :class:`astropy.units.Quantity`, if a device
        array given for either side is not contiguous, or if `values_output`
        is not the shape the weights were built for.
    """

    if getattr(values_input, "unit", None) is not None:
        raise ValueError(
            "an `astropy.units.Quantity` cannot be resampled on a device; "
            "pass its `.value` and reapply the unit to the result"
        )

    if not cuda.is_cuda_array(values_input):
        # `numpy.array` rather than `numpy.ascontiguousarray`, which leaves an
        # axis of length one with the stride of zero that `numpy.broadcast_to`
        # gave it.  `numpy` counts that as contiguous and `numba` does not.
        values_input = cuda.to_device(
            np.array(np.broadcast_to(values_input, shape_input), order="C")
        )
    elif not values_input.is_c_contiguous():
        raise ValueError(
            "`values_input` has to be contiguous to be resampled on a device, "
            "since the kernel addresses it by its strides"
        )

    weights = np.broadcast_to(np.array(weights), shape_orthogonal, subok=True)
    flat_weights = weights.reshape(-1)

    dtype = np.promote_types(
        np.dtype(values_input.dtype),
        np.dtype(flat_weights[0][2].dtype),
    )

    if values_output is None:
        result = _zeros(shape_output, dtype, threads)
    else:
        if tuple(values_output.shape) != shape_output:
            raise ValueError(
                f"{values_output.shape=} should be equal to {shape_output}"
            )
        if not values_output.is_c_contiguous():
            raise ValueError(
                "`values_output` has to be contiguous to be resampled into on "
                "a device, since the kernel addresses it by its strides"
            )
        result = values_output

    shape_resampled_input, strides_input, strides_orthogonal_input = _split(
        _strides(tuple(values_input.shape), shape_input),
        shape_input,
        axis_input,
    )
    shape_resampled_output, strides_output, strides_orthogonal_output = _split(
        _strides(shape_output, shape_output),
        shape_output,
        axis_output,
    )

    values_input_flat = values_input.reshape(-1)
    values_output_flat = result.reshape(-1)

    num_input = shape_resampled_input[~0]
    num_output = shape_resampled_output[~0]

    for index, position in enumerate(np.ndindex(*shape_orthogonal)):

        base_input = sum(p * s for p, s in zip(position, strides_orthogonal_input))
        base_output = sum(p * s for p, s in zip(position, strides_orthogonal_output))

        indices_input, indices_output, values = flat_weights[index]

        blocks = (values.size + threads - 1) // threads
        _scatter[blocks, threads](  # type: ignore[index]
            indices_input,
            indices_output,
            values,
            values_input_flat,
            values_output_flat,
            base_input,
            num_input,
            strides_input[0],
            strides_input[1],
            base_output,
            num_output,
            strides_output[0],
            strides_output[1],
        )

    return result

from typing import Any, Sequence
import numpy as np
from numba import cuda
from regridding import _util
from regridding._cuda import on_device, rows
from ._shared import num_axis
from ._host import convolve_weights_host
from ._cuda import convolve_weights_cuda

__all__ = [
    "convolve_weights",
]


def convolve_weights(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    kernel: np.ndarray,
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Convolve the output of a set of weights with a kernel, such as a
    point-spread function.

    If the weights computed by :func:`regridding.weights` are the sparse
    matrix :math:`W`, mapping each input cell :math:`j` onto the output
    cells :math:`i`, this computes the sparse matrix :math:`P W`, where

    .. math::

        P_{i' i} = K_i(i' - i)

    spreads whatever lands in output cell :math:`i` over the cells
    :math:`i'` around it.  Applying the result with
    :func:`regridding.regrid_from_weights` is the same as resampling and
    then convolving with :math:`K`, but as a single set of weights it can
    be reused, and transposed by :func:`regridding.transpose_weights` or
    :func:`regridding.transpose_weights_conservative` to give the transpose
    of the whole operation.

    Parameters
    ----------
    weights
        Weights computed by :func:`regridding.weights`.
    kernel
        The kernel, :math:`K`.
        Its last ``len(axis_output)`` axes are the kernel itself, one for
        each of `axis_output`, in the same order.
        Along an axis of length :math:`n`, the element at index
        :math:`\lfloor n / 2 \rfloor` is the center, which is where
        :func:`scipy.ndimage.convolve` places it.

        Any axes before those are broadcast against the output grid,
        ``shape_output``, so the kernel may be different for each element of
        the orthogonal axes, such as for each wavelength, or it may vary
        across the output grid along the resampled axes.
        A kernel which varies across the output grid is indexed by the cell
        the light lands in *before* it is spread, :math:`i` above.

        The kernel must be dimensionless.
    axis_input
        The resampled axes of the input grid, as given to
        :func:`regridding.weights` and :func:`regridding.regrid_from_weights`.
        If :obj:`None`, all the axes of the input grid are resampled.
        This is only needed when the kernel varies along an orthogonal axis
        which the weights are broadcast along, since the shape of the input
        grid then has to be broadcast too.
    axis_output
        The resampled axes of the output grid, as given to
        :func:`regridding.weights` and :func:`regridding.regrid_from_weights`.
        If :obj:`None`, all the axes of the output grid are resampled.

    Returns
    -------
    The convolved weights, with the same shapes as `weights`, unless the
    kernel varies along an orthogonal axis the weights are broadcast along,
    in which case the weights and both shapes are broadcast along it.

    Raises
    ------
    ValueError
        If `kernel` has too few axes, if its leading axes cannot be broadcast
        to the output grid, or if it is not dimensionless; or if
        `axis_output` does not describe the grid the weights were built for,
        which shows as the weights having more orthogonal axes than it leaves
        or as an output index outside the grid it selects.

    Notes
    -----
    The kernel acts on the output cells, so it describes how the contents of
    a cell are redistributed among its neighbors, rather than how a point is
    blurred.  For a point-spread function :math:`h` sampled on cells of width
    :math:`\Delta`, that is :math:`h` convolved with a cell twice, once for
    the cell the light lands in and once for the cell it is collected in,
    and then sampled at the cell centers.

    Light which the kernel spreads beyond the edge of the output grid is
    lost, as it would be off the edge of a sensor, so the total of the
    result is reduced near the edges.

    Each input cell is convolved independently of every other, in a scratch
    box covering its footprint: the bounding box of the output cells it
    reaches, grown by the kernel.  Each of its weights is spread through the
    kernel into the box, and the box is then read out in order, so the cost
    is one multiply-add for each weight and element of the kernel, plus one
    visit to each cell of the box.  Weights ordered by input cell, as
    :func:`regridding.weights` returns them, come out ordered and with each
    ``(input, output)`` pair once.

    A kernel which varies along some of the resampled axes is stored with
    one row for each cell along those axes only, so a kernel which varies
    along one axis of a large grid costs no more than that axis.

    The result has more weights than `weights` does, by roughly the factor
    by which the kernel grows the footprint of an input cell: for a cell
    which lands on two output cells across, a kernel three cells across
    makes the weights about four times as long, and seven cells across about
    eighteen times.

    Weights built with ``device="cuda"`` are convolved on the device, and the
    result is left there.  Their empty slots, which carry an index of
    ``-1``, are dropped.

    See Also
    --------
    :func:`regridding.weights`
    :func:`regridding.regrid_from_weights`
    :func:`regridding.transpose_weights_conservative`

    Examples
    --------

    Rotate an array onto a new grid, and blur it with a Gaussian kernel in the
    same set of weights.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define the input grid
        x_input = np.linspace(-4, 4, num=17)
        y_input = np.linspace(-4, 4, num=17)
        x_input, y_input = np.meshgrid(x_input, y_input, indexing="ij")

        # Rotate the input grid
        angle = 0.3
        x_rotated = x_input * np.cos(angle) - y_input * np.sin(angle)
        y_rotated = x_input * np.sin(angle) + y_input * np.cos(angle)

        # Define a uniform output grid
        x_output = np.linspace(-6, 6, num=25)
        y_output = np.linspace(-6, 6, num=25)
        x_output, y_output = np.meshgrid(x_output, y_output, indexing="ij")

        # Define an array of values with two bright cells
        values_input = np.zeros((16, 16))
        values_input[4, 4] = 1
        values_input[10, 8] = 1

        # Save the weights which rotate the input array onto the output grid
        weights = regridding.weights(
            coordinates_input=(x_rotated, y_rotated),
            coordinates_output=(x_output, y_output),
            method="conservative",
        )

        # Define a Gaussian kernel, five cells across
        offset = np.arange(-2, 3)
        kernel = np.exp(-np.square(offset) / 2)
        kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]
        kernel = kernel / kernel.sum()

        # Blur the output of the weights with the kernel
        weights_blurred = regridding.convolve_weights(weights, kernel)

        # Apply both sets of weights
        values_rotated = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        values_blurred = regridding.regrid_from_weights(
            *weights_blurred,
            values_input=values_input,
        )

        # Plot the original, rotated, and blurred arrays
        fig, axs = plt.subplots(
            ncols=3,
            sharex=True,
            sharey=True,
            figsize=(9, 3.4),
            constrained_layout=True,
        )
        axs[0].pcolormesh(x_rotated, y_rotated, values_input);
        axs[0].set_title(f"original, total {values_input.sum():.2f}");
        axs[1].pcolormesh(x_output, y_output, values_rotated);
        axs[1].set_title(f"rotated, total {values_rotated.sum():.2f}");
        axs[2].pcolormesh(x_output, y_output, values_blurred);
        axs[2].set_title(f"rotated and blurred, total {values_blurred.sum():.2f}");
        for ax in axs:
            ax.set_aspect("equal");
    """

    weights_array, shape_input, shape_output = weights

    weights_array = np.asarray(weights_array)

    ndim = len(shape_output)
    ndim_input = len(shape_input)

    if axis_output is None:
        axis_output = tuple(range(ndim))
    axis_output = tuple(np.lib.array_utils.normalize_axis_tuple(axis_output, ndim))

    if axis_input is None:
        axis_input = tuple(range(ndim_input))
    axis_input = tuple(np.lib.array_utils.normalize_axis_tuple(axis_input, ndim_input))

    ndim_kernel = len(axis_output)

    if ndim_kernel > num_axis:  # pragma: nocover
        raise ValueError(
            f"a kernel can act along at most {num_axis} axes, got {ndim_kernel}"
        )

    # the weights carry one element for each position along the orthogonal
    # axes, so they cannot have more axes than `axis_output` leaves over
    axis_orthogonal = tuple(ax for ax in range(ndim) if ax not in axis_output)
    if weights_array.ndim > len(axis_orthogonal):
        raise ValueError(
            f"the weights have {weights_array.ndim} orthogonal axes, but "
            f"axis_output={axis_output} leaves only {len(axis_orthogonal)} of "
            f"the {ndim} axes of the output grid, {shape_output}"
        )

    try:
        kernel = _util._dimensionless(kernel)
    except ValueError as error:
        unit = getattr(kernel, "unit", None)
        if unit is None:
            raise
        raise ValueError(f"the kernel must be dimensionless, got {unit}") from error

    if kernel.ndim < ndim_kernel:
        raise ValueError(
            f"the kernel has {kernel.ndim} axes, fewer than the "
            f"{ndim_kernel} axes it acts along"
        )

    # the flat indices address the resampled axes in ascending order, so
    # the axes of the kernel itself are put in that order too
    order = np.argsort(axis_output)
    axis_output = tuple(axis_output[i] for i in order)
    kernel = np.moveaxis(
        kernel,
        source=tuple(range(kernel.ndim - ndim_kernel, kernel.ndim)),
        destination=tuple(kernel.ndim - ndim_kernel + i for i in np.argsort(order)),
    )

    shape_kernel = kernel.shape[kernel.ndim - ndim_kernel :]
    shape_grid = kernel.shape[: kernel.ndim - ndim_kernel]

    if len(shape_grid) > ndim:
        raise ValueError(
            f"the leading axes of the kernel, {shape_grid}, have more axes "
            f"than the output grid, {shape_output}"
        )
    shape_grid = (1,) * (ndim - len(shape_grid)) + shape_grid
    kernel = kernel.reshape(shape_grid + shape_kernel)

    shape_resampled = tuple(shape_output[ax] for ax in axis_output)
    for ax in axis_output:
        if shape_grid[ax] not in (1, shape_output[ax]):
            raise ValueError(
                f"the leading axes of the kernel, {shape_grid}, cannot be "
                f"broadcast to the output grid, {shape_output}"
            )

    # the output grid can be broadcast along the orthogonal axes, so those
    # are checked against the weights, which are not
    shape_orthogonal_kernel = tuple(shape_grid[ax] for ax in axis_orthogonal)
    try:
        np.broadcast_shapes(
            shape_orthogonal_kernel,
            weights_array.shape,
            tuple(shape_output[ax] for ax in axis_orthogonal),
        )
    except ValueError as error:
        raise ValueError(
            f"the leading axes of the kernel, {shape_grid}, cannot be "
            f"broadcast to the output grid, {shape_output}"
        ) from error

    # put the resampled axes of the kernel's grid next to the kernel itself
    # and flatten them, so that every element of the orthogonal axes has one
    # row for each cell of the grid the kernel varies over.  A kernel which
    # varies along only some of the resampled axes keeps only those, rather
    # than being copied out to one row for every output cell.
    shape_grid_resampled = tuple(shape_grid[ax] for ax in axis_output)
    kernel = np.transpose(
        kernel,
        axis_orthogonal + axis_output + tuple(range(ndim, ndim + ndim_kernel)),
    )
    kernel = kernel.reshape(
        shape_orthogonal_kernel
        + (int(np.prod(shape_grid_resampled)), int(np.prod(shape_kernel)))
    )

    # the orthogonal shape of the result is the weights' own, unless the
    # kernel varies along an axis the weights are broadcast along
    while len(shape_orthogonal_kernel) > weights_array.ndim:
        if shape_orthogonal_kernel[0] != 1:
            break
        shape_orthogonal_kernel = shape_orthogonal_kernel[1:]
        kernel = kernel[0]
    shape_orthogonal = np.broadcast_shapes(
        weights_array.shape,
        shape_orthogonal_kernel,
    )

    if shape_orthogonal != weights_array.shape:
        shape_input, shape_output = _broadcast_orthogonal(
            shape_orthogonal=shape_orthogonal,
            shape_input=shape_input,
            shape_output=shape_output,
            axis_input=axis_input,
            axis_output=axis_output,
        )

    weights_array = np.broadcast_to(weights_array, shape_orthogonal)

    size_resampled = int(np.prod(shape_resampled))

    # the offset of every element of the kernel from its center, along each
    # resampled axis
    offsets = np.stack(
        np.unravel_index(np.arange(int(np.prod(shape_kernel))), shape_kernel),
        axis=~0,
    )
    offsets = offsets - np.array(shape_kernel) // 2

    shape_grid_resampled = np.array(shape_grid_resampled, dtype=np.int64)
    shape_resampled = np.array(shape_resampled, dtype=np.int64)
    shape_kernel = np.array(shape_kernel, dtype=np.int64)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)

    device = on_device(weights_array)

    # set on the device if any weight's output cell lies outside the grid,
    # and read once all the elements are done
    outside = None

    # the kernels on the device, which `numba` ships no type for, and the
    # one each element reads, so that a kernel shared by every element is
    # sent once
    kernel_device: Any = None
    kernel_row: Any = None

    if device:
        shape_grid_resampled = cuda.to_device(shape_grid_resampled)
        shape_resampled = cuda.to_device(shape_resampled)
        shape_kernel = cuda.to_device(shape_kernel)
        offsets = cuda.to_device(offsets)
        outside = cuda.to_device(np.zeros(1, dtype=np.int64))
        kernel_device, kernel_row = rows(
            kernel,
            ndim=2,
            shape_orthogonal=shape_orthogonal,
        )
    else:
        kernel = np.broadcast_to(kernel, shape_orthogonal + kernel.shape[~1:])

    result = np.empty(shape_orthogonal, dtype=object)

    for index in np.ndindex(*shape_orthogonal):

        indices_input, indices_output, values = weights_array[index]

        if device:
            result[index] = convolve_weights_cuda(
                indices_input=indices_input,
                indices_output=indices_output,
                values=values,
                kernel=kernel_device[kernel_row[index]],
                shape_grid=shape_grid_resampled,
                shape_output=shape_resampled,
                shape_kernel=shape_kernel,
                offsets=offsets,
                size_output=size_resampled,
                outside=outside,
            )
            continue

        indices_input = np.asarray(indices_input)
        indices_output = np.asarray(indices_output)

        # an index outside the grid means `axis_output` does not describe the
        # grid the weights were built for
        if indices_output.size:
            if (
                indices_output.max() >= size_resampled
                or indices_output.min() < -size_resampled
            ):
                raise ValueError(_message_outside(axis_output, shape_output))

        unit_values = getattr(values, "unit", None)
        if unit_values is not None:
            values = getattr(values, "value")

        indices_input, indices_output, values = convolve_weights_host(
            indices_input=indices_input,
            indices_output=indices_output,
            values=np.asarray(values),
            kernel=kernel[index],
            shape_grid=shape_grid_resampled,
            shape_output=shape_resampled,
            shape_kernel=shape_kernel,
            offsets=offsets,
        )

        if unit_values is not None:
            values = values << unit_values

        result[index] = (indices_input, indices_output, values)

    if outside is not None and outside.copy_to_host()[0]:
        raise ValueError(_message_outside(axis_output, shape_output))

    return result, shape_input, shape_output


def _message_outside(
    axis_output: tuple[int, ...],
    shape_output: tuple[int, ...],
) -> str:
    """
    Explain an output index which lies outside the grid.

    Parameters
    ----------
    axis_output
        The resampled axes of the output grid, as given.
    shape_output
        The shape of the output grid.
    """
    return (
        f"the weights address output cells outside the grid selected by "
        f"axis_output={axis_output} from {shape_output}; it should be the "
        f"axis_output given to `regridding.weights()`"
    )


def _broadcast_orthogonal(
    shape_orthogonal: tuple[int, ...],
    shape_input: tuple[int, ...],
    shape_output: tuple[int, ...],
    axis_input: tuple[int, ...],
    axis_output: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Broadcast the orthogonal axes of the input and output grids against the
    orthogonal shape of a set of weights.

    The orthogonal axes are matched from the right, as
    :func:`regridding.regrid_from_weights` matches them.

    Parameters
    ----------
    shape_orthogonal
        The shape of the array of weights.
    shape_input
        The shape of the input grid.
    shape_output
        The shape of the output grid.
    axis_input
        The resampled axes of the input grid.
    axis_output
        The resampled axes of the output grid.

    Raises
    ------
    ValueError
        If either grid has fewer orthogonal axes than the weights, which
        is what happens when `axis_input` is left out for weights which
        have orthogonal axes.
    """

    result = []

    for shape, axis, name in (
        (shape_input, axis_input, "axis_input"),
        (shape_output, axis_output, "axis_output"),
    ):
        axis_orthogonal = [ax for ax in range(len(shape)) if ax not in axis]
        if len(axis_orthogonal) < len(shape_orthogonal):
            raise ValueError(
                f"the kernel adds orthogonal axes {shape_orthogonal} to the "
                f"weights, but {name}={axis} leaves the grid {shape} with "
                f"only {len(axis_orthogonal)} orthogonal axes to broadcast"
            )
        shape = list(shape)
        for ax, num in zip(reversed(axis_orthogonal), reversed(shape_orthogonal)):
            shape[ax] = np.broadcast_shapes((shape[ax],), (num,))[0]
        result.append(tuple(shape))

    shape_input, shape_output = result

    return shape_input, shape_output

from typing import Sequence, Literal
import numpy as np
from regridding import _util
from ._weights_multilinear import _weights_multilinear
from ._weights_conservative import _weights_conservative
from ._weights_arrays import _weights_to_arrays

__all__ = [
    "weights",
]


def _weights_astype(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    dtype_indices: None | np.typing.DTypeLike,
    dtype_values: None | np.typing.DTypeLike,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """
    Store each element's flat arrays as narrower types.

    This is the last step of a build, so the geometry and any merging of
    repeated pairs have already been done in double precision and only the
    stored result is narrowed.

    Parameters
    ----------
    weights
        Array of per-element flat weight arrays.
    dtype_indices
        The type to store the indices as, or :obj:`None` to leave them.
    dtype_values
        The type to store the weights as, or :obj:`None` to leave them.

    Raises
    ------
    ValueError
        If an index does not fit in `dtype_indices`.  Letting it wrap around
        would address the wrong cell instead of failing.

    Notes
    -----
    The indices are only scanned when the number of cells does not already
    fit in `dtype_indices`, since the grids bound them.  Scanning is more
    expensive than the conversion itself: the arrays of indices are usually
    several times larger than the grids they address.
    """

    array, shape_input, shape_output = weights

    info = None if dtype_indices is None else np.iinfo(dtype_indices)

    # The indices address the flattened grids, so their magnitude cannot
    # exceed the number of cells, which is known without looking at them.
    # When that bound already fits there is nothing to check, which saves
    # scanning arrays that are usually far larger than the grids are.
    if info is not None:
        bound = max(
            int(np.prod(shape_input)),
            int(np.prod(shape_output)),
        )
        if bound <= info.max and -bound >= info.min:
            info = None

    flat = array.reshape(-1)

    for k in range(flat.size):

        indices_input, indices_output, values = flat[k]

        if info is not None:
            for indices, name in (
                (indices_input, "input"),
                (indices_output, "output"),
            ):
                if not indices.size:  # pragma: nocover
                    continue
                index_min = indices.min()
                index_max = indices.max()
                if index_min < info.min or index_max > info.max:
                    raise ValueError(
                        f"the {name} indices span [{index_min}, {index_max}], "
                        f"which does not fit in {np.dtype(dtype_indices)} "
                        f"([{info.min}, {info.max}])"
                    )

        if dtype_indices is not None:
            indices_input = indices_input.astype(dtype_indices)
            indices_output = indices_output.astype(dtype_indices)

        if dtype_values is not None:
            values = values.astype(dtype_values)

        flat[k] = (indices_input, indices_output, values)

    return array, shape_input, shape_output


def weights(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    method: Literal["multilinear", "conservative"] = "multilinear",
    bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    perturb: None | bool = None,
    seed: "None | int | np.random.Generator" = _util._seed_default,
    coalesce: bool = True,
    dtype_indices: None | np.typing.DTypeLike = None,
    dtype_values: None | np.typing.DTypeLike = None,
    device: None | str = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """
    Save the results of a regridding operation as a sequence of weights,
    which can be used in subsequent regridding operations on the same grid.

    The results of this function are designed to be used by
    :func:`regridding.regrid_from_weights`

    This function returns a tuple containing an array of weights,
    the shape of the input coordinates, and the shape of the output
    coordinates.  Each element of the weights array is a tuple of three flat
    arrays, ``(indices_input, indices_output, values)``, describing the
    sparse mapping for one orthogonal element; this form pickles and
    memory-maps cleanly.

    Parameters
    ----------
    coordinates_input
        Coordinates of the input grid.
    coordinates_output
        Coordinates of the output grid.
        Should have the same number of coordinates as the input grid.
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
        Weights applied to the values of the input grid before resampling.
    method
        The type of regridding to use.
    bounds
        How to treat output points that fall outside the input grid.
        Only applies when `method` is ``multilinear``; the ``conservative``
        method assigns no weight to regions outside the input grid.
        ``extrapolate`` (the default) extrapolates linearly from the nearest
        cell of the input grid, ``nan`` sets those points to
        :obj:`numpy.nan`, and ``raise`` raises a :class:`ValueError`.
    perturb
        Whether to perturb `coordinates_output` by a small value to avoid degenerate
        grids. This is helpful for some methods, like ``conservative``, which
        sometimes cannot handle degenerate grids.
        If :obj:`None` (the default), no perturbation is applied unless `method`
        is ``conservative`` and the dimensions of the grid are 2D or higher.
        If :obj:`True`, each point is perturbed using a normal distribution
        with standard deviation equal to ``1e-9`` of the grid width.
    seed
        The seed used by the pseudo-random number generator which perturbs
        `coordinates_output`.
        May be an integer or an instance of :class:`numpy.random.Generator`.
        The default is a fixed integer, so that repeated calls using the same
        grids return identical results.
        If :obj:`None`, the generator is seeded from fresh entropy,
        and each call draws an independent perturbation.
    coalesce
        Whether to merge repeated ``(input, output)`` pairs by summing their
        weights before returning.

        The conservative builders can emit several fragments for the same
        pair, and merging them shrinks the result by the mean multiplicity,
        which makes every subsequent
        :func:`regridding.regrid_from_weights` cheaper.  The merge itself
        costs a sort, so it pays for itself only if the weights are applied
        more than once or twice.

        Setting this to :obj:`False` returns the fragments as they were
        built.  The result is equivalent: applying the weights sums
        duplicates during the scatter-add either way.  This is the better
        choice when the grid changes on every call, so each set of weights
        is applied once and there is nothing to amortize the sort against.
    dtype_indices
        The type to store the indices as.
        If :obj:`None` (the default), they are left as :class:`numpy.int64`.

        The indices address flattened grids, so :class:`numpy.int32` is
        enough for any grid with fewer than about two billion cells and
        halves what they cost to store.  A range which does not fit raises
        rather than wrapping around.
    dtype_values
        The type to store the weights as.
        If :obj:`None` (the default), they are left as :class:`numpy.float64`.

        The weights are computed and, if `coalesce` is set, summed in double
        precision regardless; only the stored result is narrowed.  Storing
        them as :class:`numpy.float32` halves their size and costs about
        5e-8 in the total weight of each input cell, since
        :func:`regridding.regrid_from_weights` accumulates into a double
        precision array.
    device
        The device to build the weights on.
        If :obj:`None` (the default), they are built on the host as
        :class:`numpy.ndarray`.

        Passing ``"cuda"`` builds them with a CUDA kernel and leaves them
        in device memory, as
        :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`.

        :func:`regridding.regrid_from_weights` applies them where they
        are and leaves its result on the device as well, so a scene which
        is already there is never brought back::

            weights = regridding.weights(..., coalesce=False, device="cuda")
            image = regridding.regrid_from_weights(*weights, values_input=scene)

        The result exposes ``__cuda_array_interface__``, so
        :func:`torch.as_tensor` wraps it without copying.

        This needs the output grid to be a uniform, axis-aligned lattice,
        since only the clipping kernel is ported, and it needs `coalesce`
        to be :obj:`False`, since merging repeated pairs is not.

        Slots which received no overlap carry an index of ``-1``, and the
        weights beside them are left uninitialized.
        :func:`regridding.regrid_from_weights` skips them; anything else
        reading the weights directly has to drop them.

        `dtype_values` selects the precision the kernel clips in, rather
        than being applied to the result afterwards.  `dtype_indices` is
        not supported, since the kernel addresses its slots with
        :class:`numpy.int64`.

    See Also
    --------
    :func:`regridding.regrid`
    :func:`regridding.regrid_from_weights`

    Examples
    --------

    Regrid two arrays of values defined on the same grid using saved weights.

    .. jupyter-execute::

        import numpy as np
        import scipy.signal
        import matplotlib.pyplot as plt
        import regridding

        # Define input grid
        x_input = np.linspace(-4, 4, num=101)
        y_input = np.linspace(-4, 4, num=101)
        x_input, y_input = np.meshgrid(x_input, y_input, indexing="ij")

        # Define rotated output grid
        angle = 0.2
        x_output = x_input * np.cos(angle) - y_input * np.sin(angle)
        y_output = x_input * np.sin(angle) + y_input * np.cos(angle)

        # Define two arrays of values defined on the same grid
        values_input_1 = np.cos(np.square(x_input)) * np.cos(np.square(y_input))
        values_input_2 = np.sin(np.square(x_input) + np.square(y_input))

        # Convolve with a 2x2 uniform kernel to simulate values defined on cell centers
        values_input_1 = scipy.signal.convolve(values_input_1, np.ones((2, 2)), mode="valid")
        values_input_2 = scipy.signal.convolve(values_input_2, np.ones((2, 2)), mode="valid")

        # Save regridding weights relating the input and output grids
        weights = regridding.weights(
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
            method="conservative",
        )

        # Regrid the first array of values using the saved weights
        values_output_1 = regridding.regrid_from_weights(
            *weights,
            values_input=values_input_1,
        )

        # Regrid the second array of values using the saved weights
        values_output_2 = regridding.regrid_from_weights(
            *weights,
            values_input=values_input_2,
        )

        # Plot the original and regridded arrays of values
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs[0, 0].pcolormesh(x_input, y_input, values_input_1);
        axs[0, 0].set_title(r"values_input_1");
        axs[0, 1].pcolormesh(x_input, y_input, values_input_2);
        axs[0, 1].set_title(r"values_input_2");
        axs[1, 0].pcolormesh(x_output, y_output, values_output_1);
        axs[1, 0].set_title(r"values_output_1");
        axs[1, 1].pcolormesh(x_output, y_output, values_output_2);
        axs[1, 1].set_title(r"values_output_2");
    """
    # the numba builders cannot ingest united quantities, but the flat-array
    # form can carry them: strip the unit here and reattach it to the values
    unit_weights = getattr(weights_input, "unit", None)
    if unit_weights is not None:
        weights_input = getattr(weights_input, "value")

    if device is not None:
        if method != "conservative":
            raise ValueError(f"{device=} is only supported by the conservative method")
        if coalesce:
            raise ValueError(
                f"{device=} needs `coalesce=False`; merging repeated pairs is "
                f"not ported to the device, and weights which are applied once "
                f"have nothing to amortize the merge against"
            )
        if dtype_indices is not None:
            raise ValueError(
                f"{dtype_indices=} is not supported with {device=}; the device "
                f"kernel addresses its slots with `numpy.int64`"
            )

    if method == "multilinear":
        result = _weights_multilinear(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=axis_input,
            axis_output=axis_output,
            weights_input=weights_input,
            bounds=bounds,
            perturb=perturb,
            seed=seed,
        )
    elif method == "conservative":
        result = _weights_conservative(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=axis_input,
            axis_output=axis_output,
            weights_input=weights_input,
            perturb=perturb,
            seed=seed,
            device=device,
            dtype=dtype_values,
        )
    else:
        raise ValueError(f"unrecognized method '{method}'")

    if coalesce:
        result = _weights_to_arrays(result)

    # a device result is built in `dtype_values` to begin with, and cannot be
    # narrowed afterwards in any case, since it does not live on the host
    if device is None and (dtype_indices is not None or dtype_values is not None):
        result = _weights_astype(
            weights=result,
            dtype_indices=dtype_indices,
            dtype_values=dtype_values,
        )

    if unit_weights is not None:
        array, shape_input, shape_output = result
        flat = array.reshape(-1)
        for k in range(flat.size):
            indices_input, indices_output, values = flat[k]
            flat[k] = (indices_input, indices_output, values << unit_weights)
        result = array, shape_input, shape_output

    return result

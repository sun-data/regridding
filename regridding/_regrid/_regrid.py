from typing import Sequence, Literal
import numpy as np
import regridding
from regridding import _util
from . import regrid_from_weights

__all__ = [
    "regrid",
]


def regrid(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray = None,
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    method: Literal["multilinear", "conservative"] = "multilinear",
    bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    perturb: None | bool = None,
    seed: "None | int | np.random.Generator" = _util._seed_default,
    coalesce: bool = False,
) -> np.ndarray:
    """
    Regrid an array of values defined on a logically-rectangular curvilinear
    grid onto a new logically-rectangular curvilinear grid.

    Parameters
    ----------
    coordinates_input
        Coordinates of the input grid.
    coordinates_output
        Coordinates of the output grid.
        Should have the same number of coordinates as the input grid.
    values_input
        Input array of values to be resampled.
    values_output
        Optional array in which to place the output.
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
    method
        The type of regridding to use.
        The ``multilinear`` method interprets `coordinates_input` as the
        points where `values_input` is sampled.
        The ``conservative`` method interprets `coordinates_input` as the edges
        of the cells containing `values_input`, so `values_input` has one fewer
        element along each resampled axis, and the sum of the result matches the
        sum of `values_input`.
        The ``conservative`` method uses the algorithm described in
        :footcite:t:`Ramshaw1985`.
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
        Whether to merge repeated ``(input, output)`` pairs before applying
        the weights.
        See :func:`regridding.weights`.

        This defaults to :obj:`False`, unlike :func:`regridding.weights`,
        because this function applies the weights exactly once.  Merging
        costs a sort and only earns it back over repeated applications, so
        skipping it is about twice as fast here even though it leaves more
        pairs for the scatter-add to sum.
        Use :func:`regridding.weights` with ``coalesce=True`` if the same
        grids will be regridded more than once.

    See Also
    --------
    :func:`regridding.weights`
    :func:`regridding.regrid_from_weights`

    Examples
    --------

    Resample a 1D array onto a coarser grid without changing its total.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define the edges of the input and output grids
        x_input = np.linspace(-1, 1, num=31)
        x_output = np.linspace(-1, 1, num=11)

        # Define an array of values for each cell of the input grid
        x_center = (x_input[1:] + x_input[:~0]) / 2
        values_input = np.exp(-np.square(x_center / 0.5))

        # Resample the values onto the output grid
        values_output = regridding.regrid(
            coordinates_input=(x_input,),
            coordinates_output=(x_output,),
            values_input=values_input,
            method="conservative",
        )

        # The sum of the array is unchanged by the resampling
        print(values_input.sum(), values_output.sum())

        # Plot the result
        fig, ax = plt.subplots(constrained_layout=True);
        ax.stairs(values_input, x_input, label="input");
        ax.stairs(values_output, x_output, label="output");
        ax.legend();

    References
    ----------
    .. footbibliography::
    """
    weights, shape_input, shape_output = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
        bounds=bounds,
        perturb=perturb,
        seed=seed,
        coalesce=coalesce,
    )
    result = regrid_from_weights(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    return result

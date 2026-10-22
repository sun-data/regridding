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
        For the ``conservative`` method, the number of axes should be equal
        to the number of coordinates in the output grid.
        The ``multilinear`` method interpolates at points rather than
        resampling cells, so the points may be arranged along any number of
        axes, or none at all for a single point.
    method
        The type of regridding to use.
        The ``multilinear`` method interprets `coordinates_input` as the
        points where `values_input` is sampled, and interpolates between the
        corners of the cell which contains each output point, on a grid of
        any number of dimensions, rectilinear or curvilinear.
        The ``conservative`` method interprets `coordinates_input` as the edges
        of the cells containing `values_input`, so `values_input` has one fewer
        element along each resampled axis, and the sum of the result matches the
        sum of `values_input`.
        The ``conservative`` method sweeps the grid lines of both grids
        following :footcite:t:`Ramshaw1985`, except where the output grid is
        a uniform, axis-aligned lattice, when it clips each input cell
        against the output cells it reaches instead, after
        :footcite:t:`Sutherland1974`.
    bounds
        How to treat output points that fall outside the input grid.
        Only applies when `method` is ``multilinear``; the ``conservative``
        method assigns no weight to regions outside the input grid.
        ``extrapolate`` (the default) extrapolates linearly from the nearest
        cell of the input grid, ``nan`` sets those points to
        :obj:`numpy.nan`, and ``raise`` raises a :class:`ValueError`.
        On a grid which is not rectilinear, the nearest cell is the one along
        the edge of the grid whose continuation reaches the point, and a
        point which no continuation reaches, as can happen beyond a strongly
        curved edge, is set to :obj:`numpy.nan`.
        A point with a coordinate which is not finite is set to
        :obj:`numpy.nan`, or raises, whichever `bounds` is.
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

    Interpolate an array sampled on a distorted 2D grid at scattered points.

    .. jupyter-execute::

        # Define the vertices of the input grid
        u, v = np.meshgrid(
            np.linspace(-1, 1, num=21),
            np.linspace(-1, 1, num=21),
            indexing="ij",
        )
        x_input = u + 0.2 * v * v
        y_input = v - 0.2 * u * u

        # Sample a function on the vertices
        values_input = np.cos(3 * x_input) * np.sin(2 * y_input)

        # Define a set of random points, some of them outside the grid
        rng = np.random.default_rng(0)
        x_output = rng.uniform(-1.3, 1.3, size=500)
        y_output = rng.uniform(-1.3, 1.3, size=500)

        # Interpolate the function at the points
        values_output = regridding.regrid(
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
            values_input=values_input,
            method="multilinear",
            bounds="nan",
        )

        # Plot the input grid and the interpolated points
        fig, ax = plt.subplots(constrained_layout=True);
        ax.pcolormesh(x_input, y_input, values_input, shading="gouraud", vmin=-1, vmax=1);
        ax.scatter(x_output, y_output, c=values_output, vmin=-1, vmax=1, edgecolors="black");
        ax.set_aspect("equal");

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

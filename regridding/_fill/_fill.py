from typing import Sequence, Literal
import numpy as np
from ._gauss_sidel import fill_gauss_sidel

__all__ = [
    "fill",
]


def fill(
    a: np.ndarray,
    where: None | np.ndarray = None,
    axis: None | int | Sequence[int] = None,
    method: Literal["gauss_sidel"] = "gauss_sidel",
    **kwargs,
) -> np.ndarray:
    """
    Fill an array with missing values by interpolating from the valid points.

    Parameters
    ----------
    a
        The array with missing values to be filled
    where
        Boolean array of missing values.
        If :obj:`None` (the default), all NaN values will be filled.
    axis
        The axes to use for interpolation.
        If :obj:`None` (the default), interpolate along all the axes of `a`.
    method
        The interpolation method to use.
        The only option is "gauss_sidel", which uses the Gauss-Sidel relaxation
        technique to interpolate the valid data points.
    kwargs
        Additional method-specific keyword arguments.
        For the Gauss-Sidel method, the valid keyword arguments are:
        - ``num_iterations=100``, the number of red-black Gauss-Sidel iterations to perform.

    Examples
    --------

    Set random elements of an array to NaN, and then fill in the missing elements
    using the Gauss-Sidel relaxation method.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define the independent variables
        x = 3 * np.pi * np.linspace(-1, 1, num=51)
        y = 3 * np.pi * np.linspace(-1, 1, num=51)
        x, y = np.meshgrid(x, y, indexing="ij")

        # Define the array to remove elements from
        a = np.cos(x) * np.cos(y)

        # Define the elements of the array to remove
        where = np.random.uniform(0, 1, size=a.shape) > 0.9

        # Set random elements of the array to NaN
        a_missing = a.copy()
        a_missing[where] = np.nan

        # Fill the missing elements using Gauss-Sidel relaxation
        b = regridding.fill(a_missing, method="gauss_sidel", num_iterations=11)

        # Plot the results
        fig, axs = plt.subplots(
            ncols=3,
            figsize=(6, 3),
            sharey=True,
            constrained_layout=True,
        )
        kwargs_imshow = dict(
            vmin=a.min(),
            vmax=a.max(),
        )
        axs[0].imshow(a_missing, **kwargs_imshow);
        axs[1].imshow(b, **kwargs_imshow);
        axs[2].imshow(a - b, **kwargs_imshow);
        axs[0].set_title("original array");
        axs[1].set_title("filled array");
    """

    if where is None:
        where = np.isnan(a)

    if method == "gauss_sidel":
        return fill_gauss_sidel(
            a=a,
            where=where,
            axis=axis,
            **kwargs,
        )
    else:
        raise ValueError("Unrecognized method '{method}'")

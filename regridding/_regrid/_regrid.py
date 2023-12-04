from typing import Sequence, Literal
import numpy as np
import regridding
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
        The ``conservative`` method uses the algorithm described in
        :footcite:t:`Ramshaw1985`

    See Also
    --------
    :func:`regridding.regrid`
    :func:`regridding.regrid_from_weights`

    Examples
    --------

    Regrid a 1D array using multilinear interpolation.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define the input grid
        x_input = np.linspace(-1, 1, num=11)

        # Define the input array
        values_input = np.square(x_input)

        # Define the output grid
        x_output = np.linspace(-1, 1, num=51)

        # Regrid the input array onto the output grid
        values_output = regridding.regrid(
            coordinates_input=(x_input,),
            coordinates_output=(x_output,),
            values_input=values_input,
            method="multilinear",
        )

        # Plot the results
        plt.figure(figsize=(6, 3));
        plt.scatter(x_input, values_input, s=100, label="input", zorder=1);
        plt.scatter(x_output, values_output, label="interpolated", zorder=0);
        plt.legend();

    |

    Regrid a 2D array using conservative resampling.

    .. jupyter-execute::

        # Define the number of edges in the input grid
        num_x = 66
        num_y = 66

        # Define a dummy linear grid
        x = np.linspace(-5, 5, num=num_x)
        y = np.linspace(-5, 5, num=num_y)
        x, y = np.meshgrid(x, y, indexing="ij")

        # Define the curvilinear input grid using the dummy grid
        angle = 0.4
        x_input = x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x
        y_input = x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y

        # Define the test pattern
        pitch = 16
        a_input = 0 * x[:~0,:~0]
        a_input[::pitch, :] = 1
        a_input[:, ::pitch] = 1
        a_input[pitch//2::pitch, pitch//2::pitch] = 1

        # Define a rectilinear output grid using the limits of the input grid
        x_output = np.linspace(x_input.min(), x_input.max(), num_x // 2)
        y_output = np.linspace(y_input.min(), y_input.max(), num_y // 2)
        x_output, y_output = np.meshgrid(x_output, y_output, indexing="ij")

        # Regrid the test pattern onto the new grid
        a_output = regridding.regrid(
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
            values_input=a_input,
            method="conservative",
        )

        fig, axs = plt.subplots(
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(8, 4),
            constrained_layout=True,
        );
        axs[0].pcolormesh(x_input, y_input, a_input);
        axs[0].set_title("input array");
        axs[1].pcolormesh(x_output, y_output, a_output);
        axs[1].set_title("regridded array");

    |

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

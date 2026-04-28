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
    :func:`regridding.weights`
    :func:`regridding.regrid_from_weights`

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

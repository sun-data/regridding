from typing import Literal
import numpy as np
from regridding import _util
from ._find_indices_brute import _find_indices_brute
from ._find_indices_searchsorted import _find_indices_searchsorted

__all__ = [
    "find_indices",
]


def find_indices(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | tuple[int, ...] = None,
    axis_output: None | int | tuple[int, ...] = None,
    fill_value: None | int = None,
    method: Literal["brute", "searchsorted"] = "brute",
) -> tuple[np.ndarray, ...]:
    """
    Find the index of the input cell which contains the output vertex.

    The index is returned as one array of indices per resampled dimension,
    each with the same shape as the output grid.
    Output vertices which fall outside of the input grid are assigned
    `fill_value`.

    Parameters
    ----------
    coordinates_input
        Coordinates of the input grid.
    coordinates_output
        Coordinates of the output grid.
        Should have the same number of coordinates as the input grid.
    axis_input
        Logical axes of the input grid to search.
        If :obj:`None`, search all the axes of the input grid.
    axis_output
        Logical axes of the output grid corresponding to the searched axes
        of the input grid.
        If :obj:`None`, all the axes of the output grid correspond to searched
        axes in the input grid.
    fill_value
        Numeric value to use for the indices of output vertices which are
        outside the input grid.
        If :obj:`None` (the default), the largest representable integer is
        used.
    method
        Flag to select which search algorithm to use.
        The ``brute`` method checks every cell of the input grid, and works for
        curvilinear grids.
        The ``searchsorted`` method uses a binary search, and is much faster,
        but requires a rectilinear input grid.

    See Also
    --------
    :func:`regridding.regrid`

    Examples
    --------

    Find the cell of a 1D grid containing each point of a second grid.

    .. jupyter-execute::

        import numpy as np
        import regridding

        x_input = np.linspace(-1, 1, num=5)
        x_output = np.array([-0.9, -0.1, 0.6])

        regridding.find_indices(
            coordinates_input=(x_input,),
            coordinates_output=(x_output,),
            method="searchsorted",
        )
    """

    (
        coordinates_input,
        coordinates_output,
        axis_input,
        axis_output,
        shape_input,
        shape_output,
        shape_orthogonal,
    ) = _util._normalize_input_output_coordinates(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    if fill_value is None:
        fill_value = np.iinfo(int).max

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    coordinates_input = tuple(
        np.moveaxis(v, axis_input, axis_input_numba).reshape(-1, *shape_input_numba)
        for v in coordinates_input
    )
    coordinates_output = tuple(
        np.moveaxis(v, axis_output, axis_output_numba).reshape(-1, *shape_output_numba)
        for v in coordinates_output
    )

    if method == "brute":
        indices_output = _find_indices_brute(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            fill_value=fill_value,
        )
    elif method == "searchsorted":
        indices_output = _find_indices_searchsorted(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            fill_value=fill_value,
        )
    else:
        raise ValueError(f"method `{method}` not recognized.")

    indices_output = tuple(
        np.moveaxis(
            a=i.reshape(*shape_orthogonal, *shape_output_numba),
            source=axis_output_numba,
            destination=axis_output,
        )
        for i in indices_output
    )

    return indices_output

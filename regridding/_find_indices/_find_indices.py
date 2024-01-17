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

    Parameters
    ----------
    coordinates_input
        the source grid
    coordinates_output
        the destination grid
    axis_input
        the axes in the source grid to search
    axis_output
        the axes in the destination grid corresponding to the source grid
    fill_value
        numeric value to use for invalid indices
    method
        flag to select which search algorithm to use
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

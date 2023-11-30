from typing import Literal
import numpy as np
from regridding import _util
from ._find_indices_brute import _find_indices_brute
from ._find_indices_searchsorted import _find_indices_searchsorted

__all__ = [
    "find_indices",
]


def find_indices(
    vertices_input: tuple[np.ndarray, ...],
    vertices_output: tuple[np.ndarray, ...],
    axis_input: None | int | tuple[int, ...] = None,
    axis_output: None | int | tuple[int, ...] = None,
    fill_value: None | int = None,
    method: Literal["brute"] | Literal["searchsorted"] = "brute",
):
    """
    Find the index of the input cell which contains the output vertex.

    Parameters
    ----------
    vertices_input
        the source grid
    vertices_output
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
        vertices_input,
        vertices_output,
        axis_input,
        axis_output,
        shape_input,
        shape_output,
        shape_orthogonal,
    ) = _util._normalize_input_output_vertices(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    if fill_value is None:
        fill_value = np.iinfo(int).max

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    vertices_input = tuple(
        np.moveaxis(v, axis_input, axis_input_numba).reshape(-1, *shape_input_numba)
        for v in vertices_input
    )
    vertices_output = tuple(
        np.moveaxis(v, axis_output, axis_output_numba).reshape(-1, *shape_output_numba)
        for v in vertices_output
    )

    if method == "brute":
        indices_output = _find_indices_brute(
            vertices_input=vertices_input,
            vertices_output=vertices_output,
            fill_value=fill_value,
        )
    elif method == "searchsorted":
        indices_output = _find_indices_searchsorted(
            vertices_input=vertices_input,
            vertices_output=vertices_output,
            fill_value=fill_value,
        )
    else:
        raise ValueError(f"method `{method}` not recognized.")

    indices_output = tuple(
        np.moveaxis(
            a=i.reshape(*shape_orthogonal, *shape_output),
            source=axis_output_numba,
            destination=axis_input,
        )
        for i in indices_output
    )

    return indices_output

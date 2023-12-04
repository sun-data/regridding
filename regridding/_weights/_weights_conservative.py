from typing import Sequence
import numpy as np
import numba
from regridding import _util
from regridding._conservative_ramshaw import _conservative_ramshaw


def _weights_conservative(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
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

    shape_values_input = list(shape_input)
    for ax in axis_input:
        shape_values_input[ax] -= 1
    shape_values_input = tuple(shape_values_input)

    shape_values_output = list(shape_output)
    for ax in axis_output:
        shape_values_output[ax] -= 1
    shape_values_output = tuple(shape_values_output)

    weights = np.empty(shape_orthogonal, dtype=numba.typed.List)

    for index in np.ndindex(*shape_orthogonal):
        index_vertices_input = list(index)
        for ax in axis_input:
            index_vertices_input.insert(ax, slice(None))
        index_vertices_input = tuple(index_vertices_input)

        index_vertices_output = list(index)
        for ax in axis_output:
            index_vertices_output.insert(ax, slice(None))
        index_vertices_output = tuple(index_vertices_output)

        if len(axis_input) == 1:
            raise NotImplementedError("1D regridding not supported")

        elif len(axis_input) == 2:
            coordinates_input_x, coordinates_input_y = coordinates_input
            coordinates_output_x, coordinates_output_y = coordinates_output

            weights[index] = _conservative_ramshaw(
                grid_input=(
                    coordinates_input_x[index_vertices_input],
                    coordinates_input_y[index_vertices_input],
                ),
                grid_output=(
                    coordinates_output_x[index_vertices_output],
                    coordinates_output_y[index_vertices_output],
                ),
            )

        else:
            raise NotImplementedError(
                "Regridding operations greater than 2D are not supported"
            )

    return weights, shape_values_input, shape_values_output

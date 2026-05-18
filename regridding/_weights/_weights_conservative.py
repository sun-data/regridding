from typing import Sequence
import multiprocessing
import concurrent.futures
import numpy as np
import numba
from regridding import _util
from ._weights_conservative_1d import weights_conservative_1d
from ._weights_conservative_2d import weights_conservative_2d


def _weights_conservative(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    perturb: None | bool = True,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:

    if perturb is None:
        if len(coordinates_input) > 1:
            perturb = True
        else:
            perturb = False

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
        perturb=perturb,
    )

    shape_values_input = list(shape_input)
    for ax in axis_input:
        shape_values_input[ax] -= 1
    shape_values_input = tuple(shape_values_input)

    shape_values_output = list(shape_output)
    for ax in axis_output:
        shape_values_output[ax] -= 1
    shape_values_output = tuple(shape_values_output)

    if weights_input is not None:
        weights_input = np.broadcast_to(weights_input, shape_values_input)

    weights = np.empty(shape_orthogonal, dtype=numba.typed.List)

    if len(axis_input) == 1:

        threads = 5 * multiprocessing.cpu_count()

        with concurrent.futures.ThreadPoolExecutor(threads) as executor:

            (x_input,) = coordinates_input
            (x_output,) = coordinates_output

            x_input = np.moveaxis(x_input, axis_input, ~0)
            x_output = np.moveaxis(x_output, axis_output, ~0)

            x_input = x_input.reshape(-1, x_input.shape[~0])
            x_output = x_output.reshape(-1, x_output.shape[~0])

            if weights_input is not None:
                weights_input = np.moveaxis(weights_input, axis_input, ~0)
                weights_input = weights_input.reshape(-1, weights_input.shape[~0])

            weights = weights.reshape(-1)

            step = np.ceil(x_input.shape[0] / threads).astype(int)

            futures = []

            for t in range(threads):

                index_start = t * step
                index_stop = (t + 1) * step

                future = executor.submit(
                    weights_conservative_1d,
                    x_input=x_input,
                    x_output=x_output,
                    weights_input=weights_input,
                    weights_output=weights,
                    index_start=index_start,
                    index_stop=index_stop,
                )

                futures.append(future)

                if index_stop >= x_output.shape[0]:
                    break

            concurrent.futures.wait(futures)

        weights = weights.reshape(shape_orthogonal)

    else:

        for index in np.ndindex(*shape_orthogonal):
            index_vertices_input = list(reversed(index))

            for ax in axis_input:
                index_vertices_input.insert(~ax, slice(None))
            index_vertices_input = tuple(reversed(index_vertices_input))

            index_vertices_output = list(reversed(index))
            for ax in axis_output:
                index_vertices_output.insert(~ax, slice(None))
            index_vertices_output = tuple(reversed(index_vertices_output))

            if len(axis_input) == 2:
                coordinates_input_x, coordinates_input_y = coordinates_input
                coordinates_output_x, coordinates_output_y = coordinates_output
                if weights_input is not None:
                    weights_input_index = weights_input[index]
                else:
                    weights_input_index = None
                weights[index] = weights_conservative_2d(
                    grid_input=(
                        coordinates_input_x[index_vertices_input],
                        coordinates_input_y[index_vertices_input],
                    ),
                    grid_output=(
                        coordinates_output_x[index_vertices_output],
                        coordinates_output_y[index_vertices_output],
                    ),
                    weights_input=weights_input_index,
                )

            else:  # pragma: nocover
                raise NotImplementedError(
                    "Regridding operations greater than 2D are not supported"
                )

    return weights, shape_values_input, shape_values_output

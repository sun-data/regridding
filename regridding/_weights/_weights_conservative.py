from typing import Sequence
import itertools
import multiprocessing
import concurrent.futures
import numpy as np
from numba.typed.typedlist import List as TypedList
from regridding import _util
from ._weights_conservative_1d import weights_conservative_1d
from ._weights_conservative_2d import (
    weights_conservative_2d,
    weights_conservative_2d_clipping,
    grid_is_uniform_rectilinear,
)


def _clipping_applicable(
    coordinates_output: tuple[np.ndarray, ...],
    axis_output: tuple[int, ...],
    shape_orthogonal: tuple[int, ...],
) -> bool:
    """
    Whether every output grid is a uniform, axis-aligned lattice, and so may
    be resampled with the clipping kernel instead of the sweep.

    Every orthogonal element is checked, so that a call either uses the
    clipping kernel throughout or the sweep throughout and the result does
    not depend on which element is inspected.

    Parameters
    ----------
    coordinates_output
        The broadcast vertices of the new grid.
    axis_output
        The axes of the new grid being resampled.
    shape_orthogonal
        The shape of the axes which are not resampled.
    """

    if len(axis_output) != 2:
        return False
    if len(coordinates_output) != 2:  # pragma: nocover
        return False

    x, y = coordinates_output

    # The grids are usually broadcast across the orthogonal axes, so every
    # element reads the same memory and one check answers for all of them.
    # A zero stride guarantees that; anything else falls back to checking
    # each element, which is correct but proportional to their number.
    axis_orthogonal = tuple(a for a in range(x.ndim) if a not in axis_output)
    shared = all(x.strides[a] == 0 and y.strides[a] == 0 for a in axis_orthogonal)

    indices = np.ndindex(*shape_orthogonal)
    if shared:
        indices = itertools.islice(indices, 1)

    for index in indices:
        index_vertices = list(reversed(index))
        for ax in axis_output:
            index_vertices.insert(~ax, slice(None))
        index_vertices = tuple(reversed(index_vertices))
        if not grid_is_uniform_rectilinear((x[index_vertices], y[index_vertices])):
            return False

    return True


def _weights_conservative(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    perturb: None | bool = True,
    seed: "None | int | np.random.Generator" = _util._seed_default,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:

    if perturb is None:
        perturb = False
        if not isinstance(coordinates_input, np.ndarray):
            if len(coordinates_input) > 1:
                perturb = True

    # The clipping kernel is used when every output grid is a uniform,
    # axis-aligned lattice.  It needs that lattice exactly, and it resolves
    # degeneracies on its own, so the perturbation is both harmful and
    # unnecessary there: decide before perturbing, and skip it if we can.
    normalized = _util._normalize_input_output_coordinates(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        perturb=False,
        seed=seed,
    )

    clipping = _clipping_applicable(
        coordinates_output=normalized[1],
        axis_output=normalized[3],
        shape_orthogonal=normalized[6],
    )

    if perturb and not clipping:
        normalized = _util._normalize_input_output_coordinates(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=axis_input,
            axis_output=axis_output,
            perturb=perturb,
            seed=seed,
        )

    (
        coordinates_input,
        coordinates_output,
        axis_input,
        axis_output,
        shape_input,
        shape_output,
        shape_orthogonal,
    ) = normalized

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

    weights = np.empty(shape_orthogonal, dtype=TypedList)

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
                grid_input_index = (
                    coordinates_input_x[index_vertices_input],
                    coordinates_input_y[index_vertices_input],
                )
                grid_output_index = (
                    coordinates_output_x[index_vertices_output],
                    coordinates_output_y[index_vertices_output],
                )

                if clipping:
                    weights[index] = weights_conservative_2d_clipping(
                        grid_input=grid_input_index,
                        grid_output=grid_output_index,
                        weights_input=weights_input_index,
                    )
                else:
                    weights[index] = weights_conservative_2d(
                        grid_input=grid_input_index,
                        grid_output=grid_output_index,
                        weights_input=weights_input_index,
                    )

            else:  # pragma: nocover
                raise NotImplementedError(
                    "Regridding operations greater than 2D are not supported"
                )

    return weights, shape_values_input, shape_values_output

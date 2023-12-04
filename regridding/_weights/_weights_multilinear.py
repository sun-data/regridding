from typing import Sequence
import numpy as np
import numba
import regridding
from regridding import _util


def _weights_multilinear(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    indices_output = regridding.find_indices(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    result = _weights_from_indices_multilinear(
        indices_output=indices_output,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    return result


def _weights_from_indices_multilinear(
    indices_output: tuple[np.ndarray, ...],
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

    axis_input_numba = ~np.arange(len(axis_input))[::-1]
    axis_output_numba = ~np.arange(len(axis_output))[::-1]

    shape_input_numba = tuple(shape_input[ax] for ax in axis_input)
    shape_output_numba = tuple(shape_output[ax] for ax in axis_output)

    indices_output = tuple(
        np.moveaxis(v, axis_output, axis_output_numba).reshape(-1, *shape_output_numba)
        for v in indices_output
    )
    coordinates_input = tuple(
        np.moveaxis(v, axis_input, axis_input_numba).reshape(-1, *shape_input_numba)
        for v in coordinates_input
    )
    coordinates_output = tuple(
        np.moveaxis(v, axis_output, axis_output_numba).reshape(-1, *shape_output_numba)
        for v in coordinates_output
    )

    if len(axis_input) == 1:
        weights_list = _weights_from_indices_multilinear_1d(
            indices_output=indices_output,
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
        )
    else:
        raise ValueError(
            f"{len(axis_input)}-dimensional multilinear interpolation is not supported"
        )

    num_d = len(weights_list)
    weights = np.empty(shape=num_d, dtype=numba.typed.List)
    for d in range(num_d):
        weights[d] = weights_list[d]
    weights = weights.reshape(shape_orthogonal)

    return weights, shape_input, shape_output


@numba.njit(parallel=True)
def _weights_from_indices_multilinear_1d(
    indices_output: tuple[np.ndarray],
    coordinates_input: tuple[np.ndarray],
    coordinates_output: tuple[np.ndarray],
) -> np.ndarray:
    (i_output,) = indices_output
    (x_input,) = coordinates_input
    (x_output,) = coordinates_output

    num_d, num_i_input = x_input.shape
    num_d, num_i_output = x_output.shape

    weights = numba.typed.List()

    for d in numba.prange(num_d):
        weights_d = numba.typed.List()
        for _ in range(0):
            weights_d.append((0.0, 0.0, 0.0))

        for i in numba.prange(num_i_output):
            i0 = i_output[d, i]
            i1 = i0 + 1

            x0 = x_input[d, i0]
            x1 = x_input[d, i1]
            x = x_output[d, i]

            w1 = (x - x0) / (x1 - x0)
            w0 = 1 - w1

            weights_d.append((i0, i, w0))
            weights_d.append((i1, i, w1))

        weights.append(weights_d)

    return weights

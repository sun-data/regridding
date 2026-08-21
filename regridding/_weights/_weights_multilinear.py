from typing import Literal, Sequence
import numpy as np
import numba
from numba.typed.typedlist import List as TypedList
import regridding
from regridding import _util


def _weights_multilinear(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    perturb: None | bool = False,
    seed: "None | int | np.random.Generator" = _util._seed_default,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    indices_output = regridding.find_indices(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="searchsorted",
    )
    result = _weights_from_indices_multilinear(
        indices_output=indices_output,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        bounds=bounds,
        perturb=perturb,
        seed=seed,
    )
    return result


def _weights_from_indices_multilinear(
    indices_output: tuple[np.ndarray, ...],
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    perturb: None | bool = False,
    seed: "None | int | np.random.Generator" = _util._seed_default,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:

    if bounds not in ("extrapolate", "nan", "raise"):
        raise ValueError(
            f"Unrecognized {bounds=}, "
            f"expected one of ('extrapolate', 'nan', 'raise')."
        )

    if perturb is None:
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
        seed=seed,
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
    if weights_input is not None:
        weights_input = np.broadcast_to(weights_input, shape_input)
        weights_input = np.moveaxis(weights_input, axis_input, axis_input_numba)
        weights_input = weights_input.reshape(-1, *shape_input_numba)

    if len(axis_input) == 1:
        # `find_indices` marks an output point that falls outside the input
        # grid with `fill_value`, which is not a usable index. Resolve those
        # here so that the compiled kernel only ever sees a valid cell.
        (index,) = indices_output
        (x_input,) = coordinates_input
        (x_output,) = coordinates_output
        index_max = x_input.shape[~0] - 2
        outside = (index < 0) | (index > index_max)
        if bounds == "raise" and outside.any():
            raise ValueError(
                f"{outside.sum()} of the output points fall outside the input "
                f"grid, and {bounds=}."
            )
        # `fill_value` is a large positive number whichever side of the grid
        # the point fell out on, so clamping it alone would send every such
        # point to the last cell. Recover the side from the coordinate.
        below = x_output < x_input[..., :1]
        indices_output = (np.where(below, 0, np.clip(index, 0, index_max)),)

        weights_list = _weights_from_indices_multilinear_1d(
            indices_output=indices_output,
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            weights_input=weights_input,
            outside=outside if bounds == "nan" else np.zeros_like(outside),
        )
    else:
        raise ValueError(
            f"{len(axis_input)}-dimensional multilinear interpolation is not supported"
        )

    num_d = len(weights_list)
    weights = np.empty(shape=num_d, dtype=TypedList)
    for d in range(num_d):
        weights[d] = weights_list[d]
    weights = weights.reshape(shape_orthogonal)

    return weights, shape_input, shape_output


@numba.njit(parallel=False)
def _weights_from_indices_multilinear_1d(
    indices_output: tuple[np.ndarray, ...],
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    weights_input: None | np.ndarray,
    outside: np.ndarray,
) -> TypedList:
    (i_output,) = indices_output
    (x_input,) = coordinates_input
    (x_output,) = coordinates_output

    num_d, num_i_input = x_input.shape
    num_d, num_i_output = x_output.shape

    # `numba.typed.List()` is declared to return a plain `list` when Numba's
    # JIT is disabled, a mode this library is not usable in.
    weights: TypedList = TypedList()  # type: ignore[assignment]
    for _ in range(0):  # pragma: nocover
        weights.append(
            (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        )

    for d in range(num_d):
        # each output vertex contributes exactly two weights, so the flat
        # arrays for this element have a known size and need no compaction.
        n = 2 * num_i_output
        indices_input = np.empty(n, dtype=np.int64)
        indices_output_d = np.empty(n, dtype=np.int64)
        values = np.empty(n, dtype=np.float64)

        for i in numba.prange(num_i_output):
            i0 = i_output[d, i]
            i1 = i0 + 1

            x0 = x_input[d, i0]
            x1 = x_input[d, i1]
            x = x_output[d, i]

            w1 = (x - x0) / (x1 - x0)
            w0 = 1 - w1

            if weights_input is not None:
                w0 = w0 * weights_input[d, i0]
                w1 = w1 * weights_input[d, i1]

            if outside[d, i]:
                w0 = np.nan
                w1 = np.nan

            indices_input[2 * i] = i0
            indices_output_d[2 * i] = i
            values[2 * i] = w0

            indices_input[2 * i + 1] = i1
            indices_output_d[2 * i + 1] = i
            values[2 * i + 1] = w1

        weights.append((indices_input, indices_output_d, values))

    return weights

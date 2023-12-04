import numpy as np
import numba


def _find_indices_brute(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    fill_value: None | int = None,
) -> tuple[np.ndarray, ...]:
    if len(coordinates_input) == 1:
        result = _find_indices_brute_1d(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            fill_value=fill_value,
        )
    else:
        raise ValueError(
            f"{len(coordinates_input)}-dimensional brute-force search not supported"
        )

    return result


@numba.njit(parallel=True, cache=True)
def _find_indices_brute_1d(
    coordinates_input: tuple[np.ndarray],
    coordinates_output: tuple[np.ndarray],
    fill_value: int,
) -> tuple[np.ndarray]:
    (x_input,) = coordinates_input
    (x_output,) = coordinates_output

    num_d, num_m = x_input.shape
    num_d, num_i = x_output.shape

    result = np.empty(shape=x_output.shape, dtype=np.int32)

    for d in numba.prange(num_d):
        x_input_d = x_input[d]
        x_output_d = x_output[d]

        for i in range(num_i):
            x_output_di = x_output_d[i]
            for m in range(num_m - 1):
                if x_input_d[m] <= x_output_di <= x_input_d[m + 1]:
                    result[d, i] = m
                    break
            else:
                result[d, i] = fill_value

    return (result,)

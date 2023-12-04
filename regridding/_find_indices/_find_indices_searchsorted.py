import numpy as np
import numba


def _find_indices_searchsorted(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    fill_value: None | int = None,
) -> tuple[np.ndarray, ...]:
    if len(coordinates_input) == 1:
        result = _find_indices_searchsorted_1d(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            fill_value=fill_value,
        )
    else:
        raise ValueError(
            f"{len(coordinates_input)}-dimensional searchsorted not supported"
        )

    return result


@numba.njit(parallel=True, cache=True)
def _find_indices_searchsorted_1d(
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

        result_d = np.searchsorted(
            a=x_input_d,
            v=x_output_d,
        )

        x_input_d_min = x_input_d[0]
        for i in range(num_i):
            x_output_di = x_output_d[i]
            result_di = result_d[i]
            result_di = result_di - 1
            if x_output_di == x_input_d_min:
                result_di = 0
            elif result_di < 0:
                result_di = fill_value
            elif result_di > num_m:
                result_di = fill_value
            result_d[i] = result_di

        result[d] = result_d

    return (result,)

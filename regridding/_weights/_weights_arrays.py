import numpy as np
import numba

__all__ = [
    "_weights_to_arrays",
]


def _weights_to_arrays(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Convert ragged typed-list weights into the public flat-array form.

    The internal weight builders accumulate ``(input, output, weight)``
    triples in :class:`numba.typed.List` objects, which is the natural
    container while overlaps are being discovered but cannot be pickled and
    costs roughly twice the memory of the equivalent flat arrays.  This
    converts each element into a ``(indices_input, indices_output, values)``
    tuple of flat arrays, releasing each list as it is converted so the peak
    memory is one copy plus the growing arrays.

    Parameters
    ----------
    weights
        Ragged array of weights accumulated by an internal builder.
    """

    weights, shape_input, shape_output = weights

    shape = weights.shape
    flat = weights.reshape(-1)

    result = np.empty(flat.size, dtype=object)
    for k in range(flat.size):
        result[k] = _triples_to_arrays(flat[k])
        flat[k] = None

    return result.reshape(shape), shape_input, shape_output


@numba.njit(cache=True)
def _triples_to_arrays(
    triples: numba.typed.List,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(triples)
    indices_input = np.empty(n, dtype=np.int64)
    indices_output = np.empty(n, dtype=np.int64)
    values = np.empty(n, dtype=np.float64)
    for w in range(n):
        i, j, weight = triples[w]
        indices_input[w] = i
        indices_output[w] = j
        values[w] = weight
    return indices_input, indices_output, values

import numpy as np
import numba

__all__ = [
    "weights_to_arrays",
    "weights_from_arrays",
]


def weights_to_arrays(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Convert ragged weights into flat arrays.

    The ragged weights computed by :func:`regridding.weights` are
    :class:`numba.typed.List` objects, which cannot be pickled.
    This function converts each list of ``(input, output, weight)`` triples
    into a tuple of three flat :class:`numpy.ndarray` instances,
    ``(indices_input, indices_output, values)``, which can be pickled,
    memory-mapped, and applied directly by
    :func:`regridding.regrid_from_weights`.

    The array form also uses about half the memory of the typed lists,
    which matters when the number of triples is large.

    Parameters
    ----------
    weights
        Ragged array of weights computed by :func:`regridding.weights`
        (or one of the transpose functions).

    See Also
    --------
    :func:`weights_from_arrays`: The inverse of this function.
    """

    weights, shape_input, shape_output = weights

    shape = weights.shape

    result = np.empty(weights.size, dtype=object)
    for k, triples in enumerate(weights.reshape(-1)):
        result[k] = _triples_to_arrays(triples)

    return result.reshape(shape), shape_input, shape_output


def weights_from_arrays(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Convert flat-array weights back into ragged typed lists.

    This is the inverse of :func:`weights_to_arrays`.

    Parameters
    ----------
    weights
        Array of ``(indices_input, indices_output, values)`` tuples computed
        by :func:`weights_to_arrays`.

    See Also
    --------
    :func:`weights_to_arrays`: The inverse of this function.
    """

    weights, shape_input, shape_output = weights

    shape = weights.shape

    result = np.empty(weights.size, dtype=object)
    for k, (indices_input, indices_output, values) in enumerate(weights.reshape(-1)):
        result[k] = _arrays_to_triples(
            np.ascontiguousarray(indices_input, dtype=np.int64),
            np.ascontiguousarray(indices_output, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )

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


@numba.njit(cache=True)
def _arrays_to_triples(
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    values: np.ndarray,
) -> numba.typed.List:
    triples = numba.typed.List()
    triples.append((numba.int64(0), numba.int64(0), 0.0))
    triples.pop()
    for w in range(values.shape[0]):
        triples.append((indices_input[w], indices_output[w], values[w]))
    return triples

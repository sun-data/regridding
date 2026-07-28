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

    Repeated ``(input, output)`` pairs — the conservative clipper emits
    several fragment triples per distinct pair — are merged by summing their
    weights, which is exact (applying the weights is linear) and shrinks the
    result by the mean multiplicity.  The output of each element is sorted
    by ``(indices_input, indices_output)`` with unique pairs, a canonical
    form.

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
        result[k] = _coalesce(*_triples_to_arrays(flat[k]))
        flat[k] = None

    return result.reshape(shape), shape_input, shape_output


def _coalesce(
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.shape[0] == 0:
        return indices_input, indices_output, values

    # the builders can emit negative (wraparound) indices for descending
    # grids, so key relative to the minimum instead of assuming zero
    base_input = np.int64(indices_input.min())
    base_output = np.int64(indices_output.min())
    span_output = np.int64(indices_output.max()) - base_output + 1
    key = (indices_input - base_input) * span_output + (indices_output - base_output)

    order = np.argsort(key, kind="stable")
    key = key[order]
    values = values[order]

    boundary = np.empty(key.shape[0], dtype=bool)
    boundary[0] = True
    np.not_equal(key[1:], key[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)

    key = key[starts]
    return (
        key // span_output + base_input,
        key % span_output + base_output,
        np.add.reduceat(values, starts),
    )


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

import numpy as np

__all__ = [
    "_weights_to_arrays",
]


def _weights_to_arrays(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    r"""
    Coalesce each element's flat ``(input, output, weight)`` arrays into a
    canonical form.

    The internal weight builders each emit a ``(indices_input,
    indices_output, values)`` tuple of flat arrays. The conservative clippers
    emit several fragment triples per distinct ``(input, output)`` pair, so
    this merges repeated pairs by summing their weights — which is exact
    (applying the weights is linear) and shrinks the result by the mean
    multiplicity. Each element is returned sorted by ``(indices_input,
    indices_output)`` with unique pairs, releasing each input as it is
    processed so the peak memory is one copy plus the growing arrays.

    Parameters
    ----------
    weights
        Array of per-element flat weight arrays from an internal builder.
    """

    weights, shape_input, shape_output = weights

    shape = weights.shape
    flat = weights.reshape(-1)

    result = np.empty(flat.size, dtype=object)
    for k in range(flat.size):
        indices_input, indices_output, values = flat[k]
        result[k] = _coalesce(indices_input, indices_output, values)
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

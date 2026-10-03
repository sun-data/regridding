from typing import Any
import pytest
import numpy as np
from numba import cuda
import regridding
from .._weights_convolved._weights_convolved_test import (
    grid_input,
    grid_output,
    _rotated,
    _lattice,
)
from .._weights_convolved._cuda_test import _to_host

requires_cuda = pytest.mark.cuda
"""
Mark a test as needing a CUDA device.

The mark is what the `tests-cuda` workflow selects on and what `conftest`
skips on, so a test says once that it needs a device.
"""

axes: dict[str, Any] = dict(axis_input=(1, 2), axis_output=(1, 2))
"""The resampled axes of the grids, which have one orthogonal axis first."""

grids: dict[str, Any] = dict(
    coordinates_input=grid_input,
    coordinates_output=grid_output,
    **axes,
)
"""The grids the weights are built on, with their resampled axes."""

weights_input = np.random.default_rng(13).uniform(0.5, 1.5, size=(16, 16))
"""
A weight for each input cell which varies differently along each axis, and
is the same for both rotations, so that it is broadcast along the
orthogonal axis.
"""


def _weights(**kwargs: Any) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """The weights of the host tests, built with `kwargs`, such as a device."""
    return regridding.weights(method="conservative", **grids, **kwargs)


def _assert_matches(
    actual: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    expected: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> None:
    """
    Check that transposed weights on the device, once their empty slots are
    dropped, are the same as transposed weights on the host.

    The values are computed in the same order on both, so they agree
    exactly.
    """
    assert actual[0].shape == expected[0].shape
    assert actual[1:] == expected[1:]
    for a, e in zip(_to_host(actual)[0].reshape(-1), expected[0].reshape(-1)):
        assert np.array_equal(a[0], e[0])
        assert np.array_equal(a[1], e[1])
        assert np.array_equal(a[2], e[2])


@requires_cuda
@pytest.mark.parametrize(
    argnames="weights_input",
    argvalues=[None, weights_input],
    ids=["unweighted", "weighted"],
)
def test_matches_host(weights_input: None | np.ndarray) -> None:
    """The device transposes the same weights into the same result."""
    weights = _weights(device="cuda", weights_input=weights_input)

    actual = regridding.transpose_weights_conservative(
        weights,
        weights_input=weights_input,
        **grids,
    )
    expected = regridding.transpose_weights_conservative(
        _to_host(weights),
        weights_input=weights_input,
        **grids,
    )

    _assert_matches(actual, expected)


@requires_cuda
def test_on_device() -> None:
    """
    The result is left on the device, sharing the index arrays of the
    weights, and resamples there into what the host resamples into.
    """
    weights = _weights(device="cuda")

    result = regridding.transpose_weights_conservative(weights, **grids)

    for triple, triple_weights in zip(
        result[0].reshape(-1),
        weights[0].reshape(-1),
    ):
        for array in triple:
            assert cuda.is_cuda_array(array)
        assert triple[0] is triple_weights[1]
        assert triple[1] is triple_weights[0]
        assert triple[2].dtype == np.float64

    image = np.random.default_rng(14).random(weights[2])

    # `regrid_from_weights` is annotated with the host's return type, but
    # weights on a device leave a device array
    actual: Any = regridding.regrid_from_weights(*result, values_input=image, **axes)
    assert cuda.is_cuda_array(actual)

    expected = regridding.regrid_from_weights(
        *regridding.transpose_weights_conservative(_to_host(weights), **grids),
        values_input=image,
        **axes,
    )
    assert np.allclose(actual.copy_to_host(), expected, rtol=1e-12, atol=1e-15)


@requires_cuda
def test_dtype() -> None:
    """
    Narrower stored types are transposed as on the host: the indices keep
    their type, and the values are computed and stored in double precision.
    """
    weights = _weights(device="cuda", dtype_indices=np.int32, dtype_values=np.float32)

    actual = regridding.transpose_weights_conservative(weights, **grids)
    expected = regridding.transpose_weights_conservative(_to_host(weights), **grids)

    _assert_matches(actual, expected)
    for indices_input, indices_output, values in actual[0].reshape(-1):
        assert indices_input.dtype == np.int32
        assert indices_output.dtype == np.int32
        assert values.dtype == np.float64


@requires_cuda
def test_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Weights with more orthogonal elements than their grids, as
    :func:`regridding.convolve_weights` leaves them, are transposed as on the
    host, and each grid's volumes are sent to the device once rather than
    once for each element.
    """
    grid_single = _rotated(), _lattice()
    array, shape_input, shape_output = regridding.weights(
        *grid_single,
        method="conservative",
        device="cuda",
    )
    weights = regridding.convolve_weights(
        (array, (1, *shape_input), (1, *shape_output)),
        np.random.default_rng(15).random((3, 1, 1, 3, 3)),
        **axes,
    )
    kwargs: dict[str, Any] = dict(
        coordinates_input=tuple(c[np.newaxis] for c in grid_single[0]),
        coordinates_output=tuple(c[np.newaxis] for c in grid_single[1]),
        **axes,
    )

    sent = []
    to_device = cuda.to_device

    def _to_device(a: Any, *args: Any, **kw: Any) -> Any:
        sent.append(a.shape)
        return to_device(a, *args, **kw)

    monkeypatch.setattr(cuda, "to_device", _to_device)
    actual = regridding.transpose_weights_conservative(weights, **kwargs)
    monkeypatch.undo()

    assert actual[0].shape == (3,)
    assert sent == [(16 * 16,), (24 * 22,)]

    expected = regridding.transpose_weights_conservative(_to_host(weights), **kwargs)
    _assert_matches(actual, expected)


@requires_cuda
def test_empty() -> None:
    """
    An element with no weights is transposed into one with none, rather than
    launching a kernel with no blocks.
    """
    weights = _weights(device="cuda")
    array = weights[0].copy()
    array[1] = tuple(
        cuda.to_device(np.zeros(0, dtype=dtype))
        for dtype in (np.int64, np.int64, np.float64)
    )

    actual = regridding.transpose_weights_conservative((array, *weights[1:]), **grids)

    for a in actual[0][1]:
        assert cuda.is_cuda_array(a)
        assert a.size == 0

    expected = regridding.transpose_weights_conservative(_to_host(weights), **grids)
    _assert_matches(
        (actual[0][:1], *actual[1:]),
        (expected[0][:1], *expected[1:]),
    )


@requires_cuda
def test_transpose_weights() -> None:
    """
    The plain transpose swaps device arrays as it swaps host ones, and its
    empty slots, now on the output side, are skipped when it is applied.
    """
    weights = _weights(device="cuda")

    result = regridding.transpose_weights(weights)

    image = np.random.default_rng(16).random(weights[2])

    actual: Any = regridding.regrid_from_weights(*result, values_input=image, **axes)
    assert cuda.is_cuda_array(actual)

    expected = regridding.regrid_from_weights(
        *regridding.transpose_weights(_to_host(weights)),
        values_input=image,
        **axes,
    )
    assert np.allclose(actual.copy_to_host(), expected, rtol=1e-12, atol=1e-15)

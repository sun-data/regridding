from typing import Any
import pytest
import numpy as np
import scipy.ndimage
from numba import cuda
import regridding

requires_cuda = pytest.mark.cuda
"""
Mark a test as needing a CUDA device.

The mark is what the `tests-cuda` workflow selects on and what `conftest`
skips on, so a test says once that it needs a device.
"""


def _grids() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, ...]]:
    """A rotated grid, twice, onto a uniform lattice which can be built on."""
    t = np.linspace(-4, 4, 17)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.array([0.1, 0.7])[:, np.newaxis, np.newaxis]
    grid_input = (
        x * np.cos(angle) - y * np.sin(angle),
        x * np.sin(angle) + y * np.cos(angle),
    )
    grid_output = tuple(
        c[np.newaxis]
        for c in np.meshgrid(
            np.linspace(-6, 6, 25),
            np.linspace(-6, 5, 23),
            indexing="ij",
        )
    )
    return grid_input, grid_output


def _weights(**kwargs: Any) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    grid_input, grid_output = _grids()
    return regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=(1, 2),
        axis_output=(1, 2),
        method="conservative",
        **kwargs,
    )


def _to_host(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Bring a set of weights back from the device, dropping empty slots."""
    array, shape_input, shape_output = weights
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(*array.shape):
        indices_input, indices_output, values = (a.copy_to_host() for a in array[index])
        keep = (indices_input >= 0) & (indices_output >= 0)
        result[index] = (indices_input[keep], indices_output[keep], values[keep])
    return result, shape_input, shape_output


kernels = [
    np.random.default_rng(1).random((3, 3)),
    np.random.default_rng(2).random((5, 4)),
    np.random.default_rng(3).random((2, 1, 1, 3, 3)),
    np.random.default_rng(4).random((24, 1, 3, 3)),
]


@requires_cuda
@pytest.mark.parametrize("kernel", kernels)
def test_matches_host(kernel: np.ndarray) -> None:
    """The device convolves the same weights into the same result."""
    weights = _weights(device="cuda")

    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))
    expected = regridding.convolve_weights(
        _to_host(weights),
        kernel,
        axis_output=(1, 2),
    )

    for actual, desired in zip(
        _to_host(result)[0].reshape(-1),
        expected[0].reshape(-1),
    ):
        assert np.array_equal(actual[0], desired[0])
        assert np.array_equal(actual[1], desired[1])
        assert np.allclose(actual[2], desired[2], rtol=1e-14, atol=0)


@requires_cuda
def test_on_device() -> None:
    """
    The result is left on the device, with no empty slots, and resamples
    there into a convolved array.
    """
    weights = _weights(device="cuda")
    kernel = np.random.default_rng(5).random((3, 5))

    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    for triple in result[0].reshape(-1):
        for array in triple:
            assert cuda.is_cuda_array(array)
        assert np.all(triple[0].copy_to_host() >= 0)

    values = np.random.default_rng(6).random((2, 16, 16))
    kwargs = dict(values_input=values, axis_input=(1, 2), axis_output=(1, 2))

    actual = regridding.regrid_from_weights(*result, **kwargs)
    assert cuda.is_cuda_array(actual)

    expected = scipy.ndimage.convolve(
        regridding.regrid_from_weights(*_to_host(weights), **kwargs),
        kernel[np.newaxis],
        mode="constant",
    )
    assert np.allclose(actual.copy_to_host(), expected, rtol=1e-12, atol=1e-15)


@requires_cuda
def test_dtype() -> None:
    """Narrower stored types are kept on the device too."""
    weights = _weights(device="cuda", dtype_indices=np.int32, dtype_values=np.float32)

    result = regridding.convolve_weights(weights, np.ones((3, 3)), axis_output=(1, 2))

    for indices_input, indices_output, values in result[0].reshape(-1):
        assert indices_input.dtype == np.int32
        assert indices_output.dtype == np.int32
        assert values.dtype == np.float32


@requires_cuda
def test_empty_slots() -> None:
    """
    An empty slot is skipped whichever side carries its ``-1``: weights
    built on a device carry it on the input side, and transposed ones on the
    output side.
    """
    indices_input = np.array([-1, 3, 3, -1, 5, 0], dtype=np.int64)
    indices_output = np.array([0, 5, 6, 0, 7, -1], dtype=np.int64)
    values = np.array([0, 0.25, 0.75, 0, 1, 0], dtype=np.float64)

    weights = np.empty((), dtype=object)
    weights[()] = tuple(
        cuda.to_device(a) for a in (indices_input, indices_output, values)
    )
    shape = (4, 4)
    kernel = np.random.default_rng(7).random((3, 3))

    result = regridding.convolve_weights((weights, shape, shape), kernel)

    keep = (indices_input >= 0) & (indices_output >= 0)
    expected = np.empty((), dtype=object)
    expected[()] = (indices_input[keep], indices_output[keep], values[keep])
    expected = regridding.convolve_weights((expected, shape, shape), kernel)

    actual = _to_host(result)[0][()]
    assert np.array_equal(actual[0], expected[0][()][0])
    assert np.array_equal(actual[1], expected[0][()][1])
    assert np.allclose(actual[2], expected[0][()][2], rtol=1e-14, atol=0)


@requires_cuda
@pytest.mark.parametrize("num", [0, 2])
def test_nothing_to_spread(num: int) -> None:
    """An element with no weights, or only empty slots, becomes empty."""
    weights = np.empty((), dtype=object)
    weights[()] = (
        cuda.to_device(np.full(num, -1, dtype=np.int64)),
        cuda.to_device(np.zeros(num, dtype=np.int64)),
        cuda.to_device(np.zeros(num, dtype=np.float64)),
    )
    shape = (4, 4)

    result = regridding.convolve_weights((weights, shape, shape), np.ones((3, 3)))

    for array in result[0][()]:
        assert cuda.is_cuda_array(array)
        assert array.size == 0

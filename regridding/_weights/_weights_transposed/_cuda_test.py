from typing import Any
import pytest
import numpy as np
import astropy.units as u
from numba import cuda
import regridding

requires_cuda = pytest.mark.cuda
"""
Mark a test as needing a CUDA device.

The mark is what the `tests-cuda` workflow selects on and what `conftest`
skips on, so a test says once that it needs a device.
"""


def _rotated(
    angle: float | np.ndarray = 0.3,
    scale: float | np.ndarray = 1,
    num: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """A square grid of vertices, rotated by `angle` and scaled by `scale`."""
    t = np.linspace(-4, 4, num)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.asarray(angle)[..., np.newaxis, np.newaxis]
    scale = np.asarray(scale)[..., np.newaxis, np.newaxis]
    return (
        scale * (x * np.cos(angle) - y * np.sin(angle)),
        scale * (x * np.sin(angle) + y * np.cos(angle)),
    )


def _lattice() -> tuple[np.ndarray, np.ndarray]:
    """A uniform, axis-aligned grid of vertices, larger than `_rotated`."""
    x, y = np.meshgrid(
        np.linspace(-6, 6, 25),
        np.linspace(-6, 5, 23),
        indexing="ij",
    )
    return x, y


grid_input = _rotated(angle=np.array([0.1, 0.7]), scale=np.array([1, 0.8]))
"""
Two input grids, rotated and scaled differently, so that the weights have
an orthogonal axis along which the areas of the input cells differ.
"""

grid_output = tuple(c[np.newaxis] for c in _lattice())
"""One output grid, which both input grids are resampled onto."""

axes: dict[str, Any] = dict(axis_input=(1, 2), axis_output=(1, 2))
"""The resampled axes of the grids, which have one orthogonal axis first."""

grids: dict[str, Any] = dict(
    coordinates_input=grid_input,
    coordinates_output=grid_output,
    **axes,
)
"""The grids the weights are built on, with their resampled axes."""

rng = np.random.default_rng(13)

weights_inputs = dict(
    unweighted=None,
    shared=rng.uniform(0.5, 1.5, size=(16, 16)),
    varying=rng.uniform(0.5, 1.5, size=(2, 16, 16)),
    single=rng.uniform(0.5, 1.5, size=(16, 16)).astype(np.float32),
)
"""
The weights of the input cells to test with: none, one set which varies
across the cells and is broadcast along the orthogonal axis, one which also
varies along it, and one in single precision.
"""


weights_input_cells = rng.uniform(0.5, 1.5, size=(16, 16))
"""A weight for each input cell, for the tests to broadcast themselves."""


def _weights(**kwargs: Any) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """The weights between the test grids, built with `kwargs`."""
    return regridding.weights(method="conservative", **grids, **kwargs)


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


def _count_sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """
    Record the shape of every array sent to the device from here on.

    Parameters
    ----------
    monkeypatch
        The fixture which undoes the recording once the test is done.
    """
    sent = []
    to_device = cuda.to_device

    def _to_device(a: Any, *args: Any, **kwargs: Any) -> Any:
        sent.append(a.shape)
        return to_device(a, *args, **kwargs)

    monkeypatch.setattr(cuda, "to_device", _to_device)
    return sent


@requires_cuda
@pytest.mark.parametrize(
    argnames="weights_input",
    argvalues=list(weights_inputs.values()),
    ids=list(weights_inputs),
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
def test_percent() -> None:
    """
    A dimensionless `weights_input` with a scale, such as a percentage, is
    the number it stands for, as it is when the weights are built.
    """
    percent = np.full((16, 16), 50) * u.percent
    number = np.full((16, 16), 0.5)

    actual = regridding.transpose_weights_conservative(
        _weights(device="cuda", weights_input=percent),
        weights_input=percent,
        **grids,
    )
    expected = regridding.transpose_weights_conservative(
        _weights(device="cuda", weights_input=number),
        weights_input=number,
        **grids,
    )

    for a, e in zip(_to_host(actual)[0].reshape(-1), _to_host(expected)[0]):
        assert np.allclose(a[2], e[2], rtol=1e-14, atol=0)


@requires_cuda
def test_grids() -> None:
    """
    Grids which are not the ones the weights were built on raise before
    anything is sent to the device.
    """
    weights = _weights(device="cuda")

    with pytest.raises(ValueError, match="the weights address"):
        regridding.transpose_weights_conservative(
            weights,
            coordinates_input=grid_output,
            coordinates_output=grid_input,
            **axes,
        )


@requires_cuda
def test_outside() -> None:
    """
    Weights which address cells outside the grids they say they were built
    for raise, rather than reading past the end of them on the device.
    """
    weights = _weights(device="cuda")
    array = weights[0].copy()
    indices_input, indices_output, values = (a.copy_to_host() for a in array[1])
    # a slot which saw an overlap, since an empty one is skipped unread
    indices_output[np.flatnonzero(indices_input >= 0)[0]] = 24 * 22
    array[1] = tuple(cuda.to_device(a) for a in (indices_input, indices_output, values))

    with pytest.raises(ValueError, match="outside the input"):
        regridding.transpose_weights_conservative((array, *weights[1:]), **grids)


@requires_cuda
@pytest.mark.parametrize(
    argnames="broadcast",
    argvalues=[False, True],
    ids=["given", "broadcast"],
)
def test_sent_once(monkeypatch: pytest.MonkeyPatch, broadcast: bool) -> None:
    """
    The volumes are sent to the device once, before they are broadcast, so
    the output grid, which both input grids share, is sent once rather than
    once for each of them.  That holds when the caller has already broadcast
    the output grid along the orthogonal axis, as :mod:`named_arrays` does,
    too.
    """
    weights = _weights(device="cuda")

    kwargs = dict(grids)
    if broadcast:
        shape = grid_input[0].shape[:1] + grid_output[0].shape[1:]
        kwargs["coordinates_output"] = tuple(
            np.broadcast_to(c, shape) for c in grid_output
        )

    sent = _count_sent(monkeypatch)
    actual = regridding.transpose_weights_conservative(weights, **kwargs)
    monkeypatch.undo()

    assert sent == [(2, 16 * 16), (1, 24 * 22)]

    expected = regridding.transpose_weights_conservative(_to_host(weights), **grids)
    _assert_matches(actual, expected)


@requires_cuda
def test_single() -> None:
    """Weights with no orthogonal axes are transposed as on the host."""
    grid_single = _rotated(), _lattice()
    weights = regridding.weights(*grid_single, method="conservative", device="cuda")

    actual = regridding.transpose_weights_conservative(weights, *grid_single)
    expected = regridding.transpose_weights_conservative(
        _to_host(weights),
        *grid_single,
    )

    assert actual[0].shape == ()
    _assert_matches(actual, expected)


def _convolved() -> tuple[
    tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    dict[str, Any],
]:
    """
    Weights on the device with three orthogonal elements but grids with
    one, as :func:`regridding.convolve_weights` leaves them when its kernel
    varies along an axis the grids do not, and the grids to transpose them
    with.
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
    return weights, kwargs


@requires_cuda
def test_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Weights with more orthogonal elements than their grids, as
    :func:`regridding.convolve_weights` leaves them, are transposed as on the
    host, and each grid's volumes are sent to the device once rather than
    once for each element.
    """
    weights, kwargs = _convolved()

    sent = _count_sent(monkeypatch)
    actual = regridding.transpose_weights_conservative(weights, **kwargs)
    monkeypatch.undo()

    assert actual[0].shape == (3,)
    assert sent == [(1, 16 * 16), (1, 24 * 22)]

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


@requires_cuda
@pytest.mark.parametrize(
    argnames="weights_input",
    argvalues=[
        np.broadcast_to(weights_input_cells.astype(np.float32), (3, 16, 16)),
        np.broadcast_to(weights_input_cells * 100 * u.percent, (3, 16, 16), subok=True),
    ],
    ids=["single", "percent"],
)
def test_weights_input_sent_once(
    monkeypatch: pytest.MonkeyPatch,
    weights_input: np.ndarray,
) -> None:
    """
    A `weights_input` which the caller broadcast along the orthogonal axis
    is sent to the device once, even when it has to be converted to double
    precision or from a unit, which would copy it out to its full shape.
    """
    weights, kwargs = _convolved()

    sent = _count_sent(monkeypatch)
    actual = regridding.transpose_weights_conservative(
        weights,
        weights_input=weights_input,
        **kwargs,
    )
    monkeypatch.undo()

    assert sent == [(1, 16 * 16), (1, 24 * 22)]

    expected = regridding.transpose_weights_conservative(
        _to_host(weights),
        weights_input=weights_input,
        **kwargs,
    )
    _assert_matches(actual, expected)

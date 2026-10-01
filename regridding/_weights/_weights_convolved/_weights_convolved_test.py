import pytest
import numpy as np
import scipy.ndimage
import scipy.sparse
import astropy.units as u
import regridding
from regridding._weights._weights_arrays import _coalesce


def _rotated(num: int = 17, angle: float | np.ndarray = 0.3):
    """A square grid of vertices, rotated by `angle`."""
    t = np.linspace(-4, 4, num)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.asarray(angle)[..., np.newaxis, np.newaxis]
    return (
        x * np.cos(angle) - y * np.sin(angle),
        x * np.sin(angle) + y * np.cos(angle),
    )


def _lattice(num_x: int = 25, num_y: int = 23):
    """A uniform, axis-aligned grid of vertices, larger than `_rotated`."""
    return np.meshgrid(
        np.linspace(-6, 6, num_x),
        np.linspace(-6, 5, num_y),
        indexing="ij",
    )


angles = np.array([0.1, 0.7])
"""Two rotations, so the weights have an orthogonal axis."""

grid_input = _rotated(angle=angles)
grid_output = tuple(c[np.newaxis] for c in _lattice())

weights = regridding.weights(
    coordinates_input=grid_input,
    coordinates_output=grid_output,
    axis_input=(1, 2),
    axis_output=(1, 2),
    method="conservative",
)

values_input = np.random.default_rng(0).random((2, 16, 16))


def _regrid(weights):
    return regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        axis_input=(1, 2),
        axis_output=(1, 2),
    )


def _matrix(triple, num_input, num_output):
    """One element of a set of weights as a sparse matrix, output by input."""
    indices_input, indices_output, values = triple
    return scipy.sparse.coo_matrix(
        (values, (indices_output, indices_input)),
        shape=(num_output, num_input),
    ).tocsr()


@pytest.mark.parametrize(
    argnames="kernel",
    argvalues=[
        np.random.default_rng(1).random((3, 3)),
        np.random.default_rng(2).random((5, 4)),
        np.random.default_rng(3).random((2, 2)),
        np.random.default_rng(4).random((1, 3)),
        np.ones((1, 1)),
        np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8,
    ],
)
def test_convolve_weights(kernel: np.ndarray):
    """Convolving the weights is the same as convolving what they produce."""
    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    expected = scipy.ndimage.convolve(
        _regrid(weights),
        kernel[np.newaxis],
        mode="constant",
    )

    assert result[1] == weights[1]
    assert result[2] == weights[2]
    assert np.allclose(_regrid(result), expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_axis_order():
    """The axes of the kernel follow `axis_output` in the order given."""
    kernel = np.random.default_rng(5).random((5, 3))

    result = regridding.convolve_weights(weights, kernel.T, axis_output=(2, 1))
    expected = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    assert np.allclose(_regrid(result), _regrid(expected), rtol=1e-14)


def test_convolve_weights_ordered():
    """Weights ordered by input cell come out ordered, with no pair twice."""
    kernel = np.random.default_rng(6).random((3, 5))

    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    for indices_input, indices_output, values in result[0].reshape(-1):
        coalesced = _coalesce(indices_input, indices_output, values)
        assert np.array_equal(coalesced[0], indices_input)
        assert np.array_equal(coalesced[1], indices_output)
        assert np.array_equal(coalesced[2], values)


@pytest.mark.parametrize("axis", ["orthogonal", "resampled"])
def test_convolve_weights_varying(axis: str):
    """A kernel which varies is the matrix product with its own matrix."""
    rng = np.random.default_rng(7)

    shape = weights[2]
    num_x, num_y = shape[1], shape[2]

    if axis == "orthogonal":
        kernel = rng.random((2, 1, 1, 3, 3))
    else:
        kernel = rng.random((num_x, 1, 3, 3))

    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    kernel_full = np.broadcast_to(kernel, (2, num_x, num_y, 3, 3))

    for d in range(2):
        rows, cols, data = [], [], []
        for i in range(num_x):
            for j in range(num_y):
                for a in range(3):
                    for b in range(3):
                        ii, jj = i + a - 1, j + b - 1
                        if 0 <= ii < num_x and 0 <= jj < num_y:
                            rows.append(ii * num_y + jj)
                            cols.append(i * num_y + j)
                            data.append(kernel_full[d, i, j, a, b])
        size = num_x * num_y
        convolution = scipy.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(size, size),
        )

        expected = convolution @ _matrix(weights[0][d], 16 * 16, size)
        actual = _matrix(result[0][d], 16 * 16, size)

        assert abs(actual - expected).max() < 1e-15


@pytest.mark.parametrize("num", [3, 1])
def test_convolve_weights_broadcast(num: int):
    """
    The weights are broadcast along an orthogonal axis which the kernel
    varies along, or which the kernel only has a placeholder for.
    """
    weights_single = regridding.weights(
        coordinates_input=_rotated(),
        coordinates_output=_lattice(),
        method="conservative",
    )
    array, shape_input, shape_output = weights_single
    weights_broadcast = array, (1, *shape_input), (1, *shape_output)

    kernel = np.random.default_rng(8).random((num, 1, 1, 3, 3))

    result = regridding.convolve_weights(weights_broadcast, kernel, axis_output=(1, 2))

    assert result[0].shape == ((num,) if num > 1 else ())

    values = np.random.default_rng(9).random((num, 16, 16))
    actual = regridding.regrid_from_weights(
        *result,
        values_input=values,
        axis_input=(1, 2),
        axis_output=(1, 2),
    )
    for d in range(num):
        expected = scipy.ndimage.convolve(
            regridding.regrid_from_weights(*weights_single, values_input=values[d]),
            kernel[d, 0, 0],
            mode="constant",
        )
        assert np.allclose(actual[d], expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_empty_slots():
    """
    The shared kernel body skips empty slots when told to, as it is told to
    on a device.

    Compiled for a device, it runs where :mod:`coverage` cannot follow, so it
    is run here as plain Python instead, on weights with an empty slot on
    each side: weights built on a device carry the ``-1`` on the input side,
    and transposed ones on the output side.
    """
    from ._shared import build, num_axis

    starts_run, convolve_run = build(lambda function: function)

    indices_input = np.array([-1, 3, 3, -1, 5, 0, 5])
    indices_output = np.array([0, 5, 6, 0, 7, -1, 9])
    values = np.array([0, 0.25, 0.75, 0, 1, 0, 0.5])

    shape_output = np.array([4, 4])
    shape_kernel = np.array([3, 3])
    kernel = np.random.default_rng(13).random((1, 9))

    lower = np.empty(num_axis, dtype=np.int64)
    extent = np.empty(num_axis, dtype=np.int64)

    triples = []
    for w in range(values.size):
        if not starts_run(indices_input, indices_output, w, True):
            continue
        result = [np.empty(16, dtype=int), np.empty(16, dtype=int), np.empty(16)]
        count = convolve_run(
            indices_input,
            indices_output,
            values,
            kernel,
            False,
            shape_output,
            shape_kernel,
            True,
            w,
            lower,
            extent,
            True,
            0,
            *result,
        )
        triples.append([r[:count] for r in result])
    actual = [np.concatenate(r) for r in zip(*triples)]

    keep = (indices_input >= 0) & (indices_output >= 0)
    weights_valid = np.empty((), dtype=object)
    weights_valid[()] = (indices_input[keep], indices_output[keep], values[keep])
    expected = regridding.convolve_weights(
        (weights_valid, (8,), (4, 4)),
        kernel.reshape(3, 3),
    )[0][()]

    # the empty slot between the two weights of input cell 5 splits them into
    # two runs, which overlap, so the pairs are compared once summed
    def dense(triple):
        result = np.zeros((8, 16))
        np.add.at(result, (triple[0], triple[1]), triple[2])
        return result

    assert np.allclose(dense(actual), dense(expected), rtol=1e-14, atol=0)
    assert actual[2].size > expected[2].size


@pytest.mark.parametrize(
    argnames="x_input, x_output",
    argvalues=[
        (np.linspace(-1, 1, 21), np.linspace(-1, 1, 11)),
        (np.linspace(-1, 1, 21), np.linspace(1, -1, 11) + 1e-6),
        (np.linspace(1, -1, 21), np.linspace(-1, 1, 11) + 1e-6),
    ],
)
def test_convolve_weights_1d(x_input: np.ndarray, x_output: np.ndarray):
    """
    One-dimensional weights, including the descending grids whose indices
    count from the end.
    """
    weights_1d = regridding.weights((x_input,), (x_output,), method="conservative")
    kernel = np.array([0.2, 0.5, 0.3])

    result = regridding.convolve_weights(weights_1d, kernel)

    values = np.random.default_rng(10).random(20)
    expected = scipy.ndimage.convolve1d(
        regridding.regrid_from_weights(*weights_1d, values_input=values),
        kernel,
        mode="constant",
    )
    actual = regridding.regrid_from_weights(*result, values_input=values)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_transpose():
    """
    The conservative transpose of the convolved weights is the transpose of
    the weights after correlating with the kernel, its own transpose.
    """
    kernel = np.random.default_rng(11).random((3, 5))

    result = regridding.convolve_weights(weights, kernel, axis_output=(1, 2))

    kwargs = dict(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=(1, 2),
        axis_output=(1, 2),
    )
    transposed = regridding.transpose_weights_conservative(result, **kwargs)
    transposed_geometric = regridding.transpose_weights_conservative(weights, **kwargs)

    image = np.random.default_rng(12).random(weights[2][1:])

    actual = regridding.regrid_from_weights(
        *transposed,
        values_input=image,
        axis_input=(1, 2),
        axis_output=(1, 2),
    )
    expected = regridding.regrid_from_weights(
        *transposed_geometric,
        values_input=scipy.ndimage.correlate(image, kernel, mode="constant"),
        axis_input=(1, 2),
        axis_output=(1, 2),
    )

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_convolve_weights_dtype():
    """Narrower stored types are kept."""
    weights_narrow = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=(1, 2),
        axis_output=(1, 2),
        method="conservative",
        dtype_indices=np.int32,
        dtype_values=np.float32,
    )

    result = regridding.convolve_weights(
        weights_narrow,
        np.ones((3, 3)) / 9,
        axis_output=(1, 2),
    )

    for indices_input, indices_output, values in result[0].reshape(-1):
        assert indices_input.dtype == np.int32
        assert indices_output.dtype == np.int32
        assert values.dtype == np.float32


def test_convolve_weights_unit():
    """A unit on the weights is kept, and a dimensionless kernel is accepted."""
    weights_unit = regridding.weights(
        coordinates_input=_rotated(),
        coordinates_output=_lattice(),
        method="conservative",
        weights_input=2 * u.cm**2,
    )

    kernel = np.ones((3, 3)) / 9 * u.dimensionless_unscaled
    result = regridding.convolve_weights(weights_unit, kernel)

    values = result[0][()][2]
    assert isinstance(values, u.Quantity)
    assert values.unit == u.cm**2


def test_convolve_weights_empty():
    """An input grid entirely outside the output grid has nothing to spread."""
    x_input, y_input = _rotated()
    weights_empty = regridding.weights(
        coordinates_input=(x_input + 100, y_input),
        coordinates_output=_lattice(),
        method="conservative",
    )

    result = regridding.convolve_weights(weights_empty, np.ones((3, 3)))

    indices_input, indices_output, values = result[0][()]
    assert values.size == 0
    assert indices_input.size == 0
    assert indices_output.size == 0


@pytest.mark.parametrize(
    argnames="kernel, axis_output",
    argvalues=[
        (np.ones(3), (1, 2)),
        (np.ones((5, 1, 3, 3)), (1, 2)),
        (np.ones((3, 1, 1, 3, 3)), (1, 2)),
        (np.ones((1, 1, 1, 1, 3, 3)), (1, 2)),
        (np.ones((3, 3)) * u.mm, (1, 2)),
    ],
)
def test_convolve_weights_errors(kernel: np.ndarray, axis_output: tuple[int, ...]):
    with pytest.raises(ValueError):
        regridding.convolve_weights(weights, kernel, axis_output=axis_output)

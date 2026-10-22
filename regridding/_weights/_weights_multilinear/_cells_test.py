import pytest
import numpy as np
from . import _cells


def _cell(
    num_axis: int,
    seed: int = 0,
    scale: float = 1,
    shift: float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A grid of a single cell, the unit cube with each corner moved at random,
    as its vertices, its shape, and the offsets of its corners.
    """
    rng = np.random.default_rng(seed)
    shape = np.full(num_axis, 2, dtype=np.int64)
    corners = np.stack(np.meshgrid(*[np.arange(2.0)] * num_axis, indexing="ij"))
    corners = corners + 0.15 * rng.uniform(-1, 1, corners.shape)
    vertices = shift + scale * corners.reshape(num_axis, -1)
    return vertices, shape, _cells.offsets(shape)


def _work(num_axis: int) -> np.ndarray:
    """A workspace for :func:`_cells.inverse`."""
    return np.empty((2 * num_axis + _cells.num_work, num_axis))


def test_strides() -> None:
    result = _cells.strides(np.array([4, 5, 6]))
    np.testing.assert_array_equal(result, [30, 6, 1])


def test_offsets() -> None:
    result = _cells.offsets(np.array([4, 5, 6]))
    np.testing.assert_array_equal(result, [0, 30, 6, 36, 1, 31, 7, 37])


@pytest.mark.parametrize("num_axis", [1, 2, 3])
class TestEvaluate:

    def test_corners(self, num_axis: int) -> None:
        vertices, shape, offset = _cell(num_axis)
        x = np.empty(num_axis)
        jacobian = np.empty((num_axis, num_axis))
        result = np.empty((offset.size, num_axis))
        for b in range(offset.size):
            xi = np.array([(b >> m) & 1 for m in range(num_axis)], dtype=float)
            _cells.evaluate(vertices, 0, offset, xi, x, jacobian)
            result[b] = x
        expected = (vertices[:, offset] - vertices[:, :1]).T
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)

    def test_position(self, num_axis: int) -> None:
        vertices, shape, offset = _cell(num_axis)
        xi = np.random.default_rng(1).uniform(-0.5, 1.5, num_axis)
        x = np.empty(num_axis)
        jacobian = np.empty((num_axis, num_axis))
        _cells.evaluate(vertices, 0, offset, xi, x, jacobian)
        result = np.empty(num_axis)
        _cells.position(vertices, 0, offset, xi, result)
        np.testing.assert_array_equal(result, x)

    def test_jacobian(self, num_axis: int) -> None:
        vertices, shape, offset = _cell(num_axis)
        xi = np.random.default_rng(1).uniform(-0.5, 1.5, num_axis)
        x = np.empty(num_axis)
        jacobian = np.empty((num_axis, num_axis))
        _cells.evaluate(vertices, 0, offset, xi, x, jacobian)
        h = 1e-6
        expected = np.empty((num_axis, num_axis))
        plus = np.empty(num_axis)
        minus = np.empty(num_axis)
        for m in range(num_axis):
            step = np.zeros(num_axis)
            step[m] = h
            _cells.position(vertices, 0, offset, xi + step, plus)
            _cells.position(vertices, 0, offset, xi - step, minus)
            expected[:, m] = (plus - minus) / (2 * h)
        np.testing.assert_allclose(jacobian, expected, rtol=0, atol=1e-9)


class TestSolve:

    @pytest.mark.parametrize("num", [1, 2, 3, 5])
    def test_random(self, num: int) -> None:
        rng = np.random.default_rng(num)
        a = rng.normal(size=(num, num))
        b = rng.normal(size=num)
        expected = np.linalg.solve(a, b)
        out = np.empty(num)
        assert _cells.solve(a.copy(), b.copy(), out)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_pivot(self) -> None:
        a = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = np.array([2.0, 3.0])
        out = np.empty(2)
        assert _cells.solve(a, b, out)
        np.testing.assert_array_equal(out, [3.0, 2.0])

    def test_singular(self) -> None:
        a = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([1.0, 1.0])
        out = np.empty(2)
        assert not _cells.solve(a, b, out)


@pytest.mark.parametrize("num_axis", [1, 2, 3])
class TestInverse:

    @pytest.mark.parametrize(
        argnames="scale,shift",
        argvalues=[
            (1, 0),
            (1e-12, 0),
            (1e9, 0),
            (1e-3, 1e3),
        ],
    )
    def test_round_trip(self, num_axis: int, scale: float, shift: float) -> None:
        vertices, shape, offset = _cell(num_axis, scale=scale, shift=shift)
        rng = np.random.default_rng(2)
        expected = rng.uniform(-0.5, 1.5, (20, num_axis))
        work = _work(num_axis)
        point = np.empty(num_axis)
        result = np.empty_like(expected)
        for i in range(expected.shape[0]):
            _cells.position(vertices, 0, offset, expected[i], point)
            point = point + vertices[:, 0]
            xi = np.full(num_axis, 0.5)
            assert _cells.inverse(vertices, 0, offset, point, xi, work)
            result[i] = xi
        # the local coordinates are only as good as the point, whose rounding
        # error grows with its distance from the origin
        atol = 1e-12 + 1e-15 * shift / scale
        np.testing.assert_allclose(result, expected, rtol=0, atol=atol)

    def test_unreachable(self, num_axis: int) -> None:
        # a cell squashed flat along its first axis cannot reach a point off
        # the plane it lies in
        vertices, shape, offset = _cell(num_axis)
        vertices[0] = 0
        point = np.full(num_axis, 0.5)
        xi = np.full(num_axis, 0.5)
        assert not _cells.inverse(vertices, 0, offset, point, xi, _work(num_axis))


class TestWeights:

    def test_partition_of_unity(self) -> None:
        rng = np.random.default_rng(3)
        shape = np.array([4, 5, 3])
        cell = np.stack([rng.integers(0, n - 1, 10) for n in shape], axis=~0)
        xi = rng.uniform(-0.5, 1.5, (10, 3))
        inside = np.ones(10, dtype=bool)
        indices_input, indices_output, values = _cells.weights(
            cell, xi, inside, shape, np.empty(0), False
        )
        np.testing.assert_array_equal(indices_output, np.repeat(np.arange(10), 8))
        sums = np.bincount(indices_output, weights=values)
        np.testing.assert_allclose(sums, 1, rtol=1e-14)
        first = indices_input[::8]
        np.testing.assert_array_equal(first, np.ravel_multi_index(cell.T, shape))

    def test_corners_of_the_cell(self) -> None:
        shape = np.array([3, 4])
        cell = np.array([[1, 2]])
        xi = np.array([[0.25, 0.5]])
        indices_input, indices_output, values = _cells.weights(
            cell, xi, np.ones(1, dtype=bool), shape, np.empty(0), False
        )
        np.testing.assert_array_equal(indices_input, [6, 10, 7, 11])
        np.testing.assert_allclose(values, [0.375, 0.125, 0.375, 0.125])

    def test_weights_input(self) -> None:
        shape = np.array([3, 4])
        cell = np.array([[1, 2]])
        xi = np.array([[0.25, 0.5]])
        weights_input = np.arange(12, dtype=float)
        _, _, values = _cells.weights(
            cell, xi, np.ones(1, dtype=bool), shape, weights_input, False
        )
        expected = np.array([0.375, 0.125, 0.375, 0.125]) * [6, 10, 7, 11]
        np.testing.assert_allclose(values, expected)

    @pytest.mark.parametrize("nan_outside", [False, True])
    def test_outside(self, nan_outside: bool) -> None:
        shape = np.array([3, 4])
        cell = np.array([[1, 2], [0, 0]])
        xi = np.array([[1.5, 0.5], [0.5, 0.5]])
        inside = np.array([False, True])
        _, _, values = _cells.weights(cell, xi, inside, shape, np.empty(0), nan_outside)
        first = [-0.25, 0.75, -0.25, 0.75]
        if nan_outside:
            first = [np.nan] * 4
        np.testing.assert_allclose(values, first + [0.25] * 4)

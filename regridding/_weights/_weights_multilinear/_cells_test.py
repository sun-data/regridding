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


def _newton(
    vertices: np.ndarray,
    shape: np.ndarray,
    point: np.ndarray,
    cell: tuple[int, ...],
    clamp: bool = False,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Run Newton's method from the middle of a cell, and return whether it
    converged, the cell it stopped in, and its local coordinates there.
    """
    num_axis = shape.size
    cell_ = np.array(cell, dtype=np.int64)
    xi = np.full(num_axis, 0.5)
    converged = _cells.newton(
        vertices,
        shape,
        _cells.strides(shape),
        _cells.offsets(shape),
        point,
        clamp,
        cell_,
        xi,
        np.empty(num_axis, dtype=np.int64),
        _work(num_axis),
    )
    return converged, cell_, xi


@pytest.mark.parametrize("num_axis", [1, 2, 3])
class TestNewton:

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
        point = np.empty(num_axis)
        result = np.empty_like(expected)
        for i in range(expected.shape[0]):
            _cells.position(vertices, 0, offset, expected[i], point)
            converged, cell, result[i] = _newton(
                vertices, shape, point + vertices[:, 0], (0,) * num_axis
            )
            assert converged
            assert np.all(cell == 0)
        # the local coordinates are only as good as the point, whose rounding
        # error grows with its distance from the origin
        atol = 1e-12 + 1e-15 * shift / scale
        np.testing.assert_allclose(result, expected, rtol=0, atol=atol)

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    def test_step_overflows(self, num_axis: int) -> None:
        # a cell far smaller than its distance to the point
        vertices, shape, offset = _cell(num_axis, scale=1e-300)
        point = np.full(num_axis, 1e10)
        converged, _, _ = _newton(vertices, shape, point, (0,) * num_axis)
        assert not converged

    def test_unreachable(self, num_axis: int) -> None:
        # a cell squashed flat along its first axis cannot reach a point off
        # the plane it lies in
        vertices, shape, offset = _cell(num_axis)
        vertices[0] = 0
        point = np.full(num_axis, 0.5)
        converged, _, _ = _newton(vertices, shape, point, (0,) * num_axis)
        assert not converged

    def test_clamp(self, num_axis: int) -> None:
        # starting in the first cell of a sheared grid, the iterate crosses
        # the grid and stops in the cell at its far corner, continued past it
        axes = [np.arange(4.0 + m) for m in range(num_axis)]
        grid = list(np.meshgrid(*axes, indexing="ij"))
        grid[0] = grid[0] + 0.3 * grid[~0]
        shape = np.array(grid[0].shape, dtype=np.int64)
        vertices = np.stack([np.ravel(c) for c in grid])
        index = shape + 0.5
        point = index.copy()
        point[0] = point[0] + 0.3 * index[~0]
        converged, cell, xi = _newton(
            vertices, shape, point, (0,) * num_axis, clamp=True
        )
        assert converged
        np.testing.assert_array_equal(cell, shape - 2)
        # 1.5 vertices past the last one, so 2.5 from the start of the last cell
        np.testing.assert_allclose(xi, 2.5, rtol=0, atol=1e-12)


@pytest.mark.parametrize("num_axis", [2, 3])
@pytest.mark.parametrize("aspect", [1e4, 1e6])
def test_newton_thin_cell(num_axis: int, aspect: float) -> None:
    # a cell much longer than it is wide, at an angle to the axes.  Rounding
    # along its length moves the local coordinates across its width by far
    # more than the tolerance on a step, which once left Newton's method
    # unconverged for points well inside the cell.
    corners = np.stack(np.meshgrid(*[np.arange(2.0)] * num_axis, indexing="ij"))
    corners[1:] = corners[1:] / aspect
    angle = 0.6
    rotation = np.eye(num_axis)
    rotation[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    corners = np.einsum("ij,j...->i...", rotation, corners)
    vertices = corners.reshape(num_axis, -1)
    shape = np.full(num_axis, 2, dtype=np.int64)
    offset = _cells.offsets(shape)
    rng = np.random.default_rng(0)
    expected = rng.uniform(0.05, 0.95, (50, num_axis))
    point = np.empty(num_axis)
    result = np.empty_like(expected)
    for i in range(expected.shape[0]):
        _cells.position(vertices, 0, offset, expected[i], point)
        converged, _, result[i] = _newton(
            vertices, shape, point + vertices[:, 0], (0,) * num_axis
        )
        assert converged
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13 * aspect)


class TestWeights:

    def test_partition_of_unity(self) -> None:
        rng = np.random.default_rng(3)
        shape = np.array([4, 5, 3])
        cell = np.stack([rng.integers(0, n - 1, 10) for n in shape], axis=~0)
        xi = rng.uniform(-0.5, 1.5, (10, 3))
        inside = np.ones(10, dtype=bool)
        indices_input, indices_output, values = _cells.weights(
            cell[np.newaxis],
            xi[np.newaxis],
            inside[np.newaxis],
            shape,
            np.empty((0, 0)),
            False,
        )
        np.testing.assert_array_equal(indices_output, [np.repeat(np.arange(10), 8)])
        sums = np.bincount(indices_output[0], weights=values[0])
        np.testing.assert_allclose(sums, 1, rtol=1e-14)
        first = indices_input[0, ::8]
        np.testing.assert_array_equal(first, np.ravel_multi_index(cell.T, shape))

    def test_corners_of_the_cell(self) -> None:
        shape = np.array([3, 4])
        cell = np.array([[[1, 2]]])
        xi = np.array([[[0.25, 0.5]]])
        indices_input, indices_output, values = _cells.weights(
            cell, xi, np.ones((1, 1), dtype=bool), shape, np.empty((0, 0)), False
        )
        np.testing.assert_array_equal(indices_input, [[6, 10, 7, 11]])
        np.testing.assert_allclose(values, [[0.375, 0.125, 0.375, 0.125]])

    @pytest.mark.parametrize("per_element", [False, True])
    def test_weights_input(self, per_element: bool) -> None:
        # two elements, which share the weights of the vertices or have
        # their own
        shape = np.array([3, 4])
        cell = np.array([[[1, 2]], [[0, 0]]])
        xi = np.array([[[0.25, 0.5]], [[0.5, 0.5]]])
        weights_input = np.arange(12, dtype=float)[np.newaxis]
        if per_element:
            weights_input = np.concatenate([weights_input, 2 * weights_input])
        _, _, values = _cells.weights(
            cell, xi, np.ones((2, 1), dtype=bool), shape, weights_input, False
        )
        expected = np.array(
            [
                np.array([0.375, 0.125, 0.375, 0.125]) * [6, 10, 7, 11],
                np.array([0.25, 0.25, 0.25, 0.25]) * [0, 4, 1, 5],
            ]
        )
        if per_element:
            expected[1] *= 2
        np.testing.assert_allclose(values, expected)

    @pytest.mark.parametrize("nan_outside", [False, True])
    def test_outside(self, nan_outside: bool) -> None:
        shape = np.array([3, 4])
        cell = np.array([[[1, 2], [0, 0]]])
        xi = np.array([[[1.5, 0.5], [0.5, 0.5]]])
        inside = np.array([[False, True]])
        _, _, values = _cells.weights(
            cell, xi, inside, shape, np.empty((0, 0)), nan_outside
        )
        first = [-0.25, 0.75, -0.25, 0.75]
        if nan_outside:
            first = [np.nan] * 4
        np.testing.assert_allclose(values, [first + [0.25] * 4])

    def test_no_points(self) -> None:
        shape = np.array([3, 4])
        result = _cells.weights(
            np.empty((2, 0, 2), dtype=np.int64),
            np.empty((2, 0, 2)),
            np.empty((2, 0), dtype=bool),
            shape,
            np.empty((0, 0)),
            False,
        )
        assert [a.shape for a in result] == [(2, 0)] * 3

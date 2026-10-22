import pytest
import numpy as np
import scipy.optimize
import regridding
from . import _cells, _curvilinear


def _flat(grid: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The vertices of a grid, flattened, and its shape."""
    vertices = np.stack([np.ravel(c) for c in grid])
    return vertices.astype(float), np.array(grid[0].shape)


def _annulus(
    radius_inner: float = 1,
    num_radius: int = 6,
    num_angle: int = 37,
) -> tuple[np.ndarray, np.ndarray]:
    """
    An annulus which wraps most of the way around, whose hole made the secant
    locator of the conservative sweep report points inside it as outside.
    """
    r, t = np.meshgrid(
        np.linspace(radius_inner, 2, num_radius),
        np.linspace(0, 0.9 * 2 * np.pi, num_angle),
        indexing="ij",
    )
    return r * np.cos(t), r * np.sin(t)


def _distorted(scale: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """A grid with the kind of smooth distortion an optical system has."""
    u, v = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 7), indexing="ij")
    return (
        scale * (u + 0.05 * u * v + 0.03 * v**2),
        scale * (v + 0.04 * u**2 - 0.02 * u * v),
    )


def _alternating() -> tuple[np.ndarray, np.ndarray]:
    """
    A rotated grid whose cells alternate in width, on which Newton's method
    across the whole grid cycles between cells.
    """
    widths = np.tile([0.4, 1.0], 6)
    a, b = np.meshgrid(
        np.concatenate([[0], np.cumsum(widths)]),
        np.linspace(0, 1, 4),
        indexing="ij",
    )
    angle = 0.3
    return (
        a * np.cos(angle) - b * np.sin(angle),
        a * np.sin(angle) + b * np.cos(angle),
    )


def _distorted_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A three-dimensional grid with a smooth distortion."""
    p, q, w = np.meshgrid(
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 6),
        np.linspace(-1, 1, 4),
        indexing="ij",
    )
    return (
        p + 0.05 * q * w,
        q + 0.04 * p**2,
        w + 0.03 * p * q + 0.02 * q**2,
    )


def _thin(aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """
    A grid of cells much longer than they are wide, rotated and slightly
    curved, in which rounding along the length of a cell once kept Newton's
    method from converging on points well inside it.
    """
    a, b = np.meshgrid(
        np.linspace(0, 1, 6),
        np.linspace(0, 1 / aspect, 6),
        indexing="ij",
    )
    angle = 0.6
    return (
        a * np.cos(angle) - b * np.sin(angle) + 0.001 * a * a,
        a * np.sin(angle) + b * np.cos(angle),
    )


def _thin_3d(aspect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A three-dimensional grid of cells much longer than they are wide."""
    a, b, c = np.meshgrid(
        np.linspace(0, 1, 4),
        np.linspace(0, 1 / aspect, 4),
        np.linspace(0, 1 / aspect, 3),
        indexing="ij",
    )
    angle = 0.6
    return (
        a * np.cos(angle) - b * np.sin(angle) + 0.001 * a * a,
        a * np.sin(angle) + b * np.cos(angle),
        c + 0.3 * a,
    )


def _folded_1d() -> tuple[np.ndarray]:
    """A one-dimensional grid which turns back on itself."""
    return (np.array([0.0, 1.0, 3.0, 2.0, 4.0]),)


grids = {
    "annulus": _annulus(),
    "disk-like annulus": _annulus(radius_inner=0.05),
    "distorted": _distorted(),
    "distorted, scale 1e-12": _distorted(scale=1e-12),
    "distorted, scale 1e9": _distorted(scale=1e9),
    "alternating": _alternating(),
    "distorted 3D": _distorted_3d(),
    "thin, aspect 1e4": _thin(1e4),
    "thin, aspect 1e6": _thin(1e6),
    "thin 3D, aspect 1e4": _thin_3d(1e4),
    "folded 1D": _folded_1d(),
    "single cell": tuple(c[:2, :2] for c in _distorted()),
}


def _points(
    grid: tuple[np.ndarray, ...],
    num: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Random points over a region a little larger than the grid."""
    rng = np.random.default_rng(seed)
    lower = np.array([c.min() for c in grid])
    upper = np.array([c.max() for c in grid])
    margin = 0.1 * (upper - lower)
    return rng.uniform(lower - margin, upper + margin, (num, len(grid))).T


def _multilinear(
    corners: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """
    The position of a point of a cell, by interpolating between its corners
    along one axis after another.

    Parameters
    ----------
    corners
        The position of each corner, of shape ``(2,) * N + (N,)``.
    xi
        The local coordinates of the point.
    """
    result = corners
    for m in range(xi.size):
        result = (1 - xi[m]) * result[0] + xi[m] * result[1]
    return result


def _interior(
    grid: tuple[np.ndarray, ...],
    num: int = 100,
    seed: int = 1,
) -> np.ndarray:
    """Random points well inside random cells of a grid."""
    rng = np.random.default_rng(seed)
    stacked = np.stack(grid, axis=~0)
    num_axis = len(grid)
    result = np.empty((num_axis, num))
    for i in range(num):
        cell = [rng.integers(0, n - 1) for n in grid[0].shape]
        corners = stacked[tuple(slice(j, j + 2) for j in cell)]
        result[:, i] = _multilinear(corners, rng.uniform(0.05, 0.95, num_axis))
    return result


def _inside_brute(
    grid: tuple[np.ndarray, ...],
    points: np.ndarray,
) -> np.ndarray:
    """
    Whether each point is inside the grid, by trying every cell, in ways
    which share nothing with the code under test.

    In two dimensions a cell is the quadrilateral between its corners, since
    its edges are straight, so each quadrilateral is tested with
    :func:`regridding.geometry.point_is_inside_polygon`.  In other
    dimensions :func:`scipy.optimize.root` finds the local coordinates of the
    point in each cell whose bounding box holds it.
    """
    num_axis = len(grid)
    shape = grid[0].shape
    stacked = np.stack(grid, axis=~0)
    result = np.zeros(points.shape[1], dtype=bool)
    for i in range(points.shape[1]):
        point = points[:, i]
        for cell in np.ndindex(*(n - 1 for n in shape)):
            corners = stacked[tuple(slice(j, j + 2) for j in cell)]
            if num_axis == 2:
                x = corners[..., 0][[0, 1, 1, 0], [0, 0, 1, 1]]
                y = corners[..., 1][[0, 1, 1, 0], [0, 0, 1, 1]]
                inside = regridding.geometry.point_is_inside_polygon(
                    x=point[0],
                    y=point[1],
                    vertices_x=x,
                    vertices_y=y,
                )
            else:
                flat = corners.reshape(-1, num_axis)
                if np.any(point < flat.min(0)) or np.any(point > flat.max(0)):
                    continue
                solution = scipy.optimize.root(
                    lambda xi: _multilinear(corners, xi) - point,
                    x0=np.full(num_axis, 0.5),
                    method="hybr",
                    options={"xtol": 1e-14},
                )
                # judged by the residual, since asked for this tolerance the
                # solver reports failure once it cannot improve on a
                # solution, however good
                xi = solution.x
                residual = np.abs(_multilinear(corners, xi) - point).max()
                inside = bool(residual <= 1e-12) and bool(
                    np.all((-1e-9 <= xi) & (xi <= 1 + 1e-9))
                )
            if inside:
                result[i] = True
                break
    return result


def _locate(
    grid: tuple[np.ndarray, ...],
    points: np.ndarray,
    extrapolate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the hierarchy of a grid and locate the points in it."""
    vertices, shape = _flat(grid)
    tree = _curvilinear.tree(vertices, shape, num_chunk=3)
    return _curvilinear.locate(vertices, shape, *tree, points, extrapolate, 3)


def _position(
    grid: tuple[np.ndarray, ...],
    cell: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """The position of each point given by its cell and local coordinates."""
    vertices, shape = _flat(grid)
    num_axis = len(grid)
    offset = _cells.offsets(shape)
    stride = _cells.strides(shape)
    result = np.empty((num_axis, cell.shape[0]))
    x = np.empty(num_axis)
    for i in range(cell.shape[0]):
        base = int(np.dot(cell[i], stride))
        _cells.position(vertices, base, offset, xi[i], x)
        result[:, i] = x + vertices[:, base]
    return result


class TestTree:

    @pytest.mark.parametrize("name", list(grids))
    def test_boxes(self, name: str) -> None:
        grid = grids[name]
        vertices, shape = _flat(grid)
        lower, upper, start, shapes = _curvilinear.tree(vertices, shape, num_chunk=2)
        np.testing.assert_array_equal(shapes[0], shape - 1)
        np.testing.assert_array_equal(shapes[~0], np.ones(len(grid)))
        num_level = shapes.shape[0] - 1
        for level in range(1, num_level + 1):
            size = 2**level
            for k, block in enumerate(np.ndindex(*shapes[level])):
                index = tuple(
                    slice(size * b, min(size * (b + 1), n - 1) + 1)
                    for b, n in zip(block, shape)
                )
                expected_lower = [c[index].min() for c in grid]
                expected_upper = [c[index].max() for c in grid]
                np.testing.assert_array_equal(lower[start[level] + k], expected_lower)
                np.testing.assert_array_equal(upper[start[level] + k], expected_upper)


@pytest.mark.parametrize("name", list(grids))
class TestLocate:

    def test_inside(self, name: str) -> None:
        grid = grids[name]
        points = np.concatenate([_points(grid), _interior(grid)], axis=~0)
        cell, xi, inside = _locate(grid, points, extrapolate=False)
        np.testing.assert_array_equal(inside, _inside_brute(grid, points))
        assert np.all(inside[-100:])

    def test_local_coordinates(self, name: str) -> None:
        grid = grids[name]
        points = _points(grid)
        cell, xi, inside = _locate(grid, points, extrapolate=False)
        assert np.all((-1e-9 <= xi[inside]) & (xi[inside] <= 1 + 1e-9))
        scale = float(np.max([np.ptp(c) for c in grid]))
        np.testing.assert_allclose(
            _position(grid, cell[inside], xi[inside]),
            points[:, inside],
            rtol=0,
            atol=1e-13 * scale,
        )
        assert np.all(np.isnan(xi[~inside]))
        assert np.all(cell[~inside] == 0)

    def test_extrapolate(self, name: str) -> None:
        # every point the cells along the edge of the grid continue to is
        # reached, so the position of each point is recovered, to the
        # rounding error of a position that many cells along the continuation
        grid = grids[name]
        points = _points(grid)
        cell, xi, inside = _locate(grid, points, extrapolate=True)
        scale = float(np.max([np.ptp(c) for c in grid]))
        reach = np.prod(np.abs(xi) + np.abs(1 - xi), axis=~0)
        error = np.abs(_position(grid, cell, xi) - points).max(axis=0)
        np.testing.assert_array_less(error, 1e-12 * scale * reach)
        assert not np.all(inside)


def test_extrapolate_clamps_parallelograms() -> None:
    # on a grid of parallelograms the cell to extrapolate from is the one
    # nearest in index space
    u, v = np.meshgrid(np.arange(5.0), np.arange(4.0), indexing="ij")
    grid = (u + 0.5 * v, v)
    points = np.array([[-3.0, 10.0, 2.0, 1.0], [1.5, 1.5, -2.0, 10.0]])
    cell, xi, inside = _locate(grid, points, extrapolate=True)
    np.testing.assert_array_equal(cell, [[0, 1], [3, 1], [3, 0], [0, 2]])
    np.testing.assert_allclose(
        xi,
        [[-3.75, 0.5], [6.25, 0.5], [0.0, -2.0], [-4.0, 8.0]],
        rtol=0,
        atol=1e-14,
    )
    assert not np.any(inside)


def test_extrapolate_unreachable() -> None:
    # the first cell has no length, so nothing to the left of the grid can
    # be extrapolated to, while the right of the grid can
    grid = (np.array([0.0, 0.0, 1.0, 2.0]),)
    points = np.array([[-1.0, 3.0]])
    cell, xi, inside = _locate(grid, points, extrapolate=True)
    assert not np.any(inside)
    assert np.isnan(xi[0, 0])
    np.testing.assert_array_equal(cell[1], [2])
    np.testing.assert_allclose(xi[1], [2.0], rtol=1e-14)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_extrapolate_step_overflows() -> None:
    # cells far smaller than the distance to the point
    grid = (np.array([0.0, 1e-300, 3e-300]),)
    cell, xi, inside = _locate(grid, np.array([[1e10]]), extrapolate=True)
    assert not inside[0]
    assert np.isnan(xi[0, 0])


def test_degenerate_cell() -> None:
    # the first row of cells has no height, so Newton's method cannot find
    # a point on it in them, but the point is on the edge of the cell above
    x, y = np.meshgrid(
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0, 1.0, 2.0]),
        indexing="ij",
    )
    points = np.array([[0.1], [1.0]])
    cell, xi, inside = _locate((x, y), points, extrapolate=False)
    assert inside[0]
    np.testing.assert_array_equal(cell[0], [0, 1])
    np.testing.assert_allclose(xi[0], [0.2, 0.0], rtol=0, atol=1e-15)


def test_not_finite() -> None:
    grid = _distorted()
    points = np.array([[0.1, np.nan, np.inf], [np.nan, 0.1, 0.2]])
    cell, xi, inside = _locate(grid, points, extrapolate=True)
    assert not np.any(inside)
    assert np.all(np.isnan(xi))
    assert np.all(cell == 0)

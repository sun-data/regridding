from typing import Any
import pytest
import numpy as np
import scipy.interpolate
import astropy.units as u
import regridding

x = np.linspace(-1, 1, num=10)
y = np.linspace(-1, 1, num=11)
x_broadcasted, y_broadcasted = np.meshgrid(
    x,
    y,
    indexing="ij",
)

new_y = np.linspace(-1, 1, num=5)
new_x = np.linspace(-1, 1, num=6)

new_x_broadcasted, new_y_broadcasted = np.meshgrid(
    x,
    new_y,
    indexing="ij",
)

new_x_broadcasted_2, new_y_broadcasted_2 = np.meshgrid(
    new_x,
    y,
    indexing="ij",
)


@pytest.mark.parametrize(
    argnames="coordinates_input,"
    "coordinates_output,"
    "values_input,"
    "values_output,"
    "axis_input,"
    "axis_output,"
    "weights_input,"
    "result_expected",
    argvalues=[
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11) * u.mm,),
            (np.linspace(-1, 1, num=11) * u.mm,),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            np.empty(shape=11),
            None,
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (y,),
            (new_y,),
            x_broadcasted + y_broadcasted,
            None,
            (~0,),
            (~0,),
            None,
            new_x_broadcasted + new_y_broadcasted,
        ),
        (
            (x[..., np.newaxis],),
            (new_x[..., np.newaxis],),
            x_broadcasted + y_broadcasted,
            None,
            (0,),
            (0,),
            None,
            new_x_broadcasted_2 + new_y_broadcasted_2,
        ),
        (
            (x[..., np.newaxis],),
            (0.1 * new_x[..., np.newaxis] + 0.001 * new_y,),
            x[..., np.newaxis],
            None,
            (0,),
            (0,),
            None,
            0.1 * new_x[..., np.newaxis] + 0.001 * new_y,
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            1,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            0,
            0,
        ),
    ],
)
def test_weights_multilinear_1d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: None | np.ndarray,
    result_expected: np.ndarray,
) -> None:
    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        method="multilinear",
    )
    result = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, float)
    assert np.allclose(result, result_expected)


def _regrid(
    grid: tuple[np.ndarray, ...],
    values: np.ndarray,
    points: tuple[np.ndarray, ...],
    **kwargs: Any,
) -> np.ndarray:
    """Interpolate an array sampled on a grid at a set of points."""
    return regridding.regrid(
        coordinates_input=grid,
        coordinates_output=points,
        values_input=values,
        method="multilinear",
        **kwargs,
    )


def _affine(coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
    """
    An affine function of the coordinates, which multilinear interpolation
    reproduces exactly on any grid, since the weights of the corners of a
    cell sum to one and average the corners to the point.
    """
    result = np.full(np.broadcast(*coordinates).shape, 0.3)
    for k, c in enumerate(coordinates):
        result = result + (0.7 - 0.4 * k) * c
    return result


def _annulus() -> tuple[np.ndarray, np.ndarray]:
    """An annulus which wraps most of the way around."""
    r, t = np.meshgrid(
        np.linspace(1, 2, 5),
        np.linspace(0, 0.9 * 2 * np.pi, 31),
        indexing="ij",
    )
    return r * np.cos(t), r * np.sin(t)


def _distorted() -> tuple[np.ndarray, np.ndarray]:
    """A grid with the kind of smooth distortion an optical system has."""
    u, v = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 6), indexing="ij")
    return u + 0.05 * u * v + 0.03 * v**2, v + 0.04 * u**2 - 0.02 * u * v


def _distorted_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A three-dimensional grid with a smooth distortion."""
    p, q, w = np.meshgrid(
        np.linspace(-1, 1, 4),
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 3),
        indexing="ij",
    )
    return p + 0.05 * q * w, q + 0.04 * p**2, w + 0.03 * p * q


grids_curvilinear = {
    "annulus": _annulus(),
    "distorted": _distorted(),
    "distorted 3D": _distorted_3d(),
}


def _points(
    grid: tuple[np.ndarray, ...],
    num: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Random points over a region a little larger than the grid."""
    rng = np.random.default_rng(seed)
    lower = np.array([c.min() for c in grid])
    upper = np.array([c.max() for c in grid])
    margin = 0.1 * (upper - lower)
    return tuple(rng.uniform(lower - margin, upper + margin, (num, len(grid))).T)


class TestRectilinear:

    @pytest.mark.parametrize("num_axis", [1, 2, 3])
    @pytest.mark.parametrize("bounds", ["extrapolate", "nan"])
    def test_matches_scipy(self, num_axis: int, bounds: str) -> None:
        rng = np.random.default_rng(num_axis)
        axes = [np.sort(rng.uniform(-1, 1, 3 + m)) for m in range(num_axis)]
        # one of the axes runs backwards
        axes[0] = axes[0][::-1]
        grid = tuple(np.meshgrid(*axes, indexing="ij"))
        values = rng.normal(size=grid[0].shape)
        points = rng.uniform(-1.3, 1.3, (num_axis, 30))
        result = _regrid(grid, values, tuple(points), bounds=bounds)
        expected = scipy.interpolate.RegularGridInterpolator(
            points=axes,
            values=values,
            bounds_error=False,
            fill_value=None if bounds == "extrapolate" else np.nan,
        )(points.T)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13)
        assert np.any(np.isnan(result)) == (bounds == "nan")

    def test_descending_1d(self) -> None:
        # a grid which runs backwards used to be searched as if it ran forwards
        x = np.array([3.0, 2.0, 0.5, 0.0])
        result = _regrid((x,), x**2, (np.array([0.25, 1.0, 2.5, 4.0]),))
        np.testing.assert_allclose(result, [0.125, 1.5, 6.5, 14.0])


@pytest.mark.parametrize("name", list(grids_curvilinear))
class TestCurvilinear:

    def test_affine(self, name: str) -> None:
        grid = grids_curvilinear[name]
        points = _points(grid)
        result = _regrid(grid, _affine(grid), points)
        np.testing.assert_allclose(result, _affine(points), rtol=0, atol=1e-12)

    def test_nan(self, name: str) -> None:
        grid = grids_curvilinear[name]
        points = _points(grid)
        result = _regrid(grid, _affine(grid), points, bounds="nan")
        inside = np.isfinite(result)
        assert 0 < inside.sum() < inside.size
        np.testing.assert_allclose(
            result[inside],
            _affine(points)[inside],
            rtol=0,
            atol=1e-12,
        )

    def test_raise(self, name: str) -> None:
        grid = grids_curvilinear[name]
        with pytest.raises(ValueError, match="fall outside the input grid"):
            _regrid(grid, _affine(grid), _points(grid), bounds="raise")

    def test_raise_inside_grid(self, name: str) -> None:
        grid = grids_curvilinear[name]
        points = tuple(c[1:-1, 1:-1] for c in grid)
        result = _regrid(grid, _affine(grid), points, bounds="raise")
        np.testing.assert_allclose(result, _affine(points), rtol=0, atol=1e-12)


class TestPoints:
    """
    Multilinear interpolation weights points, rather than cells, so the
    points may be arranged in any number of axes.
    """

    def test_grid_2d_points_1d(self) -> None:
        grid = _distorted()
        points = _points(grid)
        result = _regrid(grid, _affine(grid), points, axis_output=(0,))
        assert result.shape == points[0].shape
        np.testing.assert_allclose(result, _affine(points), rtol=0, atol=1e-12)

    def test_grid_3d_points_2d(self) -> None:
        grid = _distorted_3d()
        points = tuple(c.reshape(10, 10) for c in _points(grid))
        result = _regrid(grid, _affine(grid), points)
        assert result.shape == (10, 10)
        np.testing.assert_allclose(result, _affine(points), rtol=0, atol=1e-12)

    def test_single_point(self) -> None:
        grid = _distorted()
        point = (np.array(0.1), np.array(-0.2))
        result = _regrid(grid, _affine(grid), point)
        assert result.shape == ()
        np.testing.assert_allclose(result, _affine(point), rtol=0, atol=1e-12)

    def test_conservative_axes_must_match(self) -> None:
        grid = _distorted()
        with pytest.raises(ValueError, match="number of axes in `axis_output`"):
            regridding.weights(
                coordinates_input=grid,
                coordinates_output=_points(grid),
                axis_output=(0,),
                method="conservative",
            )


class TestOrthogonal:

    def test_grid_per_element(self) -> None:
        # two grids, rotated by different angles, each with its own points
        angle = np.array([0.1, 0.7])[:, np.newaxis, np.newaxis]
        u, v = _distorted()
        grid = (
            u * np.cos(angle) - v * np.sin(angle),
            u * np.sin(angle) + v * np.cos(angle),
        )
        values = np.stack([_affine(grid)[0], 2 * _affine(grid)[1]])
        points = tuple(np.stack([c, c[::-1]]) for c in _points(_distorted()))
        result = _regrid(grid, values, points, axis_input=(1, 2), axis_output=(1,))
        expected = np.stack(
            [
                _regrid(
                    tuple(c[i] for c in grid),
                    values[i],
                    tuple(c[i] for c in points),
                )
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result[0], _affine(points)[0], atol=1e-12)

    def test_grid_shared(self) -> None:
        # one grid, broadcast over a batch of points, which gives the same
        # result as taking every point at once
        grid = tuple(c[np.newaxis] for c in _annulus())
        points = tuple(c.reshape(4, 25) for c in _points(_annulus()))
        values = _affine(_annulus())
        result = _regrid(grid, values, points, axis_input=(1, 2), axis_output=(1,))
        expected = _regrid(_annulus(), values, points)
        assert result.shape == expected.shape == (4, 25)
        np.testing.assert_array_equal(result, expected)


class TestWeightsInput:

    @pytest.mark.parametrize("name", list(grids_curvilinear))
    def test_weights_input(self, name: str) -> None:
        grid = grids_curvilinear[name]
        points = _points(grid)
        values = _affine(grid)
        weights_input = np.random.default_rng(1).uniform(0.5, 1.5, values.shape)
        weights = regridding.weights(
            coordinates_input=grid,
            coordinates_output=points,
            weights_input=weights_input,
            method="multilinear",
        )
        result = regridding.regrid_from_weights(*weights, values_input=values)
        expected = _regrid(grid, values * weights_input, points)
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


def test_units() -> None:
    grid = tuple(c * u.mm for c in _distorted())
    points = tuple(c * u.m for c in _points(_distorted()))
    result = _regrid(grid, _affine(_distorted()), points, bounds="nan")
    expected = _regrid(
        _distorted(),
        _affine(_distorted()),
        tuple(1000 * c.value for c in points),
        bounds="nan",
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    argnames="grid",
    argvalues=[
        _distorted(),
        tuple(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="ij")),
    ],
)
def test_points_not_finite(grid: tuple[np.ndarray, ...]) -> None:
    points = (np.array([0.1, np.nan, np.inf]), np.array([np.nan, 0.2, -0.3]))
    result = _regrid(grid, _affine(grid), points)
    assert np.all(np.isnan(result))
    with pytest.raises(ValueError, match="fall outside the input grid"):
        _regrid(grid, _affine(grid), points, bounds="raise")


def test_too_few_vertices() -> None:
    x = np.linspace(0, 1, 4)
    grid = np.meshgrid(x, x[:1], indexing="ij")
    with pytest.raises(ValueError, match="at least two vertices"):
        _regrid(tuple(grid), np.zeros((4, 1)), (np.array([0.5]), np.array([0.0])))

import pytest
import numpy as np
import astropy.units as u
import regridding

box_x = np.linspace(-1, 1, num=2)[..., np.newaxis]
box_y = np.linspace(-1, 1, num=2)

x = np.linspace(-1, 1, num=6)[..., np.newaxis]
y = np.linspace(-1, 1, num=6)

x2 = np.linspace(-1, 1, num=11)[..., np.newaxis]
y2 = np.linspace(-1, 1, num=11)


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
            (
                box_x,
                box_y,
            ),
            (
                2 * box_x,
                2 * box_y,
            ),
            np.array([[1]]),
            None,
            None,
            None,
            None,
            np.array([[1]]),
        ),
        (
            (
                -box_x,
                -box_y,
            ),
            (
                2 * box_x,
                2 * box_y,
            ),
            np.array([[1]]),
            None,
            None,
            None,
            None,
            np.array([[1]]),
        ),
        (
            (
                2 * box_x,
                2 * box_y,
            ),
            (
                box_x,
                box_y,
            ),
            np.array([[1]]),
            None,
            None,
            None,
            None,
            np.array([[0.25]]),
        ),
        (
            (
                box_x,
                2 * box_y,
            ),
            (
                2 * box_x,
                box_y,
            ),
            np.array([[1]]),
            None,
            None,
            None,
            None,
            np.array([[0.5]]),
        ),
        (
            (
                2 * box_x,
                box_y,
            ),
            (
                box_x,
                2 * box_y,
            ),
            np.array([[1]]),
            None,
            None,
            None,
            None,
            np.array([[0.5]]),
        ),
        (
            (
                np.linspace(-1, 1, num=3)[..., np.newaxis],
                np.linspace(-1, 1, num=3),
            ),
            (
                2 * box_x,
                2 * box_y,
            ),
            np.ones((2, 2)),
            None,
            None,
            None,
            None,
            np.array([[4]]),
        ),
        (
            (
                box_x,
                box_y,
            ),
            (
                np.linspace(-2, 2, num=3)[..., np.newaxis],
                np.linspace(-2, 2, num=3),
            ),
            np.ones((1, 1)),
            None,
            None,
            None,
            None,
            np.ones((2, 2)) / 4,
        ),
        (
            (
                x,
                y,
            ),
            (
                x,
                y,
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            None,
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
        ),
        (
            (
                x * np.cos(90 * u.deg) - y * np.sin(90 * u.deg),
                x * np.sin(90 * u.deg) + y * np.cos(90 * u.deg),
            ),
            (
                x,
                y,
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            None,
            np.rot90(np.random.RandomState(42).uniform(0, 10, size=(5, 5))),
        ),
        (
            (
                x * np.cos(180 * u.deg) - y * np.sin(180 * u.deg),
                x * np.sin(180 * u.deg) + y * np.cos(180 * u.deg),
            ),
            (
                x,
                y,
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            None,
            np.rot90(np.random.RandomState(42).uniform(0, 10, size=(5, 5)), k=2),
        ),
        (
            (
                x * np.cos(270 * u.deg) - y * np.sin(270 * u.deg),
                x * np.sin(270 * u.deg) + y * np.cos(270 * u.deg),
            ),
            (
                x,
                y,
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            None,
            np.rot90(np.random.RandomState(42).uniform(0, 10, size=(5, 5)), k=3),
        ),
        (
            (
                x,
                y,
            ),
            (
                x * np.cos(90 * u.deg) - y * np.sin(90 * u.deg),
                x * np.sin(90 * u.deg) + y * np.cos(90 * u.deg),
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            None,
            np.rot90(np.random.RandomState(42).uniform(0, 10, size=(5, 5)), k=-1),
        ),
        (
            (
                x,
                y,
            ),
            (
                x,
                y,
            ),
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
            None,
            None,
            None,
            1,
            np.random.RandomState(42).uniform(0, 10, size=(5, 5)),
        ),
        (
            (
                x,
                y,
            ),
            (
                x2,
                y2,
            ),
            10,
            None,
            None,
            None,
            1,
            10 / 2 / 2,
        ),
        (
            (
                x2,
                y2,
            ),
            (
                x,
                y,
            ),
            10,
            None,
            None,
            None,
            1,
            10 * 2 * 2,
        ),
    ],
)
def test_weights_conservative_2d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: None | np.ndarray,
    result_expected: np.ndarray,
) -> None:

    x, y = coordinates_output
    x = x + 1e-6
    y = y + 1e-6
    coordinates_output = (x, y)

    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        method="conservative",
        perturb=False,
    )
    result = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert np.allclose(result, result_expected, rtol=1e-3)


def _dense(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    The conservative weights between two grids as a dense matrix, output
    cell by input cell, checking that every index addresses a cell.
    """
    weights, shape_input, shape_output = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        method="conservative",
    )
    indices_input, indices_output, values = weights[()]
    num_input = int(np.prod(shape_input))
    num_output = int(np.prod(shape_output))
    assert np.all((0 <= indices_input) & (indices_input < num_input))
    assert np.all((0 <= indices_output) & (indices_output < num_output))
    result = np.zeros((num_output, num_input))
    np.add.at(result, (indices_output, indices_input), values)
    return result


def _area(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """The area of each cell of a grid, by the shoelace formula."""
    x0, x1, x2, x3 = x[:-1, :-1], x[1:, :-1], x[1:, 1:], x[:-1, 1:]
    y0, y1, y2, y3 = y[:-1, :-1], y[1:, :-1], y[1:, 1:], y[:-1, 1:]
    return 0.5 * np.abs(
        (x0 * y1 - x1 * y0)
        + (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y0 - x0 * y3)
    )


def _lattice(
    lower: float,
    upper: float,
    num: int,
) -> np.ndarray:
    """Points from `lower` to `upper`, not quite evenly spaced, so that the sweep is used."""
    result = np.linspace(lower, upper, num)
    return result + 0.002 * (upper - lower) * np.sin(np.arange(num))


def test_scale_invariant() -> None:
    # the weights are fractions of areas, so they do not depend on the scale
    # of the grids.  The locator which starts each row of the sweep used to
    # compare positions to an absolute tolerance, which put every row of
    # grids this small outside the input grid, and left indices which
    # addressed no cell.
    u = np.linspace(-1, 1, 9)
    grid_input = tuple(np.meshgrid(u + 0.013, u - 0.007, indexing="ij"))
    v = _lattice(-1.2, 1.2, 11)
    grid_output = tuple(np.meshgrid(v, v, indexing="ij"))
    expected = _dense(grid_input, grid_output)
    scale = 1e-12
    result = _dense(
        tuple(scale * c for c in grid_input),
        tuple(scale * c for c in grid_output),
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_annulus() -> None:
    # every output cell is inside the annulus, so the area it takes from the
    # input cells adds up to its own.  The locator which starts each row of
    # the sweep used to follow a cell of the annulus across its hole, and
    # left indices which addressed no cell.
    r, t = np.meshgrid(
        np.linspace(1, 2, 6),
        np.linspace(0, 0.9 * 2 * np.pi, 37),
        indexing="ij",
    )
    grid_input = (r * np.cos(t), r * np.sin(t))
    grid_output = tuple(
        np.meshgrid(_lattice(1.2, 1.8, 7), np.linspace(0.1, 0.5, 5), indexing="ij")
    )
    weights = _dense(grid_input, grid_output)
    np.testing.assert_allclose(
        weights @ _area(*grid_input).ravel(),
        _area(*grid_output).ravel(),
        rtol=1e-6,
    )

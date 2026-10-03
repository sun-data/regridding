from typing import Any
import pytest
import numpy as np
import astropy.units as u
import regridding
from regridding._weights._weights_transposed import _weights_transposed

x = np.linspace(-1, 1, num=10)
y = np.linspace(-1, 1, num=11)
x_broadcasted, y_broadcasted = np.meshgrid(
    x,
    y,
    indexing="ij",
)

x1 = np.linspace(-1, 1, num=10)[..., np.newaxis]
y1 = np.linspace(-1, 1, num=11)

x2 = np.linspace(-1, 1, num=5)[..., np.newaxis]
y2 = np.linspace(-1, 1, num=6)


@pytest.mark.parametrize(
    argnames="coordinates_input, values_input, axis_input, coordinates_output, values_output, axis_output, method",
    argvalues=[
        (
            (
                x_broadcasted[..., np.newaxis] + np.array([0, 0.001]),
                y_broadcasted[..., np.newaxis] + np.array([0, 0.001]),
            ),
            np.random.normal(size=(x.shape[0] - 1, y.shape[0] - 1, 2)),
            (0, 1),
            (
                1.1 * (x_broadcasted[..., np.newaxis] + np.array([0, 0.001])) + 0.01,
                1.2 * (y_broadcasted[..., np.newaxis] + np.array([0, 0.01])) + 0.001,
            ),
            None,
            (0, 1),
            "conservative",
        ),
    ],
)
def test_transpose_weights(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    method: None | str,
) -> None:
    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )

    data = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    transposed_weights = regridding.transpose_weights(weights)

    reversed_data = regridding.regrid_from_weights(
        *transposed_weights,
        values_input=data,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert reversed_data.shape == values_input.shape


@pytest.mark.parametrize(
    argnames="weights,"
    "coordinates_input,"
    "coordinates_output,"
    "axis_input,"
    "axis_output,"
    "weights_input,"
    "result_expected,",
    argvalues=[
        (
            regridding.weights(
                coordinates_input=y1,
                coordinates_output=y2,
                method="conservative",
            ),
            y1,
            y2,
            None,
            None,
            None,
            regridding.weights(
                coordinates_input=y2,
                coordinates_output=y1,
                method="conservative",
            ),
        ),
        (
            regridding.weights(
                coordinates_input=y1,
                coordinates_output=y2,
                weights_input=2,
                method="conservative",
            ),
            y1,
            y2,
            None,
            None,
            2,
            regridding.weights(
                coordinates_input=y2,
                coordinates_output=y1,
                weights_input=1 / 2,
                method="conservative",
            ),
        ),
        (
            regridding.weights(
                coordinates_input=x1,
                coordinates_output=x2,
                axis_input=0,
                axis_output=0,
                method="conservative",
            ),
            x1,
            x2,
            0,
            0,
            None,
            regridding.weights(
                coordinates_input=x2,
                coordinates_output=x1,
                axis_input=0,
                axis_output=0,
                method="conservative",
            ),
        ),
        (
            regridding.weights(
                coordinates_input=(x1, y1),
                coordinates_output=(x2, y2),
                method="conservative",
            ),
            (x1, y1),
            (x2, y2),
            None,
            None,
            None,
            regridding.weights(
                coordinates_input=(x2, y2),
                coordinates_output=(x1, y1),
                method="conservative",
            ),
        ),
    ],
)
def test_transpose_weights_conservative(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    coordinates_input: np.ndarray | tuple[np.ndarray, ...],
    coordinates_output: np.ndarray | tuple[np.ndarray, ...],
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: None | np.ndarray,
    result_expected: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> None:
    weights_transposed = regridding.transpose_weights_conservative(
        weights=weights,
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
    )

    values = regridding.regrid_from_weights(
        *weights_transposed,
        values_input=np.array(10),
        axis_input=axis_input,
        axis_output=axis_output,
    )

    values_expected = regridding.regrid_from_weights(
        *result_expected,
        values_input=np.array(10),
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert np.allclose(values, values_expected)


def test_transpose_weights_conservative_inverts_weights_input() -> None:
    """
    A conservative transpose given ``weights_input`` must *invert* that input
    weighting, not merely remove it: transposing a forward-weighted array must
    equal transposing the geometry alone applied to the pre-weighted values and
    then dividing by the weights. This keeps a weighted round trip
    ``Wᵀ(W(r))`` free of any residual ``weights_input`` factor.
    """
    rng = np.random.default_rng(0)

    grid_input = np.linspace(-1, 1, num=21)
    grid_output = grid_input + 0.03

    weights_input = rng.uniform(0.5, 3, size=grid_input.size - 1)
    values = rng.uniform(1, 2, size=grid_input.size - 1)

    weights = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        weights_input=weights_input,
        method="conservative",
    )
    weights_transposed = regridding.transpose_weights_conservative(
        weights=weights,
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        weights_input=weights_input,
    )
    result = regridding.regrid_from_weights(
        *weights_transposed,
        values_input=regridding.regrid_from_weights(*weights, values_input=values),
    )

    weights_geometry = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        method="conservative",
    )
    weights_geometry_transposed = regridding.transpose_weights_conservative(
        weights=weights_geometry,
        coordinates_input=grid_input,
        coordinates_output=grid_output,
    )
    result_expected = (
        regridding.regrid_from_weights(
            *weights_geometry_transposed,
            values_input=regridding.regrid_from_weights(
                *weights_geometry,
                values_input=weights_input * values,
            ),
        )
        / weights_input
    )

    assert np.allclose(result, result_expected)


def test_transpose_weights_conservative_inverts_weights_input_2d() -> None:
    """
    Regression test that a spatially-varying ``weights_input`` is inverted on a
    2D grid.

    The forward multiplies each weight by ``weights_input`` at its input cell,
    so the transpose must divide by that *same* cell's value. The resampled axes
    are flattened to build the cell indices, so a 2D ``weights_input`` that
    varies differently along each axis exposes any axis-order mismatch in that
    flattening (a symmetric or constant weight would not). ``perturb=False``
    keeps the two grids identical so the comparison isolates the
    ``weights_input`` handling.
    """
    rng = np.random.default_rng(0)

    grid_x = np.linspace(-1, 1, num=8)
    grid_y = np.linspace(-1, 1, num=12)
    x_input, y_input = np.meshgrid(grid_x, grid_y, indexing="ij")
    x_output, y_output = x_input + 0.03, y_input + 0.02

    # a weight that varies differently along each axis, so a transposed layout
    # would divide by the wrong cell's value.
    index_x = np.arange(grid_x.size - 1)
    index_y = np.arange(grid_y.size - 1)
    weights_input = 1.0 + 0.2 * index_x[:, np.newaxis] + 0.05 * index_y[np.newaxis, :]

    values = rng.uniform(1, 2, size=(grid_x.size - 1, grid_y.size - 1))

    weights = regridding.weights(
        coordinates_input=(x_input, y_input),
        coordinates_output=(x_output, y_output),
        weights_input=weights_input,
        method="conservative",
        perturb=False,
    )
    weights_transposed = regridding.transpose_weights_conservative(
        weights=weights,
        coordinates_input=(x_input, y_input),
        coordinates_output=(x_output, y_output),
        weights_input=weights_input,
    )
    result = regridding.regrid_from_weights(
        *weights_transposed,
        values_input=regridding.regrid_from_weights(*weights, values_input=values),
    )

    weights_geometry = regridding.weights(
        coordinates_input=(x_input, y_input),
        coordinates_output=(x_output, y_output),
        method="conservative",
        perturb=False,
    )
    weights_geometry_transposed = regridding.transpose_weights_conservative(
        weights=weights_geometry,
        coordinates_input=(x_input, y_input),
        coordinates_output=(x_output, y_output),
    )
    result_expected = (
        regridding.regrid_from_weights(
            *weights_geometry_transposed,
            values_input=regridding.regrid_from_weights(
                *weights_geometry,
                values_input=weights_input * values,
            ),
        )
        / weights_input
    )

    assert np.allclose(result, result_expected)


def _grids_rotated(
    num_grid: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    A stack of input grids, rotated and scaled differently so that their
    cells differ in area, and one output grid which they all share.

    Parameters
    ----------
    num_grid
        The number of input grids.
    """
    t = np.linspace(-1, 1, num=9)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.linspace(0.1, 0.7, num=num_grid)[:, np.newaxis, np.newaxis]
    scale = np.linspace(1, 0.8, num=num_grid)[:, np.newaxis, np.newaxis]
    grid_input = (
        scale * (x * np.cos(angle) - y * np.sin(angle)),
        scale * (x * np.sin(angle) + y * np.cos(angle)),
    )
    grid_output = (1.6 * x[np.newaxis], 1.6 * y[np.newaxis])
    return grid_input, grid_output


@pytest.mark.parametrize(
    argnames="weights_input",
    argvalues=[
        np.full((8, 8), 50) * u.percent,
        np.full((8, 8), 2) * u.cm**2,
    ],
)
def test_transpose_weights_conservative_units(weights_input: u.Quantity) -> None:
    """
    A dimensionless `weights_input` with a scale, such as a percentage, is
    the number it stands for, as it is when the weights are built, and a
    unit with dimensions is dropped, as it always has been.
    """
    grid_input, grid_output = _grids_rotated(2)
    kwargs: dict[str, Any] = dict(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=(1, 2),
        axis_output=(1, 2),
    )

    try:
        number = np.asarray(weights_input.to_value(u.dimensionless_unscaled))
    except u.UnitConversionError:
        number = np.asarray(weights_input.value)

    actual = regridding.transpose_weights_conservative(
        regridding.weights(
            **kwargs, weights_input=weights_input, method="conservative"
        ),
        weights_input=weights_input,
        **kwargs,
    )
    expected = regridding.transpose_weights_conservative(
        regridding.weights(**kwargs, weights_input=number, method="conservative"),
        weights_input=number,
        **kwargs,
    )

    for a, e in zip(actual[0], expected[0]):
        assert np.array_equal(a[0], e[0])
        assert np.array_equal(a[1], e[1])
        assert np.allclose(a[2], e[2], rtol=1e-14, atol=0)


def test_transpose_weights_conservative_outside() -> None:
    """
    Grids which are not the ones the weights were built on raise, rather
    than an `IndexError` from deep inside.
    """
    grid_input, grid_output = _grids_rotated(2)
    weights = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=(1, 2),
        axis_output=(1, 2),
        method="conservative",
    )

    with pytest.raises(ValueError, match="outside the grids"):
        regridding.transpose_weights_conservative(
            weights,
            coordinates_input=tuple(c[..., :5, :5] for c in grid_input),
            coordinates_output=grid_output,
            axis_input=(1, 2),
            axis_output=(1, 2),
        )


@pytest.mark.parametrize(
    argnames="num_grid",
    argvalues=[2, _weights_transposed._num_grids_across],
)
def test_cell_volume(num_grid: int) -> None:
    """
    The areas of a stack of grids are the same whether they are computed in
    parallel within each grid or across the grids, and a transpose with
    either many or few grids matches each grid transposed on its own.
    """
    grid_input, grid_output = _grids_rotated(num_grid)

    x, y = grid_input
    stack = np.ascontiguousarray(x), np.ascontiguousarray(y)
    assert np.array_equal(
        _weights_transposed._cell_volume_2d(stack),
        _weights_transposed._cell_volume_2d_across(stack),
    )

    kwargs: dict[str, Any] = dict(axis_input=(1, 2), axis_output=(1, 2))
    weights = regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        method="conservative",
        **kwargs,
    )
    actual = regridding.transpose_weights_conservative(
        weights,
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        **kwargs,
    )

    for t in range(num_grid):
        grid_input_t = tuple(c[t] for c in grid_input)
        grid_output_t = tuple(c[0] for c in grid_output)
        expected = regridding.transpose_weights_conservative(
            regridding.weights(grid_input_t, grid_output_t, method="conservative"),
            coordinates_input=grid_input_t,
            coordinates_output=grid_output_t,
        )
        assert np.array_equal(actual[0][t][0], expected[0][()][0])
        assert np.array_equal(actual[0][t][1], expected[0][()][1])
        assert np.allclose(actual[0][t][2], expected[0][()][2], rtol=1e-14, atol=0)

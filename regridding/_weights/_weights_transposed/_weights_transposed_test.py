import pytest
import numpy as np
import regridding

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
):
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
):
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


def test_transpose_weights_conservative_inverts_weights_input():
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

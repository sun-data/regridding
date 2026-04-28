import pytest
import numpy as np
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
    "values_input,"
    "axis_input,"
    "coordinates_output,"
    "values_output,"
    "axis_output,"
    "weights_input",
    argvalues=[
        (
            (x_broadcasted, y_broadcasted),
            np.random.normal(size=(10 - 1, 11 - 1)) * u.ph,
            None,
            (1.1 * x_broadcasted + 0.01, 1.2 * y_broadcasted + 0.01),
            None,
            None,
            None,
        ),
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
            None,
        ),
        (
            (x_broadcasted, y_broadcasted),
            np.random.normal(size=(10 - 1, 11 - 1)) * u.ph,
            None,
            (1.1 * x_broadcasted + 0.01, 1.2 * y_broadcasted + 0.01),
            None,
            None,
            1,
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
):
    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        method="conservative",
    )
    result = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    result_shape = np.array(np.broadcast(*coordinates_output).shape)

    if axis_input is None:
        result_shape = result_shape - 1
    else:
        for ax in axis_input:
            result_shape[ax] = result_shape[ax] - 1

    assert np.issubdtype(result.dtype, float)
    assert result.shape == tuple(result_shape)
    assert np.isclose(result.sum(), values_input.sum())

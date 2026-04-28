from typing import Sequence
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
    "method,",
    argvalues=[
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6),
            np.ones(10),
            2 * np.ones(5),
            None,
            None,
            "conservative",
        ),
    ],
)
def test_regrid(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | Sequence[int],
    axis_output: None | int | Sequence[int],
    method: str,
):
    result = regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )

    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
    )
    result_expected = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert np.allclose(result, result_expected)


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

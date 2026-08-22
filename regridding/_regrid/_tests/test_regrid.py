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


class TestCoalesce:
    """
    :func:`regridding.regrid` applies its weights once, so it defaults to
    leaving repeated ``(input, output)`` pairs unmerged.  That must not
    change the answer.
    """

    def test_same_result(self):
        method = "conservative"
        num_input = 21
        num_output = 17

        t = np.linspace(-0.9, 0.9, num_input)
        u = t[:, np.newaxis] * np.ones(num_input)
        v = np.ones(num_input)[:, np.newaxis] * t
        x_input = u * np.cos(0.3) - v * np.sin(0.3) + 0.15 * u * v
        y_input = u * np.sin(0.3) + v * np.cos(0.3) + 0.10 * u * u

        x_output = np.linspace(-1, 1, num_output)[:, np.newaxis] * np.ones(num_output)
        y_output = np.ones(num_output)[:, np.newaxis] * np.linspace(-1, 1, num_output)

        if method == "conservative":
            shape = (num_input - 1, num_input - 1)
        else:
            shape = (num_input, num_input)
        values_input = np.random.default_rng(42).random(shape)

        kwargs = dict(
            coordinates_input=(x_input, y_input),
            coordinates_output=(x_output, y_output),
            values_input=values_input,
            method=method,
        )

        result = regridding.regrid(**kwargs, coalesce=False)
        expected = regridding.regrid(**kwargs, coalesce=True)

        assert np.allclose(result, expected, equal_nan=True)

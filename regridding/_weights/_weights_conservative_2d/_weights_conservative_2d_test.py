import pytest
import numpy as np
import astropy.units as u
import regridding

box_x = np.linspace(-1, 1, num=2)[..., np.newaxis]
box_y = np.linspace(-1, 1, num=2)

x = np.linspace(-1, 1, num=6)[..., np.newaxis]
y = np.linspace(-1, 1, num=6)


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
                x + 1e-6,
                y + 1e-6,
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
                x + 1e-6,
                y + 1e-6,
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
                x + 1e-6,
                y + 1e-6,
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
                x + 1e-6,
                y + 1e-6,
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
                x + 1e-6,
                y + 1e-6,
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
    ],
)
def test_weights_conservative_2d(
    capsys,
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: None | np.ndarray,
    result_expected: np.ndarray,
):
    with capsys.disabled():
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

        assert np.allclose(result, result_expected, rtol=1e-3)

        assert result.shape == result_expected.shape

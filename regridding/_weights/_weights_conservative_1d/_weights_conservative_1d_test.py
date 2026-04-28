import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="x_input,"
             "x_output,"
             "values_input,"
             "values_expected,"
             "axis_input,"
             "axis_output,"
             "weights_input,",
    argvalues=[
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6),
            np.ones(10),
            2 * np.ones(5),
            None,
            None,
            None,
        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6) + 1e-6,
            np.ones(10),
            2 * np.ones(5),
            None,
            None,
            None,
        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6) - 1e-6,
            np.ones(10),
            2 * np.ones(5),
            None,
            None,
            None,
        ),
        (
            np.linspace(1, -1, num=11),
            np.linspace(-1, 1, num=21),
            np.ones(10),
            np.ones(20) / 2,
            0,
            ~0,
            None,
        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(1, -1, num=6),
            np.ones(10),
            2 * np.ones(5),
            ~0,
            0,
            None,
        ),
        (
            np.linspace(1, -1, num=11),
            np.linspace(1, -1, num=6),
            np.ones(10),
            2 * np.ones(5),
            (0,),
            0,
            None,
        ),
        (
            np.broadcast_to(np.linspace(1, -1, num=11), (3, 4, 11)),
            np.linspace(1, -1, num=6),
            np.ones(10),
            2 * np.ones(5),
            ~0,
            0,
            None,
        ),
        (
            np.broadcast_to(
                np.linspace(1, -1, num=11)[..., np.newaxis, np.newaxis], (11, 3, 4)
            ),
            np.linspace(1, -1, num=6),
            np.ones(10)[..., np.newaxis, np.newaxis],
            2 * np.ones(5),
            0,
            ~0,
            None,
        ),
        (
            np.linspace(1, -1, num=11),
            np.broadcast_to(np.linspace(1, -1, num=6), (3, 4, 6)),
            np.ones(10),
            2 * np.ones(5),
            ~0,
            ~0,
            None,
        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(1, 2, num=6),
            np.ones(10),
            0,
            None,
            None,
            None,
        ),
        (
            np.linspace(1, -1, num=11),
            np.linspace(1, -1, num=6),
            np.broadcast_to(np.ones(10), shape=(3, 4, 10)),
            2 * np.ones(5),
            ~0,
            ~0,
            None,
        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6),
            np.ones(10),
            2 * np.ones(5),
            None,
            None,
            1,
        ),
    ],
)
def test_weights_conservative_1d(
    x_input: np.ndarray | tuple[np.ndarray],
    x_output: np.ndarray | tuple[np.ndarray],
    values_input: np.ndarray,
    values_expected: np.ndarray,
    axis_input: None | int | tuple[int],
    axis_output: None | int | tuple[int],
    weights_input: None | np.ndarray,
):
    weights_output = regridding.weights(
        coordinates_input=x_input,
        coordinates_output=x_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        method="conservative",
    )

    values_output = regridding.regrid_from_weights(
        *weights_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert np.allclose(values_output, values_expected)

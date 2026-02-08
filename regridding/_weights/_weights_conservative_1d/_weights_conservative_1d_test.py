import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    "x_input, x_output, values_input, values_expected, axis",
    [
        (
            np.linspace(-1, 1, num=11),
            np.linspace(-1, 1, num=6) + 1e-6,
            np.ones(10),
            2 * np.ones(5),
            None
        ),
        (
            np.linspace(1, -1, num=11),
            np.linspace(-1, 1, num=21) + 1e-6,
            np.ones(10),
            np.ones(20) / 2,
            0,

        ),
        (
            np.linspace(-1, 1, num=11),
            np.linspace(1, -1, num=6) - 1e-6,
            np.ones(10),
            2 * np.ones(5),
            ~0,
        ),
        (
            np.linspace(1, -1, num=11),
            np.linspace(1, -1, num=6) - 1e-6,
            np.ones(10),
            2 * np.ones(5),
            (0,),
        ),
        (
            np.linspace(np.arange(5), np.arange(5) + 1, num=11, axis=~0),
            np.linspace(np.arange(5), np.arange(5) + 1, num=6, axis=~0) + 1e-6,
            np.ones(10),
            2 * np.ones(5),
            ~0,
        )
    ],
)
def test_regrid_conservative_1d(
    x_input: np.ndarray | tuple[np.ndarray],
    x_output: np.ndarray | tuple[np.ndarray],
    values_input: np.ndarray,
    values_expected: np.ndarray,
    axis: None | int | tuple[int],
):
    values_output = regridding.regrid(
        coordinates_input=x_input,
        coordinates_output=x_output,
        values_input=values_input,
        axis_input=axis,
        axis_output=axis,
        method="conservative",
    )

    assert np.allclose(values_output, values_expected)

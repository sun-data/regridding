from typing import Literal
import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="coordinates_input,coordinates_output",
    argvalues=[
        (
            (np.linspace(-1, 1, num=32),),
            (np.linspace(-1, 1, num=64),),
        ),
        (
            (np.linspace(-1, 1, num=32),),
            (np.linspace(-2, 2, num=64),),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "brute",
        "searchsorted",
        pytest.param("invalid method", marks=pytest.mark.xfail),
    ],
)
def test_find_indices_1d(
    coordinates_input: tuple[np.ndarray],
    coordinates_output: tuple[np.ndarray],
    method: Literal["brute", "searchsorted"],
):
    result = regridding.find_indices(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        method=method,
    )

    (coordinates_input_x,) = coordinates_input
    (coordinates_output_x,) = coordinates_output
    (result_x,) = result

    where = coordinates_output_x <= coordinates_input_x.max()
    where &= coordinates_output_x > coordinates_input_x.min()

    assert np.all(
        coordinates_input_x[result_x[where] + 0] <= coordinates_output_x[where]
    )
    assert np.all(
        coordinates_input_x[result_x[where] + 1] >= coordinates_output_x[where]
    )

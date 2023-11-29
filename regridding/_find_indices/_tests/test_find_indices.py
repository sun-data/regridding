from typing import Literal
import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="vertices_input,vertices_output",
    argvalues=[
        (
            (np.linspace(-1, 1, num=32),),
            (np.linspace(-1, 1, num=64),),
        ),
    ],
)
@pytest.mark.parametrize("method", ["brute"])
def test_find_indices_1d(
    vertices_input: tuple[np.ndarray],
    vertices_output: tuple[np.ndarray],
    method: Literal["brute"],
):
    result = regridding.find_indices(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        method=method,
    )

    (vertices_input_x,) = vertices_input
    (vertices_output_x,) = vertices_output
    (result_x,) = result

    assert np.all(vertices_input_x[result_x + 0] <= vertices_output_x)
    assert np.all(vertices_input_x[result_x + 1] >= vertices_output_x)

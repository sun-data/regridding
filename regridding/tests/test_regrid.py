import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="vertices_input, values_input, axis_input",
    argvalues=[
        (
            np.meshgrid(
                np.linspace(-1, 1, num=10),
                np.linspace(1, 1, num=11),
                indexing="ij",
            ),
            np.random.normal((10, 11)),
            None,
        ),
    ]
)
@pytest.mark.parametrize(
    argnames="vertices_output, values_output, axis_output",
    argvalues=[
        (
            np.meshgrid(
                1.1 * np.linspace(-1, 1, num=10) + 0.001,
                1.2 * np.linspace(1, 1, num=11) + 0.001,
                indexing="ij",
            ),
            None,
            None,
        )
    ]
)
@pytest.mark.parametrize("order", [1])
def test_regrid_conservative_2d(
    vertices_input: tuple[np.ndarray, ...],
    vertices_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    order: int = 1,
):
    result = regridding.regrid(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
        order=order,
    )

    assert np.issubdtype(result.dtype, float)
    assert result.shape == np.broadcast(*vertices_output).shape
    assert result.sum() == values_input.sum()

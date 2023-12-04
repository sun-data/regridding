import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="coordinates_input,coordinates_output,values_input,values_output,axis_input,axis_output,result_expected",
    argvalues=[
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            np.empty(shape=11),
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
    ],
)
def test_regrid_multilinear_1d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    result_expected: np.ndarray,
):
    result = regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="multilinear",
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, float)
    assert np.all(result == result_expected)


@pytest.mark.parametrize(
    argnames="coordinates_input, values_input, axis_input",
    argvalues=[
        (
            np.meshgrid(
                np.linspace(-1, 1, num=10),
                np.linspace(-1, 1, num=11),
                indexing="ij",
            ),
            np.random.normal(size=(10 - 1, 11 - 1)),
            None,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="coordinates_output, values_output, axis_output",
    argvalues=[
        (
            np.meshgrid(
                1.1 * np.linspace(-1, 1, num=10) + 0.001,
                1.2 * np.linspace(-1, 1, num=11) + 0.001,
                indexing="ij",
            ),
            None,
            None,
        )
    ],
)
def test_regrid_conservative_2d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
):
    result = regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )

    assert np.issubdtype(result.dtype, float)
    assert result.shape == tuple(np.array(np.broadcast(*coordinates_output).shape) - 1)
    assert np.isclose(result.sum(), values_input.sum())

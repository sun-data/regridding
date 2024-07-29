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
        (
            (y,),
            (new_y,),
            x_broadcasted + y_broadcasted,
            None,
            (~0,),
            (~0,),
            new_x_broadcasted + new_y_broadcasted,
        ),
        (
            (x[..., np.newaxis],),
            (new_x[..., np.newaxis],),
            x_broadcasted + y_broadcasted,
            None,
            (0,),
            (0,),
            new_x_broadcasted_2 + new_y_broadcasted_2,
        ),
        (
            (x[..., np.newaxis],),
            (0.1 * new_x[..., np.newaxis] + 0.001 * new_y,),
            x[..., np.newaxis],
            None,
            (0,),
            (0,),
            0.1 * new_x[..., np.newaxis] + 0.001 * new_y,
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
    assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames="coordinates_input, values_input, axis_input, coordinates_output, values_output, axis_output",
    argvalues=[
        (
            (x_broadcasted, y_broadcasted),
            np.random.normal(size=(10 - 1, 11 - 1)),
            None,
            (1.1 * x_broadcasted + 0.01, 1.2 * y_broadcasted + 0.01),
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
        ),
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

    result_shape = np.array(np.broadcast(*coordinates_output).shape)

    if axis_input is None:
        result_shape = result_shape - 1
    else:
        for ax in axis_input:
            result_shape[ax] = result_shape[ax] - 1

    assert np.issubdtype(result.dtype, float)
    assert result.shape == tuple(result_shape)
    assert np.isclose(result.sum(), values_input.sum())

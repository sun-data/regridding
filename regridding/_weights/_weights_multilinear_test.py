import pytest
import numpy as np
import astropy.units as u
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
    "weights_input,"
    "result_expected",
    argvalues=[
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            None,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11) * u.mm,),
            (np.linspace(-1, 1, num=11) * u.mm,),
            np.square(np.linspace(-1, 1, num=11)),
            None,
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
            None,
            new_x_broadcasted + new_y_broadcasted,
        ),
        (
            (x[..., np.newaxis],),
            (new_x[..., np.newaxis],),
            x_broadcasted + y_broadcasted,
            None,
            (0,),
            (0,),
            None,
            new_x_broadcasted_2 + new_y_broadcasted_2,
        ),
        (
            (x[..., np.newaxis],),
            (0.1 * new_x[..., np.newaxis] + 0.001 * new_y,),
            x[..., np.newaxis],
            None,
            (0,),
            (0,),
            None,
            0.1 * new_x[..., np.newaxis] + 0.001 * new_y,
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            1,
            np.square(np.linspace(-1, 1, num=11)),
        ),
        (
            (np.linspace(-1, 1, num=11),),
            (np.linspace(-1, 1, num=11),),
            np.square(np.linspace(-1, 1, num=11)),
            None,
            None,
            None,
            0,
            0,
        ),
    ],
)
def test_weights_multilinear_1d(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    values_output: None | np.ndarray,
    axis_input: None | int | tuple[int, ...],
    axis_output: None | int | tuple[int, ...],
    weights_input: None | np.ndarray,
    result_expected: np.ndarray,
):
    weights = regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        method="multilinear",
    )
    result = regridding.regrid_from_weights(
        *weights,
        values_input=values_input,
        values_output=values_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, float)
    assert np.allclose(result, result_expected)

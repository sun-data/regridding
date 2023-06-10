import pytest
import numpy as np
import scipy.ndimage
import regridding


sz_t = 9
sz_x = 10
sz_y = 11
sz_z = 12


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        np.random.random(sz_x),
    ]
)
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.arange(sz_x),
        np.linspace(0, sz_x - 1, num=21),
    ]
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[None, 0, -1]
)
def test_ndarray_linear_interpolation_1d(
        a: np.ndarray,
        x: np.ndarray,
        axis: None | int | tuple[int]
):
    result = regridding.ndarray_linear_interpolation(a=a, indices=(x,), axis=axis)
    expected = scipy.ndimage.map_coordinates(input=a, coordinates=x[np.newaxis], mode="nearest", order=1)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        np.random.random((sz_x, sz_y)),
    ]
)
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.arange(sz_x)[:, np.newaxis],
        np.linspace(0, sz_x - 1, num=100)[:, np.newaxis],
    ]
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        np.arange(sz_y)[np.newaxis, :],
        np.linspace(0, sz_y - 1, num=5)[np.newaxis, :],
    ]
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        None,
        (0, 1),
        (0, ~0),
    ]
)
def test_ndarray_linear_interpolation_2d(
        a: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        axis: None | tuple[int],
):
    x, y = np.broadcast_arrays(x, y)

    result = regridding.ndarray_linear_interpolation(a=a, indices=(x, y), axis=axis)
    expected = scipy.ndimage.map_coordinates(input=a, coordinates=np.stack([x, y]), order=1)

    assert np.allclose(result, expected)

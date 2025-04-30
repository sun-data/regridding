import pytest
import numpy as np
from . import _weights_conservative_3d

_x = np.arange(3)
_y = np.arange(4)
_z = np.arange(5)

_x, _y, _z = np.meshgrid(_x, _y, _z, indexing="ij")


@pytest.mark.parametrize(
    argnames="grid_input,grid_output",
    argvalues=[
        (
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
        )
    ]
)
def test_weights_conservative_3d(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    result = _weights_conservative_3d.weights_conservative_3d(
        grid_input=grid_input,
        grid_output=grid_output,
    )


@pytest.mark.parametrize(
    argnames="grid,result_expected",
    argvalues=[
        (
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
            np.ones((1, 1, 1))
        ),
        (
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                np.arange(5),
                indexing="ij",
            ),
            np.ones((2, 3, 4))
        ),
        (
            np.meshgrid(
                2 * np.arange(3),
                2 * np.arange(4),
                2 * np.arange(5),
                indexing="ij",
            ),
            8 * np.ones((2, 3, 4))
        ),
    ]
)
def test_cell_volume(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    result_expected: np.ndarray,
):
    result = _weights_conservative_3d._cell_volume(grid)
    assert np.allclose(result, result_expected)
    assert result.shape == result_expected.shape


@pytest.mark.parametrize(
    argnames="point,grid,result_expected",
    argvalues=[
        (
            (0.5, 0.5, 0.5),
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                np.arange(5),
                indexing="ij",
            ),
            (0, 0, 0),
        ),
        (
            (-0.5, -0.5, 3.5),
            np.meshgrid(
                -np.arange(3),
                -np.arange(4),
                np.arange(5),
                indexing="ij",
            ),
            (0, 0, 3),
        ),
    ],
)
def test_index_of_point_brute(
    point: tuple[float, float, float],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    result_expected: np.ndarray,
):
    result = _weights_conservative_3d._index_of_point_brute(
        point=point,
        grid=grid,
    )

    assert result == result_expected

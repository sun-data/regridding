import pytest
import numpy as np
from . import _grids


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
def test_volume_grid(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    result_expected: np.ndarray,
):
    result = _grids.grid_volume(grid)
    assert np.allclose(result, result_expected)
    assert result.shape == result_expected.shape

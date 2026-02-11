import pytest
import numpy as np
from . import _grids

grid = np.arange(5)


@pytest.mark.parametrize(
    argnames="point,grid,result_expected",
    argvalues=[
        (0.5, grid, 1),
        (3.5, grid, 4),
    ],
)
def test_index_of_point(
    point: float,
    grid: np.ndarray,
    result_expected: int,
):
    result = _grids.index_of_point(
        point=point,
        grid=grid,
    )
    assert result == result_expected

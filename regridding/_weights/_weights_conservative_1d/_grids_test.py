import pytest
import numpy as np
from . import _grids

_grid = np.arange(5)


@pytest.mark.parametrize(
    argnames="point,grid,result_expected",
    argvalues=[
        (0.5, _grid, 1),
        (3.5, _grid, 4),
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

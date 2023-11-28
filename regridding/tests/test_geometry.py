import pytest
import numpy as np
import regridding


@pytest.mark.parametrize("x", [0, 0.5, 1])
@pytest.mark.parametrize("y_offset", [-1, 0, 2])
@pytest.mark.parametrize("x1", [-1, 0])
@pytest.mark.parametrize("y1", [-2, 0])
@pytest.mark.parametrize("x2", [2, 3])
@pytest.mark.parametrize("y2", [2])
def test_line_equation_2d(
    x: float,
    y_offset: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    y_below = (x - x1) * (y2 - y1) / (x2 - x1) + y1 - 1
    y_equal = (x - x1) * (y2 - y1) / (x2 - x1) + y1 + 0
    y_above = (x - x1) * (y2 - y1) / (x2 - x1) + y1 + 1

    result_below = regridding.geometry.line_equation_2d(
        x=x,
        y=y_below,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    result_equal = regridding.geometry.line_equation_2d(
        x=x,
        y=y_equal,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    result_above = regridding.geometry.line_equation_2d(
        x=x,
        y=y_above,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    assert np.isclose(result_equal, 0)
    assert np.sign(result_above) != np.sign(result_below)

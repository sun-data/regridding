import pytest
import math
import regridding


@pytest.mark.parametrize(
    argnames="x,result_expected",
    argvalues=[
        (-2, -1),
        (-1, -1),
        (-0, 0),
        (0, 0),
        (1, 1),
        (0.5, 1),
        (math.nan, 0),
        (math.inf, 1),
        (-math.inf, -1),
    ],
)
def test_sign(
    x: float,
    result_expected: float,
):
    result = regridding.math.sign(x)
    assert result == result_expected

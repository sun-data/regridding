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


@pytest.mark.parametrize(
    argnames="a,result_expected",
    argvalues=[
        ((1, 1, 1), (-1, -1, -1)),
        ((-1, 1, 1), (1, -1, -1)),
    ],
)
def test_negate_3d(
    a: tuple[float, float, float],
    result_expected: tuple[float, float, float],
):
    result = regridding.math.negate_3d(a)
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="a,b,result_expected",
    argvalues=[
        ((1, 1, 1), (-1, -1, -1), (0, 0, 0)),
        ((1, 1, 1), (1, 1, 1), (2, 2, 2)),
    ],
)
def test_sum_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    result_expected: tuple[float, float, float],
):
    result = regridding.math.sum_3d(a, b)
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="a,b,result_expected",
    argvalues=[
        ((1, 1, 1), (-1, -1, -1), (2, 2, 2)),
        ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
    ],
)
def test_difference_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    result_expected: tuple[float, float, float],
):
    result = regridding.math.difference_3d(a, b)
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="r,a,result_expected",
    argvalues=[
        (2, (1, 1, 1), (2, 2, 2)),
    ],
)
def test_multiply_3d(
    r: float,
    a: tuple[float, float, float],
    result_expected: tuple[float, float, float],
):
    result = regridding.math.multiply_3d(r, a)
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="a,b,result_expected",
    argvalues=[
        ((1, 1, 1), (1, 1, 1), 3),
        ((1, 1, 1), (2, 2, 2), 6),
    ],
)
def test_dot_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    result_expected: float,
):
    result = regridding.math.dot_3d(a, b)
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="a,b,result_expected",
    argvalues=[
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ],
)
def test_cross_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    result_expected: float,
):
    result = regridding.math.cross_3d(a, b)
    assert result == result_expected

import pytest
import numpy as np
from . import _arrays


@pytest.mark.parametrize(
    argnames="index,shape,result_expected",
    argvalues=[
        (
            (0, 0),
            (11, 12),
            True,
        ),
        (
            (5, 5),
            (11, 12),
            True,
        ),
        (
            (-1, 0),
            (11, 12),
            False,
        ),
        (
            (0, 12),
            (11, 12),
            False,
        ),
    ],
)
def test_index_in_bounds(
    index: tuple[int, int],
    shape: tuple[int, int],
    result_expected: bool,
):
    result = _arrays.index_in_bounds(index=index, shape=shape)

    assert result == result_expected


@pytest.mark.parametrize(
    argnames="index",
    argvalues=[
        (1, 1),
        (5, 6),
    ],
)
@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        (11, 12),
    ],
)
def test_index_flat(
    index: tuple[int, int],
    shape: tuple[int, int],
):
    result = _arrays.index_flat(index=index, shape=shape)
    result_expected = np.ravel_multi_index(index, shape)

    assert result == result_expected


@pytest.mark.parametrize(
    argnames="index",
    argvalues=[0, 11],
)
@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        (11, 12),
    ],
)
def test_index_2d(
    index: int,
    shape: tuple[int, int],
):
    result = _arrays.index_2d(index=index, shape=shape)
    result_expected = np.unravel_index(index, shape)

    assert result == result_expected
    assert index == np.ravel_multi_index(result, shape)

from typing import Any
import pytest
import numpy as np
import astropy.units as u
from regridding import _util


def test_unbroadcast() -> None:
    """
    An axis along which every array has been broadcast is reduced to a
    length of one, unless it is one of the axes to leave whole.
    """
    a = np.broadcast_to(np.arange(3.0)[:, np.newaxis], (3, 4))
    b = np.ones((3, 4))

    assert _util._unbroadcast((a,))[0].shape == (3, 1)
    assert _util._unbroadcast((a,), axis=(1,))[0].shape == (3, 4)
    assert [c.shape for c in _util._unbroadcast((a, b))] == [(3, 4), (3, 4)]


@pytest.mark.parametrize(
    argnames="a, expected",
    argvalues=[
        ([1, 2], [1, 2]),
        (np.array([1, 2], dtype=np.float32), [1, 2]),
        (np.array([1.0, 2.0]) * u.dimensionless_unscaled, [1, 2]),
        (np.array([50.0, 150.0]) * u.percent, [0.5, 1.5]),
        (np.array([50.0, 150.0], dtype=np.float32) * u.percent, [0.5, 1.5]),
    ],
)
def test_dimensionless(a: Any, expected: list[float]) -> None:
    """A dimensionless array is the number it stands for, in double precision."""
    result = _util._dimensionless(a)
    assert result.dtype == np.float64
    assert np.allclose(result, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        np.broadcast_to(np.arange(4, dtype=np.float32), (3, 4)),
        np.broadcast_to(np.arange(4.0) * u.percent, (3, 4), subok=True),
    ],
    ids=["single", "percent"],
)
def test_dimensionless_broadcast(a: Any) -> None:
    """
    An array which has been broadcast is converted as it was before, and
    broadcast again, rather than copied out to its full shape.
    """
    result = _util._dimensionless(a)
    assert result.shape == (3, 4)
    assert result.strides[0] == 0
    assert np.allclose(result, _util._dimensionless(a.copy()), rtol=1e-15, atol=0)


def test_dimensionless_units() -> None:
    """
    A unit with dimensions raises, naming the array, unless it is not
    strict, when the unit is dropped.
    """
    a = np.array([1.0, 2.0]) * u.mm

    with pytest.raises(ValueError, match="the kernel must be dimensionless, got mm"):
        _util._dimensionless(a, name="the kernel")

    assert np.array_equal(_util._dimensionless(a, strict=False), [1, 2])

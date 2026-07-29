import pytest
import numpy as np
import regridding

_num_x = 11
_num_y = 12
_num_t = 13


@pytest.mark.parametrize(
    argnames="a,where,axis",
    argvalues=[
        (
            np.random.uniform(0, 1, size=(_num_x, _num_y)),
            np.random.uniform(0, 1, size=(_num_x, _num_y)) > 0.9,
            None,
        ),
        (
            np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y)),
            np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y)) > 0.9,
            (~1, ~0),
        ),
        (
            np.sqrt(np.random.uniform(-0.1, 1, size=(_num_x, _num_t, _num_y))),
            None,
            (0, ~0),
        ),
    ],
)
@pytest.mark.parametrize("guess", [None, 0.5])
@pytest.mark.parametrize("num_iterations", [11])
def test_fill_gauss_sidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    axis: None | tuple[int, ...],
    guess: None | float | np.ndarray,
    num_iterations: int,
):
    result = regridding.fill(
        a=a,
        where=where,
        axis=axis,
        method="gauss_seidel",
        guess=guess,
        num_iterations=num_iterations,
    )
    if where is None:
        where = np.isnan(a)

    assert np.all(np.isfinite(result))
    assert np.allclose(result[~where], a[~where])
    assert np.all(result[where] != 0)


@pytest.mark.parametrize(
    argnames="guess",
    argvalues=[
        None,
        2.0,
        np.arange(_num_t).reshape(_num_t, 1, 1),
    ],
)
def test_fill_gauss_seidel_guess(
    guess: None | float | np.ndarray,
):
    """The missing elements start at `guess`, so zero iterations returns it."""

    a = np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y))
    where = np.random.uniform(0, 1, size=a.shape) > 0.9

    result = regridding.fill(
        a=a,
        where=where,
        axis=(~1, ~0),
        guess=guess,
        num_iterations=0,
    )

    if guess is None:
        guess = np.nanmedian(np.where(where, np.nan, a), axis=(~1, ~0), keepdims=True)

    assert np.allclose(result[~where], a[~where])
    assert np.allclose(result[where], np.broadcast_to(guess, a.shape)[where])


def test_fill_gauss_seidel_missing_cluster():
    """A contiguous block of missing elements is filled with finite values."""

    x = np.linspace(-1, 1, num=32)
    a = x[:, np.newaxis] + 2 * x[np.newaxis, :]

    a_missing = a.copy()
    a_missing[10:16, 10:16] = np.nan

    result = regridding.fill(a_missing, num_iterations=1000)

    assert np.all(np.isfinite(result))

    # `a` is harmonic, so the relaxation recovers it exactly
    assert np.allclose(result, a, atol=1e-4)


def test_fill_gauss_seidel_all_missing():
    """A slice with no valid elements falls back to a guess of zero."""

    a = np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y))
    where = np.random.uniform(0, 1, size=a.shape) > 0.9
    where[0] = True

    result = regridding.fill(
        a=a,
        where=where,
        axis=(~1, ~0),
        num_iterations=11,
    )

    assert np.all(np.isfinite(result))
    assert np.all(result[0] == 0)
    assert np.allclose(result[~where], a[~where])

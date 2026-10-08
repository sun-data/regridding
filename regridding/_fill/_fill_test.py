import warnings
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
) -> None:
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
) -> None:
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


def test_fill_gauss_seidel_missing_cluster() -> None:
    """A contiguous block of missing elements is filled with finite values."""

    x = np.linspace(-1, 1, num=32)
    a = x[:, np.newaxis] + 2 * x[np.newaxis, :]

    a_missing = a.copy()
    a_missing[10:16, 10:16] = np.nan

    result = regridding.fill(a_missing, num_iterations=1000)

    assert np.all(np.isfinite(result))

    # `a` is harmonic, so the relaxation recovers it exactly
    assert np.allclose(result, a, atol=1e-4)


def test_fill_gauss_seidel_all_missing() -> None:
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


def test_fill_gauss_seidel_nan_neighbors() -> None:
    """
    NaN elements which are not being filled are left out as neighbors,
    and are left as NaN.
    """

    y = np.linspace(-1, 1, num=24)
    a = np.broadcast_to(y[:, np.newaxis], (24, 20)).copy()

    # The unknown elements, left of the elements being filled
    a[:, :5] = np.nan

    where = np.zeros(a.shape, dtype=bool)
    where[8:16, 5:10] = True

    result = regridding.fill(a, where=where, num_iterations=1000)

    assert np.all(np.isnan(result[:, :5]))
    assert np.all(np.isfinite(result[:, 5:]))

    # `a` is harmonic, and has zero gradient across the unknown elements,
    # so the relaxation recovers it exactly
    assert np.allclose(result[:, 5:], a[:, 5:], atol=1e-6)


@pytest.mark.parametrize("transpose", [False, True])
def test_fill_gauss_seidel_edges(
    transpose: bool,
) -> None:
    """The edges of the array are not periodic, along either axis."""

    x = np.linspace(-1, 1, num=20)
    a = np.broadcast_to(x, (12, 20)).copy()

    where = np.zeros(a.shape, dtype=bool)
    where[:, :5] = True

    expected = np.broadcast_to(a[:, 5:6], a.shape)

    if transpose:
        a, where, expected = a.T, where.T, expected.T

    result = regridding.fill(a, where=where, num_iterations=1000)

    # The filled elements are only connected to the first valid column,
    # so they relax to its value rather than toward the far edge of the array.
    assert np.allclose(result[where], expected[where], atol=1e-6)


def test_fill_gauss_seidel_isotropic() -> None:
    """Neighbors along each axis are weighted equally, whatever the shape."""

    a = np.random.uniform(0, 1, size=(5, 9))

    where = np.zeros(a.shape, dtype=bool)
    where[2, 4] = True

    result = regridding.fill(a, where=where, num_iterations=1)

    mean = (a[1, 4] + a[3, 4] + a[2, 3] + a[2, 5]) / 4
    assert np.isclose(result[2, 4], mean)


def test_fill_gauss_seidel_no_valid_neighbors() -> None:
    """An element with no valid neighbors is left at the guess."""

    a = np.full((3, 3), np.nan)

    where = np.zeros(a.shape, dtype=bool)
    where[1, 1] = True

    result = regridding.fill(a, where=where, guess=0.5, num_iterations=11)

    assert result[1, 1] == 0.5
    assert np.all(np.isnan(result[~where]))


@pytest.mark.parametrize(
    argnames="shape_a,shape_where",
    argvalues=[
        ((_num_x, _num_y), (_num_t, _num_x, _num_y)),
        ((_num_t, _num_x, _num_y), (_num_x, _num_y)),
    ],
)
def test_fill_gauss_seidel_broadcast(
    shape_a: tuple[int, ...],
    shape_where: tuple[int, ...],
) -> None:
    """
    Each slice of arrays broadcast against each other is filled on its own,
    as if it had been filled by itself.
    """

    rng = np.random.default_rng(seed=0)

    a = rng.uniform(0, 1, size=shape_a)
    where = rng.uniform(0, 1, size=shape_where) > 0.7

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = regridding.fill(a, where=where, axis=(~1, ~0), num_iterations=11)

    a, where = np.broadcast_arrays(a, where)

    for t in range(_num_t):
        expected = regridding.fill(a[t], where=where[t], num_iterations=11)
        assert np.allclose(result[t], expected)


def test_fill_gauss_seidel_infinite_neighbors() -> None:
    """
    Infinite elements which are not being filled are left out as neighbors,
    like NaN elements, and are left as they are.
    """

    y = np.linspace(-1, 1, num=24)
    a = np.broadcast_to(y[:, np.newaxis], (24, 20)).copy()

    a[:, :5] = np.inf
    a[::2, :5] = -np.inf

    where = np.zeros(a.shape, dtype=bool)
    where[8:16, 5:10] = True

    result = regridding.fill(a, where=where, num_iterations=1000)

    assert np.all(result[:, :5] == a[:, :5])
    assert np.allclose(result[:, 5:], a[:, 5:], atol=1e-6)


def test_fill_gauss_seidel_guess_nan() -> None:
    """
    Missing elements whose guess is NaN start from their neighbors,
    and the NaN does not spread through the rest of the missing elements.
    """

    x = np.linspace(-1, 1, num=32)
    a = x[:, np.newaxis] + 2 * x[np.newaxis, :]

    where = np.zeros(a.shape, dtype=bool)
    where[10:16, 10:16] = True

    guess = np.zeros(a.shape)
    guess[12, 12] = np.nan

    result = regridding.fill(a, where=where, guess=guess, num_iterations=1000)
    assert np.allclose(result, a, atol=1e-4)

    result = regridding.fill(a, where=where, guess=np.nan, num_iterations=1000)
    assert np.allclose(result, a, atol=1e-4)


def test_fill_gauss_seidel_cut_off() -> None:
    """
    Missing elements which are cut off from every valid element are left at
    the guess, or as NaN if the guess is NaN.
    """

    a = np.random.uniform(0, 1, size=(_num_x, _num_y))

    # A ring of unknown elements around the elements being filled
    a[2:7, 2:7] = np.nan

    where = np.zeros(a.shape, dtype=bool)
    where[3:6, 3:6] = True

    median = np.nanmedian(np.where(where, np.nan, a))

    # The elements average each other, which can round the median
    result = regridding.fill(a, where=where, num_iterations=11)
    assert np.allclose(result[where], median)

    result = regridding.fill(a, where=where, guess=np.nan, num_iterations=11)
    assert np.all(np.isnan(result[where]))


def test_fill_gauss_seidel_axis() -> None:
    """Only two interpolation axes are supported."""

    a = np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y))

    with pytest.raises(ValueError, match="The number of interpolation axes, 3,"):
        regridding.fill(a)


def test_fill_method() -> None:
    """An unrecognized method is named in the error."""

    a = np.random.uniform(0, 1, size=(_num_x, _num_y))

    with pytest.raises(ValueError, match="Unrecognized method 'foo'"):
        regridding.fill(a, method="foo")  # type: ignore[arg-type]

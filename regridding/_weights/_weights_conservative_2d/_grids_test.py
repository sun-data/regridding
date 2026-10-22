from typing import Callable
import pytest
import sys
import numpy as np
from . import _grids


@pytest.mark.parametrize(
    argnames="grid,result_expected",
    argvalues=[
        (
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
            np.ones((1, 1)),
        ),
        (
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            np.ones((2, 3)),
        ),
        (
            np.meshgrid(
                2 * np.arange(3),
                2 * np.arange(4),
                indexing="ij",
            ),
            4 * np.ones((2, 3)),
        ),
    ],
)
def test_volume_grid(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    result_expected: np.ndarray,
) -> None:
    result = _grids.grid_volume(grid)
    assert np.allclose(result, result_expected)
    assert result.shape == result_expected.shape


@pytest.mark.parametrize(
    argnames="point,grid,result_expected",
    argvalues=[
        (
            (0.5, 0.5),
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            (0, 0),
        ),
        (
            (1.5, 0.5),
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            (1, 0),
        ),
        (
            (1.5, 0.5),
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                # indexing="ij",
            ),
            (0, 1),
        ),
        (
            (-0.5, 2.5),
            np.meshgrid(
                -np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            (0, 2),
        ),
        (
            (-1, -1),
            np.meshgrid(
                -np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            (sys.maxsize, sys.maxsize),
        ),
        (
            (1, 4.5),
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                indexing="ij",
            ),
            (sys.maxsize, sys.maxsize),
        ),
        # the last vertex, whose fractional index floors to a cell past the
        # end of the grid
        (
            (4, 5),
            np.meshgrid(
                np.arange(5),
                np.arange(6),
                indexing="ij",
            ),
            (3, 4),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="func",
    argvalues=[
        _grids.index_of_point_brute,
        _grids.index_of_point_secant,
    ],
)
def test_index_of_point(
    func: Callable,
    point: tuple[float, float],
    grid: tuple[np.ndarray, np.ndarray],
    result_expected: tuple[int, int],
) -> None:
    result = func(
        point=point,
        grid=grid,
    )

    assert result == result_expected


def _annulus(
    radius_inner: float,
) -> tuple[np.ndarray, np.ndarray]:
    """An annulus which wraps most of the way around."""
    r, t = np.meshgrid(
        np.linspace(radius_inner, 2, 6),
        np.linspace(0, 0.9 * 2 * np.pi, 37),
        indexing="ij",
    )
    return r * np.cos(t), r * np.sin(t)


def _alternating() -> tuple[np.ndarray, np.ndarray]:
    """A grid whose cells alternate in width, on which Newton's method cycles."""
    widths = np.tile([0.4, 1.0], 8)
    return np.meshgrid(
        np.concatenate([[0], np.cumsum(widths)]),
        np.arange(6.0),
        indexing="ij",
    )


@pytest.mark.parametrize(
    argnames="grid",
    argvalues=[
        # an absolute tolerance on the position once returned every point of
        # a grid this small as outside it
        tuple(1e-12 * c for c in np.meshgrid(np.arange(9.0), np.arange(7.0))),
        tuple(1e9 * c for c in np.meshgrid(np.arange(9.0), np.arange(7.0))),
        # Newton's method followed the continuation of a cell across the hole
        # and returned many points of an annulus as outside it
        _annulus(radius_inner=1),
        _annulus(radius_inner=0.05),
        _alternating(),
    ],
)
def test_index_of_point_secant_matches_brute(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = grid
    rng = np.random.default_rng(0)
    margin_x = 0.1 * np.ptp(x)
    margin_y = 0.1 * np.ptp(y)
    px = rng.uniform(x.min() - margin_x, x.max() + margin_x, 50)
    py = rng.uniform(y.min() - margin_y, y.max() + margin_y, 50)
    result = [_grids.index_of_point_secant((a, b), grid) for a, b in zip(px, py)]
    expected = [_grids.index_of_point_brute((a, b), grid) for a, b in zip(px, py)]
    assert result == expected
    assert any(r != (sys.maxsize, sys.maxsize) for r in result)

"""
Utilities for inspecting and searching logically-rectangular grids of
coordinates.
"""

import numba

__all__ = [
    "shape_centers",
]


@numba.njit(cache=True)
def shape_centers(
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Given the shape of the grid of cell vertices,
    compute the shape of the grid of cell centers.

    Parameters
    ----------
    shape
        The shape of the grid of cell vertices.
    """

    nx, ny, nz = shape

    return nx - 1, ny - 1, nz - 1

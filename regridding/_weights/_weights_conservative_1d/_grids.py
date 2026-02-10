import numpy as np
import numba

__all__ = [
    "cell_length",
    "index_of_point",
]


@numba.njit(cache=True)
def cell_length(
    grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the length of each cell in a 1-dimensional grid.

    Parameters
    ----------
    grid
        A 1D grid of cell vertices.
    """

    x = grid

    (num_i,) = x.shape

    result = np.zeros(num_i - 1)

    for i in numba.prange(num_i - 1):

        length = x[i + 1] - x[i]

        result[i] = length

    return result


@numba.njit(cache=True)
def index_of_point(
    point: float,
    grid: np.ndarray,
) -> int:
    """
    Given an ordered, 1-dimensional grid,
    find the index for which a new point should be inserted into to maintain
    the ordering.

    Parameters
    ----------
    point
        The value of the new point to insert into the grid.
    grid
        A sorted grid of points.
        Can be either increasing or decreasing.
    """

    (num,) = grid.shape

    index_left = 0
    index_right = num

    while (index_right - index_left) > 1:

        index_middle = (index_left + index_right) // 2

        grid_middle = grid[index_middle]

        if grid_middle > point:
            index_right = index_middle
        else:
            index_left = index_middle

    return index_right

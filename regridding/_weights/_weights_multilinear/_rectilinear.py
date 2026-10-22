"""
Find the cell of a rectilinear grid which contains each of a set of points.

A grid is rectilinear if each of its coordinates varies along one of its
axes only, a different axis for each coordinate, and is strictly monotonic
along it.  The cell along each axis is then a binary search of that
coordinate alone.  The search is exact, needs no preparation, and a point
outside the grid is extrapolated from the edge of the grid along each axis
on its own.
"""

import numpy as np
import numba
from regridding import _util

__all__ = [
    "profiles",
    "locate",
]


def profiles(
    coordinates: tuple[np.ndarray, ...],
) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The coordinate along each axis of a grid, if the grid is rectilinear.

    Returns :obj:`None` if the grid is not rectilinear.  Otherwise, returns
    ``(profile, start, coordinate, descending)``, where ``profile`` holds the
    values of the coordinate along each axis one after another, each sorted
    in ascending order, ``profile[start[m]:start[m + 1]]`` is the one along
    axis ``m``, ``coordinate[m]`` is which coordinate that is, and
    ``descending[m]`` is whether it was reversed to sort it.

    Parameters
    ----------
    coordinates
        The coordinates of the vertices of the grid, each an array with an
        axis for each axis of the grid.  An array which has been broadcast
        along an axis, and so has a stride of zero along it, is known not to
        vary along it without its values being looked at, which makes the
        grids that :func:`numpy.broadcast_to` or ``numpy.meshgrid(...,
        sparse=True)`` give quick to recognize.
    """
    num_axis = len(coordinates)
    shape = np.array(coordinates[0].shape, dtype=np.int64)

    grid = []
    coordinate = np.full(num_axis, -1, dtype=np.int64)
    for k in range(num_axis):
        (c,) = _util._unbroadcast((coordinates[k],))
        varies = [
            m
            for m in range(num_axis)
            if c.shape[m] > 1 and np.any(c != c.take([0], axis=m))
        ]
        if len(varies) != 1:
            return None
        (m,) = varies
        if coordinate[m] >= 0:
            return None
        coordinate[m] = k
        grid.append(c)

    profile = []
    descending = np.zeros(num_axis, dtype=np.bool_)
    for m in range(num_axis):
        index: list[int | slice] = [0] * num_axis
        index[m] = slice(None)
        values = grid[coordinate[m]][tuple(index)]
        difference = np.diff(values)
        if np.all(difference < 0):
            values = values[::-1]
            descending[m] = True
        elif not np.all(difference > 0):
            return None
        profile.append(values)

    start = np.zeros(num_axis + 1, dtype=np.int64)
    start[1:] = np.cumsum(shape)

    return (
        np.ascontiguousarray(np.concatenate(profile), dtype=np.float64),
        start,
        coordinate,
        descending,
    )


@numba.njit(cache=True, parallel=True)
def locate(
    profile: np.ndarray,
    start: np.ndarray,
    coordinate: np.ndarray,
    descending: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the cell of a rectilinear grid which contains each point.

    Returns ``(cell, xi, inside)``: the first corner of the cell along each
    axis, the local coordinates of the point in the cell, and whether the
    point is inside the grid.  A point outside the grid is given the cell at
    the edge of the grid along each axis it is outside along, with local
    coordinates outside of the unit interval, which extrapolate from that
    cell.  A point with a coordinate which is not finite is outside, and its
    local coordinate along that axis is NaN.

    Parameters
    ----------
    profile
        The coordinate along each axis, from :func:`profiles`.
    start
        Where the coordinate along each axis starts in `profile`.
    coordinate
        Which coordinate varies along each axis.
    descending
        Whether the coordinate along each axis was reversed.
    points
        The points to find, of shape ``(N, number of points)``.
    """
    num_axis, num_point = points.shape

    cell = np.empty((num_point, num_axis), dtype=np.int64)
    xi = np.empty((num_point, num_axis), dtype=np.float64)
    inside = np.empty(num_point, dtype=np.bool_)

    for i in numba.prange(num_point):
        inside_i = True
        for m in range(num_axis):
            values = profile[start[m] : start[m + 1]]
            num = values.shape[0]
            value = points[coordinate[m], i]
            if not (values[0] <= value <= values[num - 1]):
                inside_i = False
            j = np.searchsorted(values, value) - 1
            if j < 0:
                j = 0
            elif j > num - 2:
                j = num - 2
            fraction = (value - values[j]) / (values[j + 1] - values[j])
            if not np.isfinite(value):
                fraction = np.nan
            if descending[m]:
                j = num - 2 - j
                fraction = 1 - fraction
            cell[i, m] = j
            xi[i, m] = fraction
        inside[i] = inside_i

    return cell, xi, inside

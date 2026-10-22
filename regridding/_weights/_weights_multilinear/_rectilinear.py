"""
Find the cell of a rectilinear grid which contains each of a set of points.

A grid is rectilinear if each of its coordinates varies along one of its
axes only, a different axis for each coordinate, and is strictly monotonic
along it.  The cell along each axis is then a binary search of that
coordinate alone.  The search is exact, needs no preparation, and a point
outside the grid is extrapolated from the edge of the grid along each axis
on its own.

The grids of many elements, such as a spectrum for each pixel of an image,
are searched at once, each with its own coordinates.
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
    The coordinate along each axis of the grid of each element, if every
    grid is rectilinear in the same way.

    Returns :obj:`None` unless the same coordinate varies along the same
    axis of every grid, and is strictly monotonic along it.  Otherwise,
    returns ``(profile, start, coordinate, descending)``, where ``profile``
    holds, for each element, the values of the coordinate along each axis
    one after another, each sorted in ascending order,
    ``profile[:, start[m]:start[m + 1]]`` is the one along axis ``m``,
    ``coordinate[m]`` is which coordinate that is, and ``descending[:, m]``
    is whether it was reversed to sort it.  If every element shares one
    grid, ``profile`` and ``descending`` have a single row.

    Parameters
    ----------
    coordinates
        The coordinates of the vertices of the grid of each element, each an
        array with an axis for the elements followed by an axis for each axis
        of the grid.  An array which has been broadcast along an axis, and
        so has a stride of zero along it, is known not to vary along it
        without its values being looked at, which makes the grids that
        :func:`numpy.broadcast_to` or ``numpy.meshgrid(..., sparse=True)``
        give quick to recognize.
    """
    num_axis = len(coordinates)
    num_element = coordinates[0].shape[0]
    shape = np.array(coordinates[0].shape[1:], dtype=np.int64)

    grid = []
    coordinate = np.full(num_axis, -1, dtype=np.int64)
    for k in range(num_axis):
        (c,) = _util._unbroadcast((coordinates[k],))
        varies = [
            m
            for m in range(num_axis)
            if c.shape[1 + m] > 1 and np.any(c != c.take([0], axis=1 + m))
        ]
        if len(varies) != 1:
            return None
        (m,) = varies
        if coordinate[m] >= 0:
            return None
        coordinate[m] = k
        grid.append(c)

    num_row = 1 if all(c.shape[0] == 1 for c in grid) else num_element

    profile = []
    descending = np.zeros((num_row, num_axis), dtype=np.bool_)
    for m in range(num_axis):
        index: list[int | slice] = [slice(None)] + [0] * num_axis
        index[1 + m] = slice(None)
        # compared as floating point numbers, since the difference of two
        # unsigned integers wraps around instead of going negative
        values = np.asarray(grid[coordinate[m]][tuple(index)], dtype=np.float64)
        values = np.broadcast_to(values, (num_row, shape[m]))
        difference = np.diff(values, axis=~0)
        reverse = np.all(difference < 0, axis=~0)
        if not np.all(reverse | np.all(difference > 0, axis=~0)):
            return None
        values = np.where(reverse[:, np.newaxis], values[:, ::-1], values)
        descending[:, m] = reverse
        profile.append(values)

    start = np.zeros(num_axis + 1, dtype=np.int64)
    start[1:] = np.cumsum(shape)

    return (
        np.ascontiguousarray(np.concatenate(profile, axis=~0)),
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
    Find the cell of the rectilinear grid of each element which contains
    each of its points.

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
        The coordinate along each axis of each grid, from :func:`profiles`.
    start
        Where the coordinate along each axis starts in `profile`.
    coordinate
        Which coordinate varies along each axis.
    descending
        Whether the coordinate along each axis of each grid was reversed.
    points
        The points to find, of shape
        ``(number of elements, N, number of points)``.
    """
    num_element, num_axis, num_point = points.shape
    num_row = profile.shape[0]

    cell = np.empty((num_element, num_point, num_axis), dtype=np.int64)
    xi = np.empty((num_element, num_point, num_axis), dtype=np.float64)
    inside = np.empty((num_element, num_point), dtype=np.bool_)

    for t in numba.prange(num_element * num_point):
        e = t // num_point
        i = t - e * num_point
        row = min(e, num_row - 1)
        inside_i = True
        for m in range(num_axis):
            values = profile[row, start[m] : start[m + 1]]
            num = values.shape[0]
            value = points[e, coordinate[m], i]
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
            if descending[row, m]:
                j = num - 2 - j
                fraction = 1 - fraction
            cell[e, i, m] = j
            xi[e, i, m] = fraction
        inside[e, i] = inside_i

    return cell, xi, inside

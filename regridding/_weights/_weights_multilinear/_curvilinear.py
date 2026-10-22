"""
Find the cell of a curvilinear grid which contains each of a set of points.

The grid is searched through a hierarchy of bounding boxes.  Level zero is
the cells themselves.  Level one divides them into blocks of two cells along
every axis, or one at the far edge of an axis with an odd number of cells,
and each level above groups the blocks of the level below in the same way,
until one block holds the whole grid.  Only the boxes of level one and above
are stored; the box of a cell is taken from its corners when it is needed.

A point is looked for in the blocks whose boxes contain it, from the top
level down, and in each cell whose box contains it, its local coordinates
are found with Newton's method.  The point is in the first cell where they
are inside the unit cube.  Since the boxes of neighboring blocks overlap
little, a point reaches its cell in a number of steps which grows as the
logarithm of the size of the grid.

Searching the boxes first means that a grid which curves back on itself,
such as an annulus, cannot lead the search astray.  A search by Newton's
method across the whole grid can follow the continuation of a cell across
the hole of the annulus to a point on the other side, and conclude that the
point is outside the grid.  It can also cycle between neighboring cells of
very different sizes.

A point which is in no cell is outside the grid.  To extrapolate to it,
Newton's method is run across the whole grid in index space, with the cells
along its edges continued beyond it, starting from the cell whose bounding
box is nearest to the point.  On a grid of parallelograms, this finds the
same cell as clamping each index of the point to the grid.
"""

import numpy as np
import numba
from . import _cells

__all__ = [
    "tolerance",
    "tree",
    "locate",
]

tolerance = 1e-9
"""
How far outside the unit cube the local coordinates of a point may be, in
units of the cell, for the point to count as inside the cell.

A point on the face shared by two cells, or on the boundary of the grid,
has local coordinates on the boundary of the unit cube, and finding them
numerically leaves them off it by a rounding error.  Without some margin,
such a point could be outside every cell.
"""

num_iteration_walk = 64
"""
The most steps of Newton's method across the whole grid when extrapolating.
"""


@numba.njit(cache=True)
def _unravel(
    index: int,
    shape: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Convert a flat index into an index along each axis of an array stored in
    row-major order.

    Parameters
    ----------
    index
        The flat index.
    shape
        The shape of the array.
    out
        Where to put the index along each axis.
    """
    for m in range(shape.shape[0] - 1, -1, -1):
        out[m] = index % shape[m]
        index = index // shape[m]


@numba.njit(cache=True)
def _ravel(
    index: np.ndarray,
    shape: np.ndarray,
) -> int:
    """
    Convert an index along each axis of an array stored in row-major order
    into a flat index.

    Parameters
    ----------
    index
        The index along each axis.
    shape
        The shape of the array.
    """
    result = 0
    for m in range(shape.shape[0]):
        result = result * shape[m] + index[m]
    return result


@numba.njit(cache=True, parallel=True)
def _boxes_of_cells(
    vertices: np.ndarray,
    shape: np.ndarray,
    shape_block: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    num_chunk: int,
) -> None:
    """
    The bounding boxes of the blocks of level one, each of two cells along
    every axis, taken from their vertices.

    A vertex whose coordinate is NaN is left out of the box.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    shape
        The number of vertices along each axis.
    shape_block
        The number of blocks along each axis.
    lower
        Where to put the lower corner of the box of each block, of shape
        ``(number of blocks, N)``.
    upper
        Where to put the upper corner of the box of each block.
    num_chunk
        The number of pieces to divide the work into.
    """
    num_axis = shape.shape[0]
    stride = _cells.strides(shape)
    num_block = lower.shape[0]

    for chunk in numba.prange(num_chunk):
        block = np.empty(num_axis, dtype=np.int64)
        first = np.empty(num_axis, dtype=np.int64)
        last = np.empty(num_axis, dtype=np.int64)
        vertex = np.empty(num_axis, dtype=np.int64)
        for k in range(chunk, num_block, num_chunk):
            _unravel(k, shape_block, block)
            for m in range(num_axis):
                first[m] = 2 * block[m]
                last[m] = min(2 * block[m] + 2, shape[m] - 1)
                vertex[m] = first[m]
            for c in range(num_axis):
                lower[k, c] = np.inf
                upper[k, c] = -np.inf
            while True:
                index = 0
                for m in range(num_axis):
                    index += vertex[m] * stride[m]
                for c in range(num_axis):
                    value = vertices[c, index]
                    if value < lower[k, c]:
                        lower[k, c] = value
                    if value > upper[k, c]:
                        upper[k, c] = value
                m = num_axis - 1
                while m >= 0:
                    vertex[m] += 1
                    if vertex[m] <= last[m]:
                        break
                    vertex[m] = first[m]
                    m -= 1
                if m < 0:
                    break


@numba.njit(cache=True, parallel=True)
def _boxes_of_boxes(
    lower_below: np.ndarray,
    upper_below: np.ndarray,
    shape_below: np.ndarray,
    shape_block: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    num_chunk: int,
) -> None:
    """
    The bounding boxes of the blocks of a level above level one, each of up
    to two blocks of the level below along every axis.

    Parameters
    ----------
    lower_below
        The lower corner of the box of each block of the level below.
    upper_below
        The upper corner of the box of each block of the level below.
    shape_below
        The number of blocks along each axis of the level below.
    shape_block
        The number of blocks along each axis of this level.
    lower
        Where to put the lower corner of the box of each block of this level.
    upper
        Where to put the upper corner of the box of each block of this level.
    num_chunk
        The number of pieces to divide the work into.
    """
    num_axis = shape_block.shape[0]
    num_block = lower.shape[0]

    for chunk in numba.prange(num_chunk):
        block = np.empty(num_axis, dtype=np.int64)
        child = np.empty(num_axis, dtype=np.int64)
        for k in range(chunk, num_block, num_chunk):
            _unravel(k, shape_block, block)
            for c in range(num_axis):
                lower[k, c] = np.inf
                upper[k, c] = -np.inf
            for b in range(1 << num_axis):
                valid = True
                for m in range(num_axis):
                    child[m] = 2 * block[m] + ((b >> m) & 1)
                    if child[m] >= shape_below[m]:
                        valid = False
                if not valid:
                    continue
                j = _ravel(child, shape_below)
                for c in range(num_axis):
                    if lower_below[j, c] < lower[k, c]:
                        lower[k, c] = lower_below[j, c]
                    if upper_below[j, c] > upper[k, c]:
                        upper[k, c] = upper_below[j, c]


def tree(
    vertices: np.ndarray,
    shape: np.ndarray,
    num_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the hierarchy of bounding boxes of a grid.

    Returns ``(lower, upper, start, shapes)``.  ``shapes[level]`` is the
    number of blocks along each axis of each level, starting with the cells
    at level zero.  The box of block ``k`` of a level above zero has the
    lower corner ``lower[start[level] + k]`` and the upper corner
    ``upper[start[level] + k]``.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``, in row-major order.
    shape
        The number of vertices along each axis.
    num_chunk
        The number of pieces to divide the work into.
    """
    num_axis = shape.size

    shapes = [shape - 1]
    while np.any(shapes[~0] > 1):
        shapes.append((shapes[~0] + 1) // 2)
    shapes = np.array(shapes, dtype=np.int64)
    num_level = shapes.shape[0] - 1

    start = np.zeros(num_level + 2, dtype=np.int64)
    start[2:] = np.cumsum(np.prod(shapes[1:], axis=~0))

    lower = np.empty((start[~0], num_axis), dtype=np.float64)
    upper = np.empty((start[~0], num_axis), dtype=np.float64)

    for level in range(1, num_level + 1):
        a, b = start[level], start[level + 1]
        if level == 1:
            _boxes_of_cells(
                vertices,
                shape,
                shapes[level],
                lower[a:b],
                upper[a:b],
                num_chunk,
            )
        else:
            a_below, b_below = start[level - 1], start[level]
            _boxes_of_boxes(
                lower[a_below:b_below],
                upper[a_below:b_below],
                shapes[level - 1],
                shapes[level],
                lower[a:b],
                upper[a:b],
                num_chunk,
            )

    return lower, upper, start, shapes


@numba.njit(cache=True)
def _in_box(
    lower: np.ndarray,
    upper: np.ndarray,
    index: int,
    point: np.ndarray,
) -> bool:
    """
    Whether a point is inside a bounding box, or on its boundary.

    Parameters
    ----------
    lower
        The lower corners of the boxes.
    upper
        The upper corners of the boxes.
    index
        Which box.
    point
        The point.
    """
    for c in range(point.shape[0]):
        if not (lower[index, c] <= point[c] <= upper[index, c]):
            return False
    return True


@numba.njit(cache=True)
def _in_cell(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    point: np.ndarray,
    xi: np.ndarray,
    work: np.ndarray,
) -> bool:
    """
    Whether a point is inside a cell, and if it is, its local coordinates.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first.
    point
        The point.
    xi
        Where to put the local coordinates of the point.
    work
        A workspace for :func:`~._cells.inverse`.
    """
    num_axis = point.shape[0]

    for c in range(num_axis):
        low = np.inf
        high = -np.inf
        for b in range(offset.shape[0]):
            value = vertices[c, base + offset[b]]
            if value < low:
                low = value
            if value > high:
                high = value
        if not (low <= point[c] <= high):
            return False

    xi[:] = 0.5
    if not _cells.inverse(vertices, base, offset, point, xi, work):
        return False

    for m in range(num_axis):
        if not (-tolerance <= xi[m] <= 1 + tolerance):
            return False

    return True


@numba.njit(cache=True)
def _search(
    vertices: np.ndarray,
    stride: np.ndarray,
    offset: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    shapes: np.ndarray,
    point: np.ndarray,
    stack_level: np.ndarray,
    stack_index: np.ndarray,
    block: np.ndarray,
    cell: np.ndarray,
    xi: np.ndarray,
    work: np.ndarray,
) -> bool:
    """
    Find the cell which contains a point by descending the hierarchy of
    bounding boxes.

    Returns whether the point is in a cell.  If it is, the first corner of
    the cell is left in `cell` and the local coordinates of the point in
    `xi`.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    stride
        The number of vertices between neighbors along each axis.
    offset
        The offset of each corner of a cell from the first.
    lower
        The lower corners of the boxes, from :func:`tree`.
    upper
        The upper corners of the boxes.
    start
        Where the boxes of each level start.
    shapes
        The number of blocks along each axis of each level.
    point
        The point to find.
    stack_level
        Room for the level of each block waiting to be searched.
    stack_index
        Room for the index of each block waiting to be searched.
    block
        Room for the index along each axis of a block.
    cell
        Where to put the first corner of the cell.
    xi
        Where to put the local coordinates of the point.
    work
        A workspace for :func:`~._cells.inverse`.
    """
    num_axis = point.shape[0]
    num_level = shapes.shape[0] - 1
    num_corner = offset.shape[0]

    if num_level == 0:
        cell[:] = 0
        return _in_cell(vertices, 0, offset, point, xi, work)

    if not _in_box(lower, upper, int(start[num_level]), point):
        return False

    stack_level[0] = num_level
    stack_index[0] = 0
    size = 1

    while size > 0:
        size -= 1
        level = stack_level[size]
        _unravel(stack_index[size], shapes[level], block)
        below = shapes[level - 1]

        if level == 1:
            # the children are cells, which are tried in order
            for b in range(num_corner):
                valid = True
                base = 0
                for m in range(num_axis):
                    cell[m] = 2 * block[m] + ((b >> m) & 1)
                    if cell[m] >= below[m]:
                        valid = False
                    base += cell[m] * stride[m]
                if valid and _in_cell(vertices, base, offset, point, xi, work):
                    return True

        else:
            # the children are blocks, which are pushed in reverse so that
            # they come off the stack in order
            for b in range(num_corner - 1, -1, -1):
                valid = True
                for m in range(num_axis):
                    cell[m] = 2 * block[m] + ((b >> m) & 1)
                    if cell[m] >= below[m]:
                        valid = False
                if not valid:
                    continue
                index = _ravel(cell, below)
                if _in_box(lower, upper, int(start[level - 1]) + index, point):
                    stack_level[size] = level - 1
                    stack_index[size] = index
                    size += 1

    return False


@numba.njit(cache=True)
def _distance(
    low: float,
    high: float,
    value: float,
) -> float:
    """
    The distance from a value to an interval, or zero if it is inside.

    Parameters
    ----------
    low
        The lower end of the interval.
    high
        The upper end of the interval.
    value
        The value.
    """
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


@numba.njit(cache=True)
def _distance_box(
    lower: np.ndarray,
    upper: np.ndarray,
    index: int,
    point: np.ndarray,
) -> float:
    """
    The square of the distance from a point to a bounding box.

    Parameters
    ----------
    lower
        The lower corners of the boxes.
    upper
        The upper corners of the boxes.
    index
        Which box.
    point
        The point.
    """
    result = 0.0
    for c in range(point.shape[0]):
        result += _distance(lower[index, c], upper[index, c], point[c]) ** 2
    return result


@numba.njit(cache=True)
def _distance_cell(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    point: np.ndarray,
) -> float:
    """
    The square of the distance from a point to the bounding box of a cell.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first.
    point
        The point.
    """
    result = 0.0
    for c in range(point.shape[0]):
        low = np.inf
        high = -np.inf
        for b in range(offset.shape[0]):
            value = vertices[c, base + offset[b]]
            if value < low:
                low = value
            if value > high:
                high = value
        result += _distance(low, high, point[c]) ** 2
    return result


@numba.njit(cache=True)
def _nearest(
    vertices: np.ndarray,
    stride: np.ndarray,
    offset: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    shapes: np.ndarray,
    point: np.ndarray,
    stack_level: np.ndarray,
    stack_index: np.ndarray,
    block: np.ndarray,
    child_index: np.ndarray,
    child_distance: np.ndarray,
    cell: np.ndarray,
    nearest: np.ndarray,
) -> None:
    """
    Find the cell whose bounding box is nearest to a point.

    This is a branch-and-bound search of the hierarchy of bounding boxes:
    a block is skipped if its box is no nearer than the nearest cell found
    so far, and the children of a block are searched nearest first, so that
    a near cell is found early.  The first corner of the cell is left in
    `nearest`.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    stride
        The number of vertices between neighbors along each axis.
    offset
        The offset of each corner of a cell from the first.
    lower
        The lower corners of the boxes, from :func:`tree`.
    upper
        The upper corners of the boxes.
    start
        Where the boxes of each level start.
    shapes
        The number of blocks along each axis of each level.
    point
        The point.
    stack_level
        Room for the level of each block waiting to be searched.
    stack_index
        Room for the index of each block waiting to be searched.
    block
        Room for the index along each axis of a block.
    child_index
        Room for the index of each child of a block.
    child_distance
        Room for the distance to each child of a block.
    cell
        Room for the index along each axis of a cell.
    nearest
        Where to put the first corner of the nearest cell.
    """
    num_axis = point.shape[0]
    num_level = shapes.shape[0] - 1
    num_corner = offset.shape[0]

    nearest[:] = 0
    if num_level == 0:
        return

    best = np.inf
    stack_level[0] = num_level
    stack_index[0] = 0
    size = 1

    while size > 0:
        size -= 1
        level = stack_level[size]
        index = stack_index[size]
        if not _distance_box(lower, upper, int(start[level]) + index, point) < best:
            continue
        _unravel(index, shapes[level], block)
        below = shapes[level - 1]

        if level == 1:
            for b in range(num_corner):
                valid = True
                base = 0
                for m in range(num_axis):
                    cell[m] = 2 * block[m] + ((b >> m) & 1)
                    if cell[m] >= below[m]:
                        valid = False
                    base += cell[m] * stride[m]
                if not valid:
                    continue
                distance = _distance_cell(vertices, base, offset, point)
                if distance < best:
                    best = distance
                    nearest[:] = cell

        else:
            num_child = 0
            for b in range(num_corner):
                valid = True
                for m in range(num_axis):
                    cell[m] = 2 * block[m] + ((b >> m) & 1)
                    if cell[m] >= below[m]:
                        valid = False
                if not valid:
                    continue
                child = _ravel(cell, below)
                distance = _distance_box(
                    lower, upper, int(start[level - 1]) + child, point
                )
                if not distance < best:
                    continue
                # insert in order of decreasing distance, so that the
                # nearest child is pushed last and comes off the stack first
                j = num_child
                while j > 0 and child_distance[j - 1] < distance:
                    child_distance[j] = child_distance[j - 1]
                    child_index[j] = child_index[j - 1]
                    j -= 1
                child_distance[j] = distance
                child_index[j] = child
                num_child += 1
            for j in range(num_child):
                stack_level[size] = level - 1
                stack_index[size] = child_index[j]
                size += 1


@numba.njit(cache=True)
def _clamp(
    s: np.ndarray,
    shape: np.ndarray,
    stride: np.ndarray,
    cell: np.ndarray,
    xi: np.ndarray,
) -> int:
    """
    Split a fractional index into the cell nearest to it and its local
    coordinates in that cell, which are outside the unit cube if the index
    is outside the grid.

    Returns the flat index of the first corner of the cell.

    Parameters
    ----------
    s
        The fractional index of a point, along each axis of the vertices.
    shape
        The number of vertices along each axis.
    stride
        The number of vertices between neighbors along each axis.
    cell
        Where to put the first corner of the cell.
    xi
        Where to put the local coordinates.
    """
    base = 0
    for m in range(s.shape[0]):
        value = s[m]
        if not value >= 0:
            value = 0.0
        elif value > shape[m] - 2:
            value = shape[m] - 2.0
        cell[m] = int(np.floor(value))
        xi[m] = s[m] - cell[m]
        base += cell[m] * stride[m]
    return base


@numba.njit(cache=True)
def _walk(
    vertices: np.ndarray,
    shape: np.ndarray,
    stride: np.ndarray,
    offset: np.ndarray,
    point: np.ndarray,
    s: np.ndarray,
    s_trial: np.ndarray,
    cell: np.ndarray,
    cell_trial: np.ndarray,
    xi: np.ndarray,
    xi_trial: np.ndarray,
    work: np.ndarray,
) -> bool:
    """
    Extrapolate the grid to a point outside of it.

    This runs Newton's method across the whole grid in index space, from the
    fractional index in `s`, continuing the cells along the edges of the grid
    beyond it.  Each step is halved until it brings the position closer to
    the point.  The first corner of the cell it stops in is left in `cell`,
    and the local coordinates of the point in that cell in `xi`.

    Returns whether it reached the point.  It does not if the continuation
    of the grid folds over before reaching it, as it can beyond a strongly
    curved edge, in which case it stops where the continuation came closest.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    shape
        The number of vertices along each axis.
    stride
        The number of vertices between neighbors along each axis.
    offset
        The offset of each corner of a cell from the first.
    point
        The point.
    s
        The fractional index to start from, which is replaced by that of the
        point.
    s_trial
        Room for a trial fractional index.
    cell
        Where to put the first corner of the cell.
    cell_trial
        Room for the first corner of a trial cell.
    xi
        Where to put the local coordinates of the point in the cell.
    xi_trial
        Room for the local coordinates in a trial cell.
    work
        A workspace of the same shape as one for :func:`~._cells.inverse`.
    """
    num_axis = point.shape[0]

    jacobian = work[:num_axis]
    matrix = work[num_axis : 2 * num_axis]
    x = work[2 * num_axis]
    residual = work[2 * num_axis + 1]
    step = work[2 * num_axis + 2]
    x_trial = work[2 * num_axis + 3]
    goal = work[2 * num_axis + 4]
    goal_trial = work[2 * num_axis + 5]

    base = _clamp(s, shape, stride, cell, xi)
    _cells.target(vertices, base, point, goal)
    _cells.evaluate(vertices, base, offset, xi, x, jacobian)
    error = _cells.distance_squared(x, goal)

    reached = False

    for _ in range(num_iteration_walk):

        if not error > 0:
            reached = error == 0
            break

        matrix[:, :] = jacobian
        for k in range(num_axis):
            residual[k] = x[k] - goal[k]
        if not _cells.solve(matrix, residual, step):
            break

        size = 0.0
        scale = 1.0
        for m in range(num_axis):
            size = max(size, abs(step[m]))
            scale = max(scale, abs(s[m]))
        if not size < np.inf:
            break

        fraction = 1.0
        accepted = False
        for _ in range(_cells.num_halving):
            for m in range(num_axis):
                s_trial[m] = s[m] - fraction * step[m]
            base = _clamp(s_trial, shape, stride, cell_trial, xi_trial)
            _cells.target(vertices, base, point, goal_trial)
            _cells.position(vertices, base, offset, xi_trial, x_trial)
            error_trial = _cells.distance_squared(x_trial, goal_trial)
            if error_trial < error:
                accepted = True
                break
            fraction = fraction / 2

        if not accepted:
            # no point along the step is closer, which happens once the step
            # is down to the rounding error of the position
            reached = size <= _cells.tolerance * scale
            break

        s[:] = s_trial

        if fraction * size <= _cells.tolerance * scale:
            reached = True
            break

        base = _clamp(s, shape, stride, cell, xi)
        _cells.target(vertices, base, point, goal)
        _cells.evaluate(vertices, base, offset, xi, x, jacobian)
        error = _cells.distance_squared(x, goal)

    _clamp(s, shape, stride, cell, xi)

    return reached


@numba.njit(cache=True, parallel=True)
def locate(
    vertices: np.ndarray,
    shape: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    shapes: np.ndarray,
    points: np.ndarray,
    extrapolate: bool,
    num_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the cell of a curvilinear grid which contains each point.

    Returns ``(cell, xi, inside)``: the first corner of the cell along each
    axis, the local coordinates of the point in the cell, and whether the
    point is inside the grid.  A point outside the grid, or with a
    coordinate which is not finite, is given the first cell and local
    coordinates of NaN, unless `extrapolate` is set and its coordinates are
    finite.  An extrapolated point which the continuation of the grid does
    not reach keeps the cell it was extrapolated from, with local
    coordinates of NaN.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    shape
        The number of vertices along each axis.
    lower
        The lower corners of the boxes, from :func:`tree`.
    upper
        The upper corners of the boxes.
    start
        Where the boxes of each level start.
    shapes
        The number of blocks along each axis of each level.
    points
        The points to find, of shape ``(N, number of points)``.
    extrapolate
        Whether to find the cell to extrapolate from for a point outside the
        grid.
    num_chunk
        The number of pieces to divide the work into.
    """
    num_axis, num_point = points.shape
    num_level = shapes.shape[0] - 1

    stride = _cells.strides(shape)
    offset = _cells.offsets(shape)
    num_corner = offset.shape[0]
    capacity = num_level * num_corner + 1

    cell = np.zeros((num_point, num_axis), dtype=np.int64)
    xi = np.empty((num_point, num_axis), dtype=np.float64)
    inside = np.zeros(num_point, dtype=np.bool_)

    for chunk in numba.prange(num_chunk):
        stack_level = np.empty(capacity, dtype=np.int64)
        stack_index = np.empty(capacity, dtype=np.int64)
        block = np.empty(num_axis, dtype=np.int64)
        cell_i = np.empty(num_axis, dtype=np.int64)
        cell_trial = np.empty(num_axis, dtype=np.int64)
        point = np.empty(num_axis, dtype=np.float64)
        xi_i = np.empty(num_axis, dtype=np.float64)
        xi_trial = np.empty(num_axis, dtype=np.float64)
        s = np.empty(num_axis, dtype=np.float64)
        s_trial = np.empty(num_axis, dtype=np.float64)
        child_index = np.empty(num_corner, dtype=np.int64)
        child_distance = np.empty(num_corner, dtype=np.float64)
        work = np.empty((2 * num_axis + _cells.num_work, num_axis))

        for i in range(chunk, num_point, num_chunk):
            finite = True
            for c in range(num_axis):
                point[c] = points[c, i]
                if not np.isfinite(point[c]):
                    finite = False

            found = False
            if finite:
                found = _search(
                    vertices,
                    stride,
                    offset,
                    lower,
                    upper,
                    start,
                    shapes,
                    point,
                    stack_level,
                    stack_index,
                    block,
                    cell_i,
                    xi_i,
                    work,
                )

            if found:
                inside[i] = True
            elif finite and extrapolate:
                _nearest(
                    vertices,
                    stride,
                    offset,
                    lower,
                    upper,
                    start,
                    shapes,
                    point,
                    stack_level,
                    stack_index,
                    block,
                    child_index,
                    child_distance,
                    cell_trial,
                    cell_i,
                )
                for m in range(num_axis):
                    s[m] = cell_i[m] + 0.5
                reached = _walk(
                    vertices,
                    shape,
                    stride,
                    offset,
                    point,
                    s,
                    s_trial,
                    cell_i,
                    cell_trial,
                    xi_i,
                    xi_trial,
                    work,
                )
                if not reached:
                    xi_i[:] = np.nan
            else:
                cell_i[:] = 0
                xi_i[:] = np.nan

            for m in range(num_axis):
                cell[i, m] = cell_i[m]
                xi[i, m] = xi_i[m]

    return cell, xi, inside

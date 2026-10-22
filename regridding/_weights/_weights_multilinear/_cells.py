r"""
The multilinear map from the unit cube onto a cell of a logically-rectangular
grid, in any number of dimensions.

A grid of :math:`N` dimensions has :math:`N` coordinates, each sampled on the
same array of vertices.  Here the vertices are stored flat, in row-major
order, as an array of shape ``(N, number of vertices)``, so that one compiled
function serves any number of dimensions.

A cell is addressed by its first corner, the vertex with the smallest index
along every axis.  Its :math:`2^N` corners are numbered so that bit :math:`m`
of a corner's number says whether that corner is one vertex further along
axis :math:`m`.  A point of the cell has the local coordinates
:math:`\xi \in [0, 1]^N`, and corner :math:`b` contributes to it with the
weight

.. math::

    w_b(\xi) = \prod_{m=0}^{N-1}
    \begin{cases}
        \xi_m & \text{if bit } m \text{ of } b \text{ is set,} \\
        1 - \xi_m & \text{otherwise,}
    \end{cases}

so that the position of the point is :math:`x(\xi) = \sum_b w_b(\xi) x_b`,
where :math:`x_b` is the position of corner :math:`b`.  The same weights
interpolate any array sampled on the vertices, which is what
:func:`regridding.weights` saves.

Positions are measured from the first corner of the cell,
:math:`x(\xi) - x_0 = \sum_b w_b(\xi) (x_b - x_0)`, since the weights sum to
one.  Their rounding error is then a fraction of the size of the cell, rather
than of the distance of the cell from the origin, so that the local
coordinates of a point can be found to the same precision wherever the grid
is.

Outside the unit cube, the same expressions continue the cell, which is how
a point outside the grid is extrapolated.
"""

import numpy as np
import numba

__all__ = [
    "num_work",
    "strides",
    "offsets",
    "evaluate",
    "position",
    "target",
    "distance_squared",
    "solve",
    "newton",
    "weights",
]

num_work = 7
"""
The number of rows of a workspace for :func:`newton`, beyond the
:math:`2N` rows of the two matrices.

A workspace has the shape ``(2 * N + num_work, N)``.
"""

tolerance = 1e-13
"""
The size of a step, in units of the local coordinates, below which
Newton's method has converged.
"""

rounding = 64 * np.finfo(np.float64).eps
"""
The rounding error of a position, for each corner of a cell, as a fraction
of the largest of the terms it is the sum of.

See :func:`newton`.
"""

num_iteration = 64
"""The most steps Newton's method takes before it gives up."""

num_halving = 30
"""The most times a step of Newton's method is halved before it gives up."""


@numba.njit(cache=True)
def strides(
    shape: np.ndarray,
) -> np.ndarray:
    """
    The number of vertices between neighbors along each axis of a grid
    stored in row-major order.

    Parameters
    ----------
    shape
        The number of vertices along each axis.
    """
    num_axis = shape.shape[0]
    result = np.empty(num_axis, dtype=np.int64)
    stride = 1
    for m in range(num_axis - 1, -1, -1):
        result[m] = stride
        stride = stride * shape[m]
    return result


@numba.njit(cache=True)
def offsets(
    shape: np.ndarray,
) -> np.ndarray:
    """
    The offset of each corner of a cell from its first corner, in the flat
    index of the vertices.

    Parameters
    ----------
    shape
        The number of vertices along each axis.
    """
    num_axis = shape.shape[0]
    stride = strides(shape)
    num_corner = 1 << num_axis
    result = np.zeros(num_corner, dtype=np.int64)
    for b in range(num_corner):
        for m in range(num_axis):
            if (b >> m) & 1:
                result[b] += stride[m]
    return result


@numba.njit(cache=True)
def evaluate(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    xi: np.ndarray,
    x: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    The position of a point of a cell, measured from the first corner of the
    cell, and its derivative with respect to the local coordinates.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first, from :func:`offsets`.
    xi
        The local coordinates of the point.
    x
        Where to put the position of the point, measured from the first
        corner of the cell.
    jacobian
        Where to put the derivative, an :math:`N \times N` array whose element
        ``[k, m]`` is the derivative of coordinate :math:`k` with respect to
        :math:`\xi_m`.
    """
    num_axis = xi.shape[0]
    x[:] = 0
    jacobian[:, :] = 0
    for b in range(offset.shape[0]):
        index = base + offset[b]
        weight = 1.0
        for m in range(num_axis):
            if (b >> m) & 1:
                weight *= xi[m]
            else:
                weight *= 1 - xi[m]
        for k in range(num_axis):
            x[k] += weight * (vertices[k, index] - vertices[k, base])
        for m in range(num_axis):
            # the product of every factor but the one along axis m, which is
            # computed directly rather than by dividing, since a factor can
            # be zero
            derivative = 1.0 if (b >> m) & 1 else -1.0
            for j in range(num_axis):
                if j != m:
                    if (b >> j) & 1:
                        derivative *= xi[j]
                    else:
                        derivative *= 1 - xi[j]
            for k in range(num_axis):
                jacobian[k, m] += derivative * (vertices[k, index] - vertices[k, base])


@numba.njit(cache=True)
def position(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    xi: np.ndarray,
    x: np.ndarray,
) -> None:
    """
    The position of a point of a cell, measured from the first corner of the
    cell.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first, from :func:`offsets`.
    xi
        The local coordinates of the point.
    x
        Where to put the position of the point, measured from the first
        corner of the cell.
    """
    num_axis = xi.shape[0]
    x[:] = 0
    for b in range(offset.shape[0]):
        index = base + offset[b]
        weight = 1.0
        for m in range(num_axis):
            if (b >> m) & 1:
                weight *= xi[m]
            else:
                weight *= 1 - xi[m]
        for k in range(num_axis):
            x[k] += weight * (vertices[k, index] - vertices[k, base])


@numba.njit(cache=True)
def solve(
    a: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
) -> bool:
    """
    Solve the square system ``a @ out = b`` by Gaussian elimination with
    partial pivoting.

    Both `a` and `b` are overwritten.  Returns whether the system could be
    solved, which it cannot if `a` is singular.

    Parameters
    ----------
    a
        The matrix of the system.
    b
        The right-hand side of the system.
    out
        Where to put the solution.
    """
    num = b.shape[0]
    for column in range(num):
        pivot = column
        largest = abs(a[column, column])
        for row in range(column + 1, num):
            if abs(a[row, column]) > largest:
                largest = abs(a[row, column])
                pivot = row
        if not largest > 0:
            return False
        if pivot != column:
            for j in range(num):
                a[column, j], a[pivot, j] = a[pivot, j], a[column, j]
            b[column], b[pivot] = b[pivot], b[column]
        for row in range(column + 1, num):
            factor = a[row, column] / a[column, column]
            for j in range(column, num):
                a[row, j] -= factor * a[column, j]
            b[row] -= factor * b[column]
    for row in range(num - 1, -1, -1):
        total = b[row]
        for j in range(row + 1, num):
            total -= a[row, j] * out[j]
        out[row] = total / a[row, row]
    return True


@numba.njit(cache=True)
def target(
    vertices: np.ndarray,
    base: int,
    point: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    The position of a point measured from the first corner of a cell, as
    :func:`evaluate` and :func:`position` measure it.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    base
        The flat index of the first corner of the cell.
    point
        The point.
    out
        Where to put the result.
    """
    for k in range(point.shape[0]):
        out[k] = point[k] - vertices[k, base]


@numba.njit(cache=True)
def distance_squared(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    The square of the distance between two points.

    Parameters
    ----------
    x
        The first point.
    y
        The second point.
    """
    result = 0.0
    for k in range(x.shape[0]):
        result += (x[k] - y[k]) ** 2
    return result


@numba.njit(cache=True)
def _move(
    shape: np.ndarray,
    stride: np.ndarray,
    clamp: bool,
    cell: np.ndarray,
    xi: np.ndarray,
) -> int:
    """
    Move a point given by a cell and its local coordinates in that cell to
    the cell its local coordinates fall in, if `clamp` is set.

    The cell is clamped to the grid, so a point beyond the grid stays in the
    cell along its edge, with local coordinates outside the unit cube.  The
    local coordinates shift by whole cells, which is exact.  Returns the flat
    index of the first corner of the cell, moved or not.

    Parameters
    ----------
    shape
        The number of vertices along each axis.
    stride
        The number of vertices between neighbors along each axis.
    clamp
        Whether to move the point to the cell its local coordinates fall in.
    cell
        The first corner of the cell, which is replaced by that of the cell
        the point is moved to.
    xi
        The local coordinates of the point, which are replaced by those in
        the cell it is moved to.
    """
    base = 0
    for m in range(cell.shape[0]):
        if clamp:
            shift = np.floor(xi[m])
            low = -float(cell[m])
            high = float(shape[m] - 2 - cell[m])
            if shift != shift:
                shift = 0.0
            elif shift < low:
                shift = low
            elif shift > high:
                shift = high
            cell[m] += int(shift)
            xi[m] -= shift
        base += cell[m] * stride[m]
    return base


@numba.njit(cache=True)
def _floor(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    xi: np.ndarray,
    goal: np.ndarray,
) -> float:
    """
    The square of the distance between the position of a point of a cell
    and the point it should be at, below which the difference is no more
    than rounding error.

    Each coordinate of the position, measured from the first corner of the
    cell, is a sum over the corners of their weights times their distances
    from the first corner, and the point is measured from the first corner
    too.  Each rounds to a multiple of :data:`rounding` of the largest of
    its terms: the sum of the magnitudes of the weights times the extent of
    the cell along that coordinate, and the distance of the point from the
    first corner.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first.
    xi
        The local coordinates of the point.
    goal
        The point, measured from the first corner of the cell.
    """
    num_axis = xi.shape[0]
    total = 1.0
    for m in range(num_axis):
        total *= abs(xi[m]) + abs(1 - xi[m])
    factor = (1 << num_axis) * rounding
    result = 0.0
    for k in range(num_axis):
        extent = 0.0
        for b in range(offset.shape[0]):
            extent = max(extent, abs(vertices[k, base + offset[b]] - vertices[k, base]))
        error = factor * (total * extent + abs(goal[k]))
        result += error * error
    return result


@numba.njit(cache=True)
def newton(
    vertices: np.ndarray,
    shape: np.ndarray,
    stride: np.ndarray,
    offset: np.ndarray,
    point: np.ndarray,
    clamp: bool,
    cell: np.ndarray,
    xi: np.ndarray,
    cell_trial: np.ndarray,
    work: np.ndarray,
) -> bool:
    """
    Find the local coordinates of a point by Newton's method.

    The iterate is a cell and local coordinates in it, starting from those
    given.  Without `clamp`, it stays in its cell, whose continuation is
    searched beyond it, so the local coordinates come out outside the unit
    cube if the point is outside the cell.  With `clamp`, it moves after
    every step to the cell its local coordinates fall in, clamped to the
    grid, so it can cross the grid and continue past its edges, which is how
    a point outside the grid is extrapolated.

    Each step is halved until it brings the position closer to the point, so
    that the method does not run away on a strongly curved cell.  It has
    converged once a step is shorter than :data:`tolerance`, in units of the
    local coordinates, or once the distance from the position to the point
    is down to the rounding error of the position (see :func:`_floor`).  The
    second is needed for a cell much longer than it is wide, if it is not
    aligned with the axes: rounding along its length moves the local
    coordinate across its width by much more than the first allows, and no
    step then brings the position any closer.  Neither depends on the units,
    the size, or the position of the grid.

    Returns whether it converged.  The cell and local coordinates are left
    wherever it stopped.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    shape
        The number of vertices along each axis.
    stride
        The number of vertices between neighbors along each axis.
    offset
        The offset of each corner from the first, from :func:`offsets`.
    point
        The point to find.
    clamp
        Whether the iterate moves between cells.
    cell
        The first corner of the cell to start in, which is replaced by that
        of the cell it stops in.
    xi
        The local coordinates to start from, which are replaced by those
        where it stops.
    cell_trial
        Room for the first corner of the cell of a trial step.
    work
        A workspace of shape ``(2 * N + num_work, N)``.
    """
    num_axis = xi.shape[0]

    jacobian = work[:num_axis]
    matrix = work[num_axis : 2 * num_axis]
    x = work[2 * num_axis]
    residual = work[2 * num_axis + 1]
    step = work[2 * num_axis + 2]
    goal = work[2 * num_axis + 3]
    xi_trial = work[2 * num_axis + 4]
    x_trial = work[2 * num_axis + 5]
    goal_trial = work[2 * num_axis + 6]

    base = _move(shape, stride, clamp, cell, xi)
    target(vertices, base, point, goal)
    evaluate(vertices, base, offset, xi, x, jacobian)
    error = distance_squared(x, goal)

    for _ in range(num_iteration):

        if error <= _floor(vertices, base, offset, xi, goal):
            return True

        matrix[:, :] = jacobian
        for k in range(num_axis):
            residual[k] = x[k] - goal[k]
        if not solve(matrix, residual, step):
            return False

        size = 0.0
        scale = 1.0
        for m in range(num_axis):
            size = max(size, abs(step[m]))
            scale = max(scale, abs(xi[m]))
        if not size < np.inf:
            return False

        fraction = 1.0
        accepted = False
        base_trial = base
        for _ in range(num_halving):
            cell_trial[:] = cell
            for m in range(num_axis):
                xi_trial[m] = xi[m] - fraction * step[m]
            base_trial = _move(shape, stride, clamp, cell_trial, xi_trial)
            target(vertices, base_trial, point, goal_trial)
            position(vertices, base_trial, offset, xi_trial, x_trial)
            error_trial = distance_squared(x_trial, goal_trial)
            if error_trial < error:
                accepted = True
                break
            fraction = fraction / 2

        if not accepted:
            # no point along the step is closer, which happens once the step
            # is down to the rounding error of the position
            return size <= tolerance * scale

        cell[:] = cell_trial
        xi[:] = xi_trial
        base = base_trial

        if fraction * size <= tolerance * scale:
            return True

        target(vertices, base, point, goal)
        evaluate(vertices, base, offset, xi, x, jacobian)
        error = distance_squared(x, goal)

    return False  # pragma: nocover


@numba.njit(cache=True, parallel=True)
def weights(
    cell: np.ndarray,
    xi: np.ndarray,
    inside: np.ndarray,
    shape: np.ndarray,
    weights_input: np.ndarray,
    nan_outside: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The weight of each corner of the cell containing each point, for each
    of a number of elements.

    Returns the arrays ``(indices_input, indices_output, values)``, each with
    a row for each element, which holds :math:`2^N` consecutive weights for
    each point, one for each corner of its cell, in the order the corners
    are numbered.

    Parameters
    ----------
    cell
        The first corner of the cell of each point, of shape
        ``(number of elements, number of points, N)``.
    xi
        The local coordinates of each point in its cell, of the same shape.
    inside
        Whether each point is inside the grid, of shape
        ``(number of elements, number of points)``.
    shape
        The number of vertices along each axis.
    weights_input
        Weights to multiply the weight of each vertex by, flattened, of shape
        ``(number of elements, number of vertices)``, or with a single row
        which every element shares, or with none for no weights.
    nan_outside
        Whether to set the weights of a point outside the grid to NaN rather
        than extrapolate.
    """
    num_element, num_point, num_axis = cell.shape
    stride = strides(shape)
    offset = offsets(shape)
    num_corner = offset.shape[0]
    num_weighted = weights_input.shape[0]

    size = num_point * num_corner
    indices_input = np.empty((num_element, size), dtype=np.int64)
    indices_output = np.empty((num_element, size), dtype=np.int64)
    values = np.empty((num_element, size), dtype=np.float64)

    for t in numba.prange(num_element * num_point):
        e = t // num_point
        i = t - e * num_point
        base = 0
        for m in range(num_axis):
            base += cell[e, i, m] * stride[m]
        for b in range(num_corner):
            index = base + offset[b]
            weight = 1.0
            for m in range(num_axis):
                if (b >> m) & 1:
                    weight *= xi[e, i, m]
                else:
                    weight *= 1 - xi[e, i, m]
            if num_weighted > 0:
                weight *= weights_input[min(e, num_weighted - 1), index]
            if nan_outside and not inside[e, i]:
                weight = np.nan
            w = i * num_corner + b
            indices_input[e, w] = index
            indices_output[e, w] = i
            values[e, w] = weight

    return indices_input, indices_output, values

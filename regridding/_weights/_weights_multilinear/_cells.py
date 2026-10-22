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
    "inverse",
    "weights",
]

num_work = 6
"""
The number of rows of a workspace for :func:`inverse`, beyond the
:math:`2N` rows of the two matrices.

A workspace has the shape ``(2 * N + num_work, N)``.
"""

tolerance = 1e-13
"""
The size of the last step, in units of the local coordinates, below which
Newton's method has converged.
"""

num_iteration = 32
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
                jacobian[k, m] += derivative * (
                    vertices[k, index] - vertices[k, base]
                )


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
def inverse(
    vertices: np.ndarray,
    base: int,
    offset: np.ndarray,
    point: np.ndarray,
    xi: np.ndarray,
    work: np.ndarray,
) -> bool:
    """
    Find the local coordinates of a point in a cell, which may be outside
    the unit cube if the point is outside the cell.

    This uses Newton's method, starting from the value of `xi` it is given.
    Each step is halved until it brings the cell's position closer to the
    point, so that the method does not run away on a strongly curved cell.
    It has converged once a step is shorter than :data:`tolerance`, which is
    in units of the local coordinates and so does not depend on the units
    or the size of the grid.

    Returns whether it converged.  If it did not, `xi` is left wherever the
    method stopped.

    Parameters
    ----------
    vertices
        The coordinates of every vertex of the grid, of shape
        ``(N, number of vertices)``.
    base
        The flat index of the first corner of the cell.
    offset
        The offset of each corner from the first, from :func:`offsets`.
    point
        The point to find.
    xi
        The local coordinates to start from, which are replaced by the
        result.
    work
        A workspace of shape ``(2 * N + num_work, N)``.
    """
    num_axis = xi.shape[0]

    jacobian = work[:num_axis]
    matrix = work[num_axis : 2 * num_axis]
    x = work[2 * num_axis]
    residual = work[2 * num_axis + 1]
    step = work[2 * num_axis + 2]
    trial = work[2 * num_axis + 3]
    x_trial = work[2 * num_axis + 4]
    goal = work[2 * num_axis + 5]

    target(vertices, base, point, goal)
    evaluate(vertices, base, offset, xi, x, jacobian)
    error = distance_squared(x, goal)

    for _ in range(num_iteration):

        if error == 0:
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
        for _ in range(num_halving):
            for m in range(num_axis):
                trial[m] = xi[m] - fraction * step[m]
            position(vertices, base, offset, trial, x_trial)
            error_trial = distance_squared(x_trial, goal)
            if error_trial < error:
                accepted = True
                break
            fraction = fraction / 2

        if not accepted:
            # no point along the step is closer, which happens once the step
            # is down to the rounding error of the position
            return size <= tolerance * scale

        xi[:] = trial

        if fraction * size <= tolerance * scale:
            return True

        evaluate(vertices, base, offset, xi, x, jacobian)
        error = distance_squared(x, goal)

    return False


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
    The weight of each corner of the cell containing each point.

    Returns the flat arrays ``(indices_input, indices_output, values)``,
    with :math:`2^N` consecutive weights for each point, one for each corner
    of its cell, in the order the corners are numbered.

    Parameters
    ----------
    cell
        The first corner of the cell of each point, of shape
        ``(number of points, N)``.
    xi
        The local coordinates of each point in its cell, of the same shape.
    inside
        Whether each point is inside the grid.
    shape
        The number of vertices along each axis.
    weights_input
        Weights to multiply the weight of each vertex by, as a flat array, or
        an empty array for none.
    nan_outside
        Whether to set the weights of a point outside the grid to NaN rather
        than extrapolate.
    """
    num_point, num_axis = cell.shape
    stride = strides(shape)
    offset = offsets(shape)
    num_corner = offset.shape[0]
    weighted = weights_input.shape[0] > 0

    indices_input = np.empty(num_point * num_corner, dtype=np.int64)
    indices_output = np.empty(num_point * num_corner, dtype=np.int64)
    values = np.empty(num_point * num_corner, dtype=np.float64)

    for i in numba.prange(num_point):
        base = 0
        for m in range(num_axis):
            base += cell[i, m] * stride[m]
        for b in range(num_corner):
            index = base + offset[b]
            weight = 1.0
            for m in range(num_axis):
                if (b >> m) & 1:
                    weight *= xi[i, m]
                else:
                    weight *= 1 - xi[i, m]
            if weighted:
                weight *= weights_input[index]
            if nan_outside and not inside[i]:
                weight = np.nan
            w = i * num_corner + b
            indices_input[w] = index
            indices_output[w] = i
            values[w] = weight

    return indices_input, indices_output, values

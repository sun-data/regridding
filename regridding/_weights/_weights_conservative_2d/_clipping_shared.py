"""
The clipping algorithm itself, in a form both the host and a CUDA device can
compile.

:mod:`~regridding._weights._weights_conservative_2d._clipping` runs this on
the CPU through :func:`numba.njit`, and
:mod:`~regridding._weights._weights_conservative_2d._clipping_cuda` runs the
same source on a GPU through :func:`numba.cuda.jit`.  A CUDA kernel cannot
call a :func:`numba.njit` function, so the two targets cannot share a
*compiled* function, but they can share the Python source it is compiled
from, which is what :func:`build` arranges.

Sharing the source is what keeps the two implementations honest: a fix to
the clipping applies to both, and the device kernel cannot quietly drift
away from the host kernel it is tested against.

Array arguments are annotated as :class:`numpy.ndarray`, which is what they
are on the host.  On a device they are
:class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`, which is not a
subclass but is indexed the same way.

Both targets clip in double precision whatever the weights are stored as,
so the arithmetic here is ordinary double-precision arithmetic.  The one
thing it avoids is halving each term of the shoelace, which is a division
per edge where halving the total is one division per polygon.
"""

from typing import Any, Callable
import numpy as np

__all__ = [
    "num_slot",
    "check_indices_fit",
    "build",
]

num_slot = 16
"""
The number of vertex slots reserved for a polygon being clipped.

Clipping a simple quadrilateral against the four edges of a cell cannot
produce more than twelve vertices.  The boundary of the result is made of
pieces of the cell's edges and pieces of the quadrilateral's edges: each
edge of the quadrilateral meets the convex cell in at most one segment,
giving at most four, and each edge of the cell meets the quadrilateral in
at most two segments, giving at most eight.

A convex quadrilateral is bounded by eight, since the intersection of two
convex regions has only the edges of its two operands.  The extra four
appear when the cell is not convex, which is legitimate under a strong
enough distortion.  Cells whose edges cross each other are not supported,
as noted in
:func:`~regridding._weights._weights_conservative_2d._clipping.weights_conservative_2d_clipping`.
"""


def check_indices_fit(
    num_input: int,
    num_output: int,
    dtype_indices: "np.typing.DTypeLike",
) -> None:
    """
    Check that the indices this grid needs fit in the type they go in.

    The kernels write the indices rather than narrowing them afterwards, so
    one too large for its type would wrap round with nothing left to notice
    it.  The flattened grids bound the indices, so whether they fit is known
    before any are written, which is when it has to be known.

    Parameters
    ----------
    num_input
        The number of cells in the input grid.
    num_output
        The number of cells in the output grid.
    dtype_indices
        The type the indices are stored as.

    Raises
    ------
    ValueError
        If either grid needs an index the type cannot hold.
    """
    bound = max(num_input, num_output)
    info = np.iinfo(dtype_indices)
    if bound > info.max:
        raise ValueError(
            f"the grids need an index of up to {bound}, which does not fit "
            f"in {np.dtype(dtype_indices)}"
        )


def _corners(
    x: "np.ndarray",
    y: "np.ndarray",
    index_x: int,
    index_y: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Gather the four vertices of an input cell, in winding order.

    Parameters
    ----------
    x
        The :math:`x` coordinates of the input grid's vertices.
    y
        The :math:`y` coordinates of the input grid's vertices.
    index_x
        The index of the cell along the first axis.
    index_y
        The index of the cell along the second axis.
    """
    return (
        x[index_x, index_y],
        x[index_x + 1, index_y],
        x[index_x + 1, index_y + 1],
        x[index_x, index_y + 1],
        y[index_x, index_y],
        y[index_x + 1, index_y],
        y[index_x + 1, index_y + 1],
        y[index_x, index_y + 1],
    )


def _bounds(
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    y1: float,
    y2: float,
    y3: float,
    y4: float,
    num_output_x: int,
    num_output_y: int,
) -> tuple[int, int, int, int]:
    """
    Find the block of output cells a cell's bounding box touches.

    The result is clamped to the output grid, so a cell reaching past the
    edge of the grid contributes only the part which lands on it.

    Both the counting pass and the clipping pass call this, so the slice
    each cell is given is exactly the slice it writes.

    Parameters
    ----------
    x1, x2, x3, x4
        The :math:`x` coordinates of the cell's vertices.
    y1, y2, y3, y4
        The :math:`y` coordinates of the cell's vertices.
    num_output_x
        The number of output cells along the first axis.
    num_output_y
        The number of output cells along the second axis.
    """
    # `min` and `max` are written out rather than called.  A two-argument
    # `min` does not compile for a CUDA device under `numba` 0.67, where it
    # resolves to the single-iterable form and raises a signature mismatch,
    # and this source has to compile for both targets.
    lower_x = x1
    if x2 < lower_x:
        lower_x = x2
    if x3 < lower_x:
        lower_x = x3
    if x4 < lower_x:
        lower_x = x4

    upper_x = x1
    if x2 > upper_x:
        upper_x = x2
    if x3 > upper_x:
        upper_x = x3
    if x4 > upper_x:
        upper_x = x4

    lower_y = y1
    if y2 < lower_y:
        lower_y = y2
    if y3 < lower_y:
        lower_y = y3
    if y4 < lower_y:
        lower_y = y4

    upper_y = y1
    if y2 > upper_y:
        upper_y = y2
    if y3 > upper_y:
        upper_y = y3
    if y4 > upper_y:
        upper_y = y4

    # floor and ceil, without leaving the integer domain the caller needs
    index_lower_x = int(lower_x)
    if lower_x < index_lower_x:
        index_lower_x -= 1
    index_lower_y = int(lower_y)
    if lower_y < index_lower_y:
        index_lower_y -= 1
    index_upper_x = int(upper_x)
    if upper_x > index_upper_x:
        index_upper_x += 1
    index_upper_y = int(upper_y)
    if upper_y > index_upper_y:
        index_upper_y += 1

    if index_lower_x < 0:
        index_lower_x = 0
    if index_lower_y < 0:
        index_lower_y = 0
    if index_upper_x > num_output_x:
        index_upper_x = num_output_x
    if index_upper_y > num_output_y:
        index_upper_y = num_output_y

    return index_lower_x, index_upper_x, index_lower_y, index_upper_y


def _clip_halfplane(
    x_in: "np.ndarray",
    y_in: "np.ndarray",
    num_in: int,
    axis: int,
    sign: float,
    bound: float,
    x_out: "np.ndarray",
    y_out: "np.ndarray",
) -> int:
    """
    Clip a polygon against a half-plane, Sutherland-Hodgman style.

    Vertices are kept where ``sign * (vertex[axis] - bound) >= 0``, and a new
    vertex is emitted wherever an edge crosses the boundary.  The winding
    order of the input is preserved, so the signed area of the result carries
    the orientation of the input.

    Parameters
    ----------
    x_in
        The :math:`x` coordinates of the polygon's vertices.
    y_in
        The :math:`y` coordinates of the polygon's vertices.
    num_in
        The number of valid vertices in the input polygon.
    axis
        The coordinate axis of the half-plane boundary, 0 or 1.
    sign
        The side of the boundary to keep.
    bound
        The boundary coordinate.
    x_out
        An output array for the :math:`x` coordinates of the result.
    y_out
        An output array for the :math:`y` coordinates of the result.
    """
    num_out = 0

    for k in range(num_in):

        x1 = x_in[k]
        y1 = y_in[k]

        k_next = k + 1
        if k_next == num_in:
            k_next = 0

        x2 = x_in[k_next]
        y2 = y_in[k_next]

        if axis == 0:
            distance_1 = sign * (x1 - bound)
            distance_2 = sign * (x2 - bound)
        else:
            distance_1 = sign * (y1 - bound)
            distance_2 = sign * (y2 - bound)

        inside_1 = distance_1 >= 0
        inside_2 = distance_2 >= 0

        if inside_1:
            x_out[num_out] = x1
            y_out[num_out] = y1
            num_out += 1

        if inside_1 != inside_2:
            denominator = distance_1 - distance_2
            if denominator != 0:
                t = distance_1 / denominator
                x_out[num_out] = x1 + t * (x2 - x1)
                y_out[num_out] = y1 + t * (y2 - y1)
                num_out += 1

    return num_out


def build(
    jit: Callable[[Callable], Any],
    cross_2d: Callable,
) -> tuple[Callable, Callable]:
    """
    Compile the shared kernel bodies for one target.

    Parameters
    ----------
    jit
        A callable which compiles a plain Python function for the target,
        such as :func:`numba.njit` for the host or :func:`numba.cuda.jit`
        with ``device=True`` for a CUDA device.
    cross_2d
        The target's compiled :func:`regridding.geometry.cross_2d`.  A
        kernel can only call a function compiled for its own target, so
        this cannot be imported here and has to be supplied by the caller.

    Returns
    -------
    The compiled ``(num_pair, clip_cell)``, being the counting pass and the
    clipping pass.
    """

    corners = jit(_corners)
    bounds = jit(_bounds)
    clip_halfplane = jit(_clip_halfplane)

    def area_signed(x, y, num):
        """
        Compute the signed area of a polygon by the shoelace formula.

        Parameters
        ----------
        x
            The :math:`x` coordinates of the polygon's vertices.
        y
            The :math:`y` coordinates of the polygon's vertices.
        num
            The number of valid vertices.
        """
        result = 0.0

        for k in range(num):
            k_next = k + 1
            if k_next == num:
                k_next = 0
            result += cross_2d((x[k], y[k]), (x[k_next], y[k_next]))

        return result / 2

    area_signed = jit(area_signed)

    def num_pair(x, y, index_x, index_y, num_output_x, num_output_y):
        """
        Count the output cells an input cell can touch.

        The prefix sum of this over the grid is where each cell writes its
        part of the result, which is what lets the cells be visited in any
        order.

        Parameters
        ----------
        x
            The :math:`x` coordinates of the input grid's vertices.
        y
            The :math:`y` coordinates of the input grid's vertices.
        index_x
            The index of the cell along the first axis.
        index_y
            The index of the cell along the second axis.
        num_output_x
            The number of output cells along the first axis.
        num_output_y
            The number of output cells along the second axis.
        """
        x1, x2, x3, x4, y1, y2, y3, y4 = corners(x, y, index_x, index_y)
        lower_x, upper_x, lower_y, upper_y = bounds(
            x1, x2, x3, x4, y1, y2, y3, y4, num_output_x, num_output_y
        )

        span_x = upper_x - lower_x
        span_y = upper_y - lower_y
        if span_x < 0:
            span_x = 0
        if span_y < 0:
            span_y = 0

        return span_x * span_y

    num_pair = jit(num_pair)

    def clip_cell(
        x,
        y,
        weights_input,
        num_output_x,
        num_output_y,
        index_x,
        index_y,
        index_cell,
        index_write,
        subject_x,
        subject_y,
        clipped_x,
        clipped_y,
        indices_input,
        indices_output,
        values,
    ):
        """
        Clip one input cell against the output cells its bounding box touches.

        The cell writes only into the slice beginning at `index_write`, which
        the counting pass reserved for it, so the result does not depend on
        the order the cells are visited.  Slots which receive no overlap keep
        the sentinel index of ``-1`` they were initialized with.

        Parameters
        ----------
        x
            The :math:`x` coordinates of the input grid's vertices, expressed
            in output-cell units.
        y
            The :math:`y` coordinates of the input grid's vertices, expressed
            in output-cell units.
        weights_input
            Weights applied to the values of the input grid before resampling.
        num_output_x
            The number of output cells along the first axis.
        num_output_y
            The number of output cells along the second axis.
        index_x
            The index of this cell along the first axis.
        index_y
            The index of this cell along the second axis.
        index_cell
            The flattened index of this cell.
        index_write
            The index at which this cell's slice of the result begins.
        subject_x
            Scratch space of `num_slot` elements, supplied by the caller
            because the host and the device allocate it differently.
        subject_y
            Scratch space of `num_slot` elements.
        clipped_x
            Scratch space of `num_slot` elements.
        clipped_y
            Scratch space of `num_slot` elements.
        indices_input
            An output array for the flattened index of the input cell.
        indices_output
            An output array for the flattened index of the output cell.
        values
            An output array for the weights.
        """
        x1, x2, x3, x4, y1, y2, y3, y4 = corners(x, y, index_x, index_y)
        lower_x, upper_x, lower_y, upper_y = bounds(
            x1, x2, x3, x4, y1, y2, y3, y4, num_output_x, num_output_y
        )

        # Shift the cell onto its own block of candidates.  The shoelace sums
        # differences of coordinates, so working at the scale of the block
        # rather than of the whole grid keeps the significant figures that
        # single precision would otherwise lose to cancellation.
        x1 = x1 - lower_x
        x2 = x2 - lower_x
        x3 = x3 - lower_x
        x4 = x4 - lower_x
        y1 = y1 - lower_y
        y2 = y2 - lower_y
        y3 = y3 - lower_y
        y4 = y4 - lower_y

        subject_x[0] = x1
        subject_x[1] = x2
        subject_x[2] = x3
        subject_x[3] = x4
        subject_y[0] = y1
        subject_y[1] = y2
        subject_y[2] = y3
        subject_y[3] = y4

        area_cell = area_signed(subject_x, subject_y, 4)

        if area_cell == 0:
            return

        weight_cell = weights_input[index_x, index_y] / area_cell

        for cell_x in range(lower_x, upper_x):
            for cell_y in range(lower_y, upper_y):

                subject_x[0] = x1
                subject_x[1] = x2
                subject_x[2] = x3
                subject_x[3] = x4
                subject_y[0] = y1
                subject_y[1] = y2
                subject_y[2] = y3
                subject_y[3] = y4

                bound_x = cell_x - lower_x
                bound_y = cell_y - lower_y

                # the candidate cells come from the bounding box, so the cell
                # always overlaps the slab being clipped against in `x` and
                # these two clips cannot empty the polygon; only the `y` clips
                # below need an early exit
                num = clip_halfplane(
                    subject_x,
                    subject_y,
                    4,
                    0,
                    1.0,
                    bound_x,
                    clipped_x,
                    clipped_y,
                )
                num = clip_halfplane(
                    clipped_x,
                    clipped_y,
                    num,
                    0,
                    -1.0,
                    bound_x + 1,
                    subject_x,
                    subject_y,
                )
                num = clip_halfplane(
                    subject_x,
                    subject_y,
                    num,
                    1,
                    1.0,
                    bound_y,
                    clipped_x,
                    clipped_y,
                )
                if num < 3:
                    continue
                num = clip_halfplane(
                    clipped_x,
                    clipped_y,
                    num,
                    1,
                    -1.0,
                    bound_y + 1,
                    subject_x,
                    subject_y,
                )
                if num < 3:
                    continue

                area = area_signed(subject_x, subject_y, num)

                if area == 0:
                    continue

                indices_input[index_write] = index_cell
                indices_output[index_write] = cell_x * num_output_y + cell_y
                values[index_write] = area * weight_cell
                index_write += 1

    clip_cell = jit(clip_cell)

    return num_pair, clip_cell

import math
import numba
import regridding

__all__ = [
    "line_equation_2d",
    "bounding_boxes_intersect_2d",
    "two_line_segment_intersection_parameters",
]


@numba.njit(inline="always", error_model="numpy")
def line_equation_2d(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Test if a given point lies above, on, or below a line specified by two
    endpoints.

    Returns zero if a point is on the line.
    Otherwise, points above the line have the opposite sign of points below
    the line.

    Parameters
    ----------
    x
        :math:`x`-component of the test point
    y
        :math:`y`-component of the test point
    x1
        :math:`x`-component of the first endpoint of the line
    y1
        :math:`y`-component of the first endpoint of the line
    x2
        :math:`x`-component of the second endpoint of the line
    y2
        :math:`y`-component of the second endpoint of the line
    """
    result = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return result


@numba.njit(inline="always", error_model="numpy")
def bounding_boxes_intersect_2d(
    x_p1: float,
    y_p1: float,
    x_p2: float,
    y_p2: float,
    x_q1: float,
    y_q1: float,
    x_q2: float,
    y_q2: float,
) -> bool:
    """
    Test if two bounding boxes, :math:`p` and :math:`q`, intersect.

    Parameters
    ----------
    x_p1
        :math:`x`-coordinate of the first point of box :math:`p`
    y_p1
        :math:`y`-coordinate of the first point of box :math:`p`
    x_p2
        :math:`x`-coordinate of the second point of box :math:`p`
    y_p2
        :math:`y`-coordinate of the second point of box :math:`p`
    x_q1
        :math:`x`-coordinate of the first point of box :math:`q`
    y_q1
        :math:`y`-coordinate of the first point of box :math:`q`
    x_q2
        :math:`x`-coordinate of the second point of box :math:`q`
    y_q2
        :math:`y`-coordinate of the second point of box :math:`q`
    """
    dx_p = x_p2 - x_p1
    dy_p = y_p2 - y_p1
    dx_q = x_q2 - x_q1
    dy_q = y_q2 - y_q1

    x_p = x_p1 + dx_p / 2
    y_p = y_p1 + dy_p / 2
    x_q = x_q1 + dx_q / 2
    y_q = y_q1 + dy_q / 2

    intersect_x = (2 * abs(x_p - x_q)) <= (dx_p + dx_q)
    intersect_y = (2 * abs(y_p - y_q)) <= (dy_p + dy_q)

    return intersect_x and intersect_y


@numba.njit(inline="always", error_model="numpy")
def two_line_segment_intersection_parameters(
    x_p1: float,
    y_p1: float,
    x_p2: float,
    y_p2: float,
    x_q1: float,
    y_q1: float,
    x_q2: float,
    y_q2: float,
) -> tuple[float, float, float]:
    r"""
    Computes the parameters
    (:math:`\text{sdet}`, :math:`\text{tdet}`, and :math:`\text{det}`)
    associated with the intersection of two 2D line segments,
    :math:`p` and :math:`q`, as described in :footcite:t:`GraphicsGemsIII`.

    Parameters
    ----------
    x_p1
        :math:`x` component of the first point in line :math:`p`
    y_p1
        :math:`y` component of the first point in line :math:`p`
    x_p2
        :math:`x` component of the second point in line :math:`p`
    y_p2
        :math:`y` component of the second point in line :math:`p`
    x_q1
        :math:`x` component of the first point in line :math:`q`
    y_q1
        :math:`y` component of the first point in line :math:`q`
    x_q2
        :math:`x` component of the second point in line :math:`q`
    y_q2
        :math:`x` component of the second point in line :math:`q`

    References
    ----------

    .. footbibliography::

    """
    bounding_boxes_intersect = bounding_boxes_intersect_2d(
        x_p1=x_p1,
        y_p1=y_p1,
        x_p2=x_p2,
        y_p2=y_p2,
        x_q1=x_q1,
        y_q1=y_q1,
        x_q2=x_q2,
        y_q2=y_q2,
    )
    if not bounding_boxes_intersect:
        return math.nan, math.nan, math.nan

    a = line_equation_2d(
        x=x_q1,
        y=y_q1,
        x1=x_p1,
        y1=y_p1,
        x2=x_p2,
        y2=y_p2,
    )
    b = line_equation_2d(
        x=x_q2,
        y=y_q2,
        x1=x_p1,
        y1=y_p1,
        x2=x_p2,
        y2=y_p2,
    )
    if regridding.math.sign(a) == regridding.math.sign(b):
        return math.nan, math.nan, math.nan

    c = line_equation_2d(
        x=x_p1,
        y=y_p1,
        x1=x_q1,
        y1=y_q1,
        x2=x_q2,
        y2=y_q2,
    )
    d = line_equation_2d(
        x=x_p2,
        y=y_p2,
        x1=x_q1,
        y1=y_q1,
        x2=x_q2,
        y2=y_q2,
    )
    if regridding.math.sign(c) == regridding.math.sign(d):
        return math.nan, math.nan, math.nan

    det = a - b
    sdet = +a
    tdet = -c

    return sdet, tdet, det

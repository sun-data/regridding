"""
Numba-compiled computational geometry routines needed for regridding operations.
"""

import math
import numpy as np
import numba
import regridding

__all__ = [
    "line_equation_2d",
    "bounding_boxes_intersect_2d",
    "bounding_boxes_intersect_3d",
    "two_line_segment_intersection_parameters",
    "line_triangle_intersection_parameters",
    "line_triangle_intersection",
    "point_is_inside_polygon",
]


@numba.njit(cache=True, inline="always", error_model="numpy")
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


@numba.njit(cache=True, inline="always", error_model="numpy")
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

    If the edges of the two boxes are touching, it's counted as an intersection.

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

    Examples
    --------

    Create 4 boxes :math:`P`, :math:`Q`, :math:`R`, and :math:`S`.
    Check if boxes :math:`Q`, :math:`R`, and :math:`S` intersect with
    box :math:`P`, and plot the boxes as filled if they intersect,
    and unfilled if they don't intersect.

    .. jupyter-execute::

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import regridding

        # box P
        x_p1 = 0
        y_p1 = 0
        x_p2 = 1
        y_p2 = 1

        # box Q
        x_q1 = 1.1
        y_q1 = 1.1
        x_q2 = 2.1
        y_q2 = 2.1

        # box R
        x_r1 = 1
        y_r1 = 1
        x_r2 = 2
        y_r2 = 2

        # box S
        x_s1 = 0.9
        y_s1 = 0.9
        x_s2 = 1.9
        y_s2 = 1.9

        p_and_q_intersect = regridding.geometry.bounding_boxes_intersect_2d(
            x_p1=x_p1, y_p1=y_p1,
            x_p2=x_p2, y_p2=y_p2,
            x_q1=x_q1, y_q1=y_q1,
            x_q2=x_q2, y_q2=y_q2,
        )

        p_and_r_intersect = regridding.geometry.bounding_boxes_intersect_2d(
            x_p1=x_p1, y_p1=y_p1,
            x_p2=x_p2, y_p2=y_p2,
            x_q1=x_r1, y_q1=y_r1,
            x_q2=x_r2, y_q2=y_r2,
        )

        p_and_s_intersect = regridding.geometry.bounding_boxes_intersect_2d(
            x_p1=x_p1, y_p1=y_p1,
            x_p2=x_p2, y_p2=y_p2,
            x_q1=x_s1, y_q1=y_s1,
            x_q2=x_s2, y_q2=y_s2,
        )

        fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(9, 3), constrained_layout=True)
        axs[0].add_patch(
            mpl.patches.Rectangle(xy=(x_p1, y_p1), width=x_p2 - x_p1, height=y_p2 - y_p1, fill=p_and_q_intersect)
        )
        axs[0].add_patch(
            mpl.patches.Rectangle(xy=(x_q1, y_q1), width=x_q2 - x_q1, height=y_q2 - y_q1, fill=p_and_q_intersect)
        )
        axs[1].add_patch(
            mpl.patches.Rectangle(xy=(x_p1, y_p1), width=x_p2 - x_p1, height=y_p2 - y_p1, fill=p_and_r_intersect)
        )
        axs[1].add_patch(
            mpl.patches.Rectangle(xy=(x_r1, y_r1), width=x_r2 - x_r1, height=y_r2 - y_r1, fill=p_and_r_intersect)
        )
        axs[2].add_patch(
            mpl.patches.Rectangle(xy=(x_p1, y_p1), width=x_p2 - x_p1, height=y_p2 - y_p1, fill=p_and_s_intersect)
        )
        axs[2].add_patch(
            mpl.patches.Rectangle(xy=(x_s1, y_s1), width=x_s2 - x_s1, height=y_s2 - y_s1, fill=p_and_s_intersect)
        )
        axs[0].autoscale_view();
    """
    if x_p1 > x_p2:
        x_p1, x_p2 = x_p2, x_p1
    if y_p1 > y_p2:
        y_p1, y_p2 = y_p2, y_p1
    if x_q1 > x_q2:
        x_q1, x_q2 = x_q2, x_q1
    if y_q1 > y_q2:
        y_q1, y_q2 = y_q2, y_q1

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


@numba.njit(cache=True, inline="always", error_model="numpy")
def bounding_boxes_intersect_3d(
    x_p1: float,
    y_p1: float,
    z_p1: float,
    x_p2: float,
    y_p2: float,
    z_p2: float,
    x_q1: float,
    y_q1: float,
    z_q1: float,
    x_q2: float,
    y_q2: float,
    z_q2: float,
) -> bool:
    """
    Test if two bounding boxes, :math:`p` and :math:`q`, intersect.

    If the edges of the two boxes are touching, it's counted as an intersection.

    Parameters
    ----------
    x_p1
        :math:`x`-coordinate of the first point of box :math:`p`
    y_p1
        :math:`y`-coordinate of the first point of box :math:`p`
    z_p1
        :math:`z`-coordinate of the first point of box :math:`p`
    x_p2
        :math:`x`-coordinate of the second point of box :math:`p`
    y_p2
        :math:`y`-coordinate of the second point of box :math:`p`
    z_p2
        :math:`z`-coordinate of the second point of box :math:`p`
    x_q1
        :math:`x`-coordinate of the first point of box :math:`q`
    y_q1
        :math:`y`-coordinate of the first point of box :math:`q`
    z_q1
        :math:`z`-coordinate of the first point of box :math:`q`
    x_q2
        :math:`x`-coordinate of the second point of box :math:`q`
    y_q2
        :math:`y`-coordinate of the second point of box :math:`q`
    z_q2
        :math:`z`-coordinate of the second point of box :math:`q`
    """
    if x_p1 > x_p2:
        x_p1, x_p2 = x_p2, x_p1
    if y_p1 > y_p2:
        y_p1, y_p2 = y_p2, y_p1
    if z_p1 > z_p2:
        z_p1, z_p2 = z_p2, z_p1
    if x_q1 > x_q2:
        x_q1, x_q2 = x_q2, x_q1
    if y_q1 > y_q2:
        y_q1, y_q2 = y_q2, y_q1
    if z_q1 > z_q2:
        z_q1, z_q2 = z_q2, z_q1

    dx_p = x_p2 - x_p1
    dy_p = y_p2 - y_p1
    dz_p = z_p2 - z_p1

    dx_q = x_q2 - x_q1
    dy_q = y_q2 - y_q1
    dz_q = z_q2 - z_q1

    x_p = x_p1 + dx_p / 2
    y_p = y_p1 + dy_p / 2
    z_p = z_p1 + dz_p / 2

    x_q = x_q1 + dx_q / 2
    y_q = y_q1 + dy_q / 2
    z_q = z_q1 + dz_q / 2

    intersect_x = (2 * abs(x_p - x_q)) <= (dx_p + dx_q)
    intersect_y = (2 * abs(y_p - y_q)) <= (dy_p + dy_q)
    intersect_z = (2 * abs(z_p - z_q)) <= (dz_p + dz_q)

    return intersect_x and intersect_y and intersect_z


@numba.njit(cache=True, inline="always", error_model="numpy")
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
    :math:`p` and :math:`q`.

    This function uses the method described in :footcite:t:`GraphicsGemsIII`.
    :math:`\text{sdet}`, :math:`\text{tdet}`, and :math:`\text{det}` are expected
    to be used to compute the quantities

    .. math::

        s &= \text{sdet} / \text{det} \\
        t &= \text{tdet} / \text{det},

    which can be used to compute the intersection :math:`(x, y)` using the
    parametric equations of the lines:

    .. math::

        x &= (1 - s) * x_{\text{q1}} + s * x_{\text{q2}} = (1 - t) * x_{\text{p1}} + t * x_{\text{p2}} \\
        y &= (1 - s) * y_{\text{q1}} + s * y_{\text{q2}} = (1 - t) * y_{\text{p1}} + t * y_{\text{p2}}.

    The quantities :math:`s` and :math:`t` are not computed directly
    because doing so leads to a loss of precision.

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

    Examples
    --------

    Plot two lines and their intersection

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import regridding

        # line P
        x_p1 = 0
        y_p1 = 0
        x_p2 = 2
        y_p2 = 2

        # line Q
        x_q1 = 0
        y_q1 = 1
        x_q2 = 2
        y_q2 = 0

        sdet, tdet, det = regridding.geometry.two_line_segment_intersection_parameters(
            x_p1=x_p1, y_p1=y_p1,
            x_p2=x_p2, y_p2=y_p2,
            x_q1=x_q1, y_q1=y_q1,
            x_q2=x_q2, y_q2=y_q2,
        )

        s = sdet / det

        x = (1 - s) * x_p1 + s * x_p2
        y = (1 - s) * y_p1 + s * y_p2

        plt.figure()
        plt.plot([x_p1, x_p2], [y_p1, y_p2], label="line $p$")
        plt.plot([x_q1, x_q2], [y_q1, y_q2], label="line $q$")
        plt.scatter(x, y, color="black", zorder=10, label="intersection")
        plt.legend();

    |

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
        return math.inf, math.inf, 1

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
    if (a != 0) and (b != 0):
        if regridding.math.sign(a) == regridding.math.sign(b):
            return math.inf, math.inf, 1

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
    if (c != 0) and (d != 0):
        if regridding.math.sign(c) == regridding.math.sign(d):
            return math.inf, math.inf, 1

    det = a - b
    sdet = +a
    tdet = -c

    return sdet, tdet, det


@numba.njit(cache=True, inline="always", error_model="numpy")
def line_triangle_intersection_parameters(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    triangle: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float]:
    """
    Compute the parameters :math:`t,u,v` describing the point of intersection
    between a line and triangle in 3D.

    Parameters
    ----------
    line
        Two 3D points representing the line segment.
    triangle
        Three 3D points representing the triangle.

    See Also
    --------
    :func:`line_triangle_intersection`: A function which can use these parameters to compute the actual intersection.
    """

    l_a, l_b = line

    p_0, p_1, p_2 = triangle

    l_ab = regridding.math.difference_3d(l_b, l_a)
    l_ab = regridding.math.negate_3d(l_ab)

    p_01 = regridding.math.difference_3d(p_1, p_0)
    p_02 = regridding.math.difference_3d(p_2, p_0)

    n = regridding.math.cross_3d(p_01, p_02)

    det = regridding.math.dot_3d(l_ab, n)

    s = regridding.math.difference_3d(l_a, p_0)

    t = n
    u = regridding.math.cross_3d(p_02, l_ab)
    v = regridding.math.cross_3d(l_ab, p_01)

    t = regridding.math.dot_3d(t, s) / det
    u = regridding.math.dot_3d(u, s) / det
    v = regridding.math.dot_3d(v, s) / det

    return t, u, v


@numba.njit(cache=True, inline="always", error_model="numpy")
def line_triangle_intersection(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    tuv: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Compute the 3D point where a line intersects a triangle
    using the Parametric form described in the
    `Line-plane intersection Wikipedia article <https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Parametric_form>`_.


    Parameters
    ----------
    line
        Two 3D points representing a line segment.
    tuv
        Intersection parameters computed using
        :func:`line_triangle_intersection_parameters`.

    See Also
    --------
    :func:`line_triangle_intersection_parameters`: the function used to compute `tuv`.

    Examples
    --------

    Compute the intercept between a line and a triangle and plot the result.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import regridding

        # Define a line segment
        line = (
            (1, 1, -1),
            (-1, -1, 1),
        )

        # Define a triangle
        triangle = (
            (0, 1, 1),
            (1, -1, -1),
            (-1, -1, -1),
        )

        # Compute the intercept parameters between the line and triangle
        tuv = regridding.geometry.line_triangle_intersection_parameters(
            line=line,
            triangle=triangle,
        )

        # Compute the actual intercept using the parameters
        intercept = regridding.geometry.line_triangle_intersection(
            line=line,
            tuv=tuv,
        )

        # Plot the result
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.plot(
            *np.array(line).T
        );
        ax.plot(
            *np.array(triangle).T.take(np.arange(-1, 3), axis=1)
        );
        ax.scatter(
            *intercept
        );
    """
    l_a, l_b = line

    t, u, v = tuv

    l_ab = regridding.math.difference_3d(l_b, l_a)

    tl_ab = regridding.math.multiply_3d(t, l_ab)

    return regridding.math.sum_3d(l_a, tl_ab)


@numba.njit(cache=True, inline="always", error_model="numpy")
def point_is_inside_polygon(
    x: float,
    y: float,
    vertices_x: np.ndarray,
    vertices_y: np.ndarray,
) -> bool:
    """
    Check if a given point is inside or on the boundary of a polygon specified
    by its vertices.

    This function uses the extended winding number algorithm described by
    :footcite:t:`Kumar2018`, which addresses boundary issues with the "classic"
    winding number algorithm in :footcite:t:`Alciatore1995`.

    Parameters
    ----------
    x
        :math:`x` component of the test point
    y
        :math:`y` component of the test point
    vertices_x
        :math:`x` component of the polygon's vertices
    vertices_y
        :math:`y` component of the polygon's vertices

    References
    ----------

    .. footbibliography::

    """

    vertices_x = vertices_x - x
    vertices_y = vertices_y - y

    w = 0

    for v in range(len(vertices_x)):
        i = v - 1

        x0 = vertices_x[i + 0]
        y0 = vertices_y[i + 0]
        x1 = vertices_x[i + 1]
        y1 = vertices_y[i + 1]

        if regridding.math.sign(y0) != regridding.math.sign(y1):
            x_intercept = x0 + y0 * (x1 - x0) / (y1 - y0)

            if x_intercept > 0:
                if (y0 == 0) and (x0 > 0):
                    if y1 > 0:
                        w = w + 0.5
                    elif y1 < 0:
                        w = w - 0.5
                elif (y1 == 0) and (x1 > 0):
                    if y0 < 0:
                        w = w + 0.5
                    elif y0 > 0:
                        w = w - 0.5
                elif y0 < 0:
                    w = w + 1
                elif y0 > 0:
                    w = w - 1

            elif x_intercept < 0:
                if (y0 == 0) and (x0 < 0):
                    if y1 < 0:
                        w = w + 0.5
                    elif y1 > 0:
                        w = w - 0.5
                elif (y1 == 0) and (x1 < 0):
                    if y0 > 0:
                        w = w + 0.5
                    elif y0 < 0:
                        w = w - 0.5
                elif y0 > 0:
                    w = w + 1
                elif y0 < 0:
                    w = w - 1

    if w == 0:
        return False
    else:
        return True


@numba.njit(cache=True, inline="always", error_model="numpy")
def solid_angle(
    point: tuple[float, float, float],
    triangle: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> bool:
    r"""
    Calculate the solid angle subtended by a triangle with respect to a
    query point using the :cite:t:`Oosterom1985` method.

    Parameters
    ----------
    point
        A 3D query point.
    triangle
        A list of 3D vertices describing the triangle.
        Vertices oriented

    Notes
    -----

    The solid angle :math:`Omega` subtended by a triangular surface is

    .. math::

        \tan \left( \frac{1}{2} \Omega \right) = \frac{\vec{a} \cdot (\vec{b} \times \vec{c})}
            {a b c + (\vec{a} \cdot \vec{b}) \, c + (\vec{a} \cdot \vec{c}) \, b + (\vec{b} \cdot \vec{c}) \, a}

    """



import pytest
import math
import numpy as np
import numba
import regridding


@pytest.mark.parametrize("x", [0, 0.5, 1])
@pytest.mark.parametrize("y_offset", [-1, 0, 2])
@pytest.mark.parametrize("x1", [-1, 0])
@pytest.mark.parametrize("y1", [-2, 0])
@pytest.mark.parametrize("x2", [2, 3])
@pytest.mark.parametrize("y2", [2])
def test_line_equation_2d(
    x: float,
    y_offset: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    y_below = (x - x1) * (y2 - y1) / (x2 - x1) + y1 - 1
    y_equal = (x - x1) * (y2 - y1) / (x2 - x1) + y1 + 0
    y_above = (x - x1) * (y2 - y1) / (x2 - x1) + y1 + 1

    result_below = regridding.geometry.line_equation_2d(
        x=x,
        y=y_below,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    result_equal = regridding.geometry.line_equation_2d(
        x=x,
        y=y_equal,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    result_above = regridding.geometry.line_equation_2d(
        x=x,
        y=y_above,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    assert np.isclose(result_equal, 0)
    assert np.sign(result_above) != np.sign(result_below)


_box = (
    (-2, -2, -2),
    (+2, +2, +2),
)


@pytest.mark.parametrize(
    argnames="point,box,result_expected",
    argvalues=[
        (
            (1, 1, 1),
            _box,
            True,
        ),
        (
            (10, 1, 1),
            _box,
            False,
        ),
        (
            (1, 10, 1),
            _box,
            False,
        ),
        (
            (1, 1, 10),
            _box,
            False,
        ),
        (
            (-2, -2, -2),
            _box,
            True,
        ),
        (
            (2, 0, 0),
            _box,
            True,
        ),
    ],
)
def test_point_is_inside_box_3d(
    point: tuple[float, float, float],
    box: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    result_expected: bool,
):
    result = regridding.geometry.point_is_inside_box_3d(
        point=point,
        box=box,
    )

    assert result == result_expected


@pytest.mark.parametrize(
    argnames="x_p1,y_p1,x_p2,y_p2,x_q1,y_q1,x_q2,y_q2,result_expected",
    argvalues=[
        (0, 0, 1, 1, 0, 0, 1, 1, True),
        (0, 0, 1, 1, 2, 0, 3, 1, False),
        (0, 0, 2, 2, 1, 1, 3, 3, True),
        (0, 0, 1, 1, 1, 1, 2, 2, True),
        (-1, -1, 0, 0, 0, 0, 1, 1, True),
        (-1, 1, 0, 0, 0, 0, 1, 1, True),
        (-1, -1, 0, 0, 1, 0, 0, 1, True),
        (1, 1, 0, 0, 2, 2, 1, 1, True),
    ],
)
def test_bounding_boxes_intersect_2d(
    x_p1: float,
    y_p1: float,
    x_p2: float,
    y_p2: float,
    x_q1: float,
    y_q1: float,
    x_q2: float,
    y_q2: float,
    result_expected: bool,
):
    result = regridding.geometry.bounding_boxes_intersect_2d(
        x_p1=x_p1,
        y_p1=y_p1,
        x_p2=x_p2,
        y_p2=y_p2,
        x_q1=x_q1,
        y_q1=y_q1,
        x_q2=x_q2,
        y_q2=y_q2,
    )
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="x_p1,y_p1,z_p1,x_p2,y_p2,z_p2,x_q1,y_q1,z_q1,x_q2,y_q2,z_q2,result_expected",
    argvalues=[
        (0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, True),
        (0, 0, 0, 1, 1, 1, 2, 0, 0, 3, 1, 1, False),
        (0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, True),
        (0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, True),
        (-1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True),
        (-1, 1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, True),
        (0, 0, 0, -1, 1, -1, 1, 1, 1, 0, 0, 0, True),
    ],
)
def test_bounding_boxes_intersect_3d(
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
    result_expected: bool,
):
    result = regridding.geometry.bounding_boxes_intersect_3d(
        x_p1=x_p1,
        y_p1=y_p1,
        z_p1=z_p1,
        x_p2=x_p2,
        y_p2=y_p2,
        z_p2=z_p2,
        x_q1=x_q1,
        y_q1=y_q1,
        z_q1=z_q1,
        x_q2=x_q2,
        y_q2=y_q2,
        z_q2=z_q2,
    )
    assert result == result_expected


@pytest.mark.parametrize(
    argnames="x_p1,y_p1,x_p2,y_p2,x_q1,y_q1,x_q2,y_q2,sdet_expected,tdet_expected,det_expected",
    argvalues=[
        (0, 0, 1, 1, 2, 0, 3, 1, math.inf, math.inf, 1),
        (-1, -1, 0, 0, 1, 0, 0, 1, math.inf, math.inf, 1),
        (1, 0, 0, 1, -1, -1, 0, 0, math.inf, math.inf, 1),
        (-1, -1, 1, 1, 1, -1, -1, 1, 4, 4, 8),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 2, 2, 1, 1, 3, 3, 0, 0, 0),
    ],
)
def test_two_line_segment_intersection_parameters(
    x_p1: float,
    y_p1: float,
    x_p2: float,
    y_p2: float,
    x_q1: float,
    y_q1: float,
    x_q2: float,
    y_q2: float,
    sdet_expected: float,
    tdet_expected: float,
    det_expected: float,
):
    sdet, tdet, det = regridding.geometry.two_line_segment_intersection_parameters(
        x_p1=x_p1,
        y_p1=y_p1,
        x_p2=x_p2,
        y_p2=y_p2,
        x_q1=x_q1,
        y_q1=y_q1,
        x_q2=x_q2,
        y_q2=y_q2,
    )

    if math.isnan(sdet_expected):
        assert math.isnan(sdet)
    else:
        assert sdet == sdet_expected

    if math.isnan(tdet_expected):
        assert math.isnan(tdet)
    else:
        assert tdet == tdet_expected

    if math.isnan(det_expected):
        assert math.isnan(det)
    else:
        assert det == det_expected


@pytest.mark.parametrize(
    argnames="line,triangle,result",
    argvalues=[
        (
            ((0, 0, -1), (0, 0, 1)),
            ((1, 0, 0), (-1, 1, 0), (-1, -1, 0)),
            (0, 0, 0),
        ),
        (
            ((0.5, 0, -1), (0.5, 0, 1)),
            ((1, 0, 0), (-1, 1, 0), (-1, -1, 0)),
            (0.5, 0, 0),
        ),
        (
            ((1, 0, -1), (1, 0, 1)),
            ((1, 0, 0), (-1, 1, 0), (-1, -1, 0)),
            (1, 0, 0),
        ),
        (
            ((1, 0, -1), (-1, 0, 1)),
            ((1, 0, 0), (-1, 1, 0), (-1, -1, 0)),
            (0, 0, 0),
        ),
    ],
)
def test_line_triangle_intersection(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    triangle: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    result: tuple[float, float, float],
):
    tuv = regridding.geometry.line_triangle_intersection_parameters(
        line=line,
        triangle=triangle,
    )

    intersection = regridding.geometry.line_triangle_intersection(
        line=line,
        tuv=tuv,
    )

    assert np.allclose(intersection, result)


@pytest.mark.parametrize(
    argnames="x,y,result_expected",
    argvalues=[
        (8, 4, False),
        (7, 4, False),
        (6, 4, False),
        (5, 4, True),
        (4, 4, True),
        (3.5, 4, True),
        (3, 4, True),
        (2, 4, True),
        (1, 4, False),
        (0, 4, False),
        (-1, 4, False),
        (-2, 4, False),
        (-3, 4, False),
        (-4, 4, False),
        (-5, 4, False),
        (-6, 4, False),
        (-7, 4, False),
        (-8, 4, False),
        (3, 8, False),
        (3, 7, False),
        (3, 6, False),
        (3, 5, True),
        (3, 4, True),
        (3, 3.5, True),
        (3, 3, True),
        (3, 2, True),
        (3, 1, False),
        (3, 0, False),
        (3, -1, False),
        (3, -2, False),
        (3, -3, False),
        (3, -4, False),
        (3, -5, False),
        (3, -6, False),
        (3, -7, False),
        (3, -8, False),
    ],
)
def test_point_is_inside_red_cross(
    x: float,
    y: float,
    result_expected: bool,
):
    vertices_x = np.array([2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1]) + 1
    vertices_y = np.array([2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 2]) + 1

    result = regridding.geometry.point_is_inside_polygon(
        x=x,
        y=y,
        vertices_x=vertices_x,
        vertices_y=vertices_y,
    )
    assert result == result_expected

    result_reversed = regridding.geometry.point_is_inside_polygon(
        x=x,
        y=y,
        vertices_x=vertices_x[::-1],
        vertices_y=vertices_y[::-1],
    )
    assert result_reversed == result_expected


_triangle_octant = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


@pytest.mark.parametrize(
    argnames="point,triangle,result_expected",
    argvalues=[
        (
            (10000, 0, 0),
            _triangle_octant,
            0,
        ),
        (
            (0, 0, 0),
            _triangle_octant,
            math.pi / 2,
        ),
        (
            (1, 0, 0),
            _triangle_octant,
            0,
        ),
        (
            (2, -1, 0),
            _triangle_octant,
            0,
        ),
        (
            (1, 0, 0),
            _triangle_octant,
            0,
        ),
        (
            (1 / 3, 1 / 3, 1 / 3 - 1e-6),
            _triangle_octant,
            2 * np.pi,
        ),
        (
            (1 / 3, 1 / 3, 1 / 3 + 1e-6),
            _triangle_octant,
            -2 * np.pi,
        ),
    ],
)
def test_solid_angle(
    point: tuple[float, float, float],
    triangle: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    result_expected: float,
):
    result = regridding.geometry.solid_angle(
        point=point,
        triangle=triangle,
    )

    assert np.allclose(result, result_expected)


_polyhedron_octant = [
    ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
]


@pytest.mark.parametrize(
    argnames="point,polyhedron,result_expected",
    argvalues=[
        (
            (0.1, 0.1, 0.1),
            _polyhedron_octant,
            True,
        ),
        (
            (1, 1, 1),
            _polyhedron_octant,
            False,
        ),
        (
            (100, -100, 100),
            _polyhedron_octant,
            False,
        ),
        (
            (1 / 3, 1 / 3, 1 / 3),
            _polyhedron_octant,
            True,
        ),
        (
            (1 / 3, 1 / 3, 1 / 3 + 1e-6),
            _polyhedron_octant,
            False,
        ),
        (
            (1 / 2, 0, 0),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 1 / 2, 0),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 0, 1 / 2),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 0, 0),
            _polyhedron_octant,
            True,
        ),
        (
            (1, 0, 0),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 1, 0),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 0, 1),
            _polyhedron_octant,
            True,
        ),
        (
            (0, 0, -1e-6),
            _polyhedron_octant,
            False,
        ),
    ],
)
def test_point_is_inside_polyhedron(
    point: tuple[float, float, float],
    polyhedron: numba.typed.List[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ],
    result_expected: bool,
):
    result = regridding.geometry.point_is_inside_polyhedron(
        point=point,
        polyhedron=polyhedron,
    )

    assert result == result_expected

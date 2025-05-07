"""
Create lists of intercepts generated during axis sweeps and determine their
contribution to the final list of weights.
"""

import numpy as np
import numba
import regridding as rg
from . import _arrays
from . import _grids


@numba.njit(cache=True)
def empty(
    shape_input: tuple[int, int, int],
    shape_output: tuple[int, int, int],
) -> numba.typed.List:
    """
    Create an empty list of intercepts between each plane in the
    input grid and each plane in the output grid.

    There are three planes in the input grid and thre

    Parameters
    ----------
    shape_input
        The shape of the input grid.
    shape_output
        The shape of the output grid.
    """
    intercepts = numba.typed.List()
    for a in range(len(shape_input)):
        intercepts_a = numba.typed.List()
        for b in range(len(shape_output)):
            intercepts_ab = numba.typed.List()
            for i in range(shape_input[a]):
                intercepts_abi = numba.typed.List()
                for j in range(shape_output[b]):
                    intercepts_abij = numba.typed.List()
                    for x in range(0):
                        intercepts_abij.append((0, 0, (0.0, 0.0, 0.0)))
                    intercepts_abi.append(intercepts_abij)
                intercepts_ab.append(intercepts_abi)
            intercepts_a.append(intercepts_ab)
        intercepts.append(intercepts_a)
    return intercepts


def insert_intercept(
    intercepts: numba.typed.List[tuple[int, int, tuple[float, float, float]]],
    intercept_new: tuple[int, int, tuple[float, float, float]],
) -> None:
    """
    Insert a new intercept into the current list of intercepts at the correct
    point so that the list maintains sorted order.

    Parameters
    ----------
    intercepts
        The current list of intercepts.
    intercept_new
        A new intercept to insert into the list.
    """


def _line_point_closest_approach_parameter(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    point: tuple[float, float, float],
) -> float:
    """
    Compute the parameter describing the point of closest approach of line
    segment to a point.

    Based on a `StackExchange answer by John Hughes <https://math.stackexchange.com/a/2193733/502546>`_.

    Parameters
    ----------
    line
        A line segment in 3D.
    point
        A point in 3D
    """

    a, b = line
    p = point

    v = rg.math.difference_3d(b, a)
    u = rg.math.difference_3d(a, p)

    numerator = -rg.math.dot_3d(v, u)
    denominator = rg.math.dot_3d(v, v)

    return numerator / denominator


def _line_point_closest_approach(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    t: float,
) -> tuple[float, float, float]:
    """
    Compute the point of closest approach of line segment to a point using
    the parameter computed using :func:`_line_point_closest_approach_parameter`.

    Based on a `StackExchange answer by John Hughes <https://math.stackexchange.com/a/2193733/502546>`_.

    Parameters
    ----------
    line
        A line segment in 3D.
    t
        The parameter of closest approach.
    """
    a, b = line

    s = 1 - t

    a2 = rg.math.multiply_3d(s, a)
    b2 = rg.math.multiply_3d(t, b)

    return rg.math.sum_3d(a2, b2)


@numba.njit(cache=True)
def sweep(
    intercepts: numba.typed.List,
    weights: numba.typed.List,
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
) -> None:
    """
    Sweep through the list of intercepts and add the volume of each
    segment to the list of weights.

    Parameters
    ----------
    intercepts
        A sequence of intercepts and the corresponding indices in the
        input/output grids.
    weights
        The current list of weights.
        New weights will be appended to this list.
    grid_input
        The vertices of the input grid.
    grid_output
        The vertices of the output grid.
    volume_input
        The volume of each cell in the input grid.
    """

    x_input, y_input, z_input = grid_input
    x_output, y_output, z_output = grid_output

    shape_input = x_input.shape
    shape_output = x_output.shape

    shape_centers_input = _grids.shape_centers(shape_input)
    shape_centers_output = _grids.shape_centers(shape_output)

    for a in range(len(intercepts)):

        intercepts_a = intercepts[a]

        normal_a = [1 if axis == a else 0 for axis in range(len(intercepts))]

        for b in range(len(intercepts_a)):

            intercepts_ab = intercepts_a[b]

            normal_b = [1 if axis == b else 0 for axis in range(len(intercepts))]

            for i in range(len(intercepts_ab)):

                intercepts_abi = intercepts_ab[i]

                for j in range(len(intercepts_abi)):

                    intercepts_abij = intercepts_abi[j]

                    if not intercepts_abij:
                        continue

                    i0_input, i0_output, p0 = intercepts_abij[0]

                    i0_input = _arrays.index_3d(i0_input, shape_input)
                    i0_output = _arrays.index_3d(i0_output, shape_output)

                    for t in range(1, len(intercepts_abij)):

                        i1_input, i1_output, p1 = intercepts_abij[t]

                        i1_input = _arrays.index_3d(i1_input, shape_input)
                        i1_output = _arrays.index_3d(i1_output, shape_output)

                        i0_input_lower = rg.math.difference_3d(i0_input, normal_a)
                        i0_input_upper = i0_input

                        i0_output_lower = rg.math.difference_3d(i0_output, normal_b)
                        i0_output_upper = i0_output

                        apex_input = (
                            x_input[i0_input],
                            y_input[i0_input],
                            z_input[i0_input],
                        )
                        apex_output = (
                            x_output[i0_output],
                            y_output[i0_output],
                            z_output[i0_output],
                        )

                        vol_input = rg.geometry.volume_tetrahedron(apex_input, p0, p1)
                        vol_output = rg.geometry.volume_tetrahedron(apex_output, p0, p1)

                        volume_input_lower = volume_input[i0_input_lower]
                        volume_input_upper = volume_input[i0_output_upper]

                        input_lower_in_bounds = _arrays.index_in_bounds(
                            index=i0_input_lower,
                            shape=shape_centers_input,
                        )
                        input_upper_in_bounds = _arrays.index_in_bounds(
                            index=i0_input_upper,
                            shape=shape_centers_input,
                        )
                        output_lower_in_bounds = _arrays.index_in_bounds(
                            index=i0_output_lower,
                            shape=shape_centers_output,
                        )
                        output_upper_in_bounds = _arrays.index_in_bounds(
                            index=i0_output_upper,
                            shape=shape_centers_output,
                        )

                        i0_input_lower = _arrays.index_flat(
                            index=i0_input_lower,
                            shape=shape_input,
                        )
                        i0_input_upper = _arrays.index_flat(
                            index=i0_input_upper,
                            shape=shape_input,
                        )
                        i0_output_lower = _arrays.index_flat(
                            index=i0_output_lower,
                            shape=shape_output,
                        )
                        i0_output_upper = _arrays.index_flat(
                            index=i0_output_upper,
                            shape=shape_output,
                        )

                        if input_lower_in_bounds:
                            if output_lower_in_bounds:
                                weight_lower_lower = (
                                    i0_input_lower,
                                    i0_output_lower,
                                    (-vol_input - vol_output) / volume_input_lower,
                                )
                                weights.append(weight_lower_lower)

                            if output_upper_in_bounds:
                                weight_lower_upper = (
                                    i0_input_lower,
                                    i0_output_upper,
                                    (-vol_input + vol_output) / volume_input_lower,
                                )
                                weights.append(weight_lower_upper)

                        if input_upper_in_bounds:
                            if output_lower_in_bounds:
                                weight_upper_lower = (
                                    i0_input_upper,
                                    i0_output_lower,
                                    (vol_input - vol_output) / volume_input_upper,
                                )
                                weights.append(weight_upper_lower)

                            if output_upper_in_bounds:
                                weight_upper_upper = (
                                    i0_input_upper,
                                    i0_output_upper,
                                    (vol_input + vol_output) / volume_input_upper,
                                )
                                weights.append(weight_upper_upper)

                        p0 = p1
                        i0_input = i1_input
                        i0_output = i1_output

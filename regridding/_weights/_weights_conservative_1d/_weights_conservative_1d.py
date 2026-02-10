import sys
import numpy as np
import numba
from . import _grids

__all__ = [
    "weights_conservative_1d",
]


def weights_conservative_1d(
    x_input: np.ndarray,
    x_output: np.ndarray,
    weights: np.ndarray,
    index_start: int,
    index_stop: int,
) -> numba.typed.List[tuple[int, int, float]]:
    """
    For each cell of `grid_output`,
    compute the fraction of volume shared with each cell of `grid_input`
    and save as a list of weights.

    Parameters
    ----------
    x_input
        The vertices of the old grid.
        Must be a 2D array where the last axis represents the
        grid and the first axis represents a stack of independent grids.
    x_output
        The vertices of the new grid.
        Must be a 2D array where the last axis represents the
        grid and the first axis represents a stack of independent grids.
    weights
        An array of weights to update.
        Must be a 1D array of objects.
    index_start
        The first index in the stack to update.
    index_stop
        The last index in the stack to update.
    """
    for i in range(index_start, index_stop):
        weights[i] = _weights_conservative_1d(
            x_input=x_input[i],
            x_output=x_output[i],
        )


@numba.njit(cache=True)
def _weights_conservative_1d(
    x_input: np.ndarray,
    x_output: np.ndarray,
) -> numba.typed.List[tuple[int, int, float]]:
    """
    For each cell of `grid_output`,
    compute the fraction of volume shared with each cell of `grid_input`
    and save as a list of weights.

    Parameters
    ----------
    x_input
        The vertices of the old grid.
        Must be a 1D monotonic array.
    x_output
        The vertices of the new grid.
        Must be a 1D monotonic array.
    """

    x_sweep = x_input
    x_static = x_output

    num_sweep, = x_sweep.shape
    num_static, = x_static.shape

    weights = numba.typed.List()
    for x in range(0):  # pragma: nocover
        weights.append((0, 0, 0.0))

    x_sweep_left = x_sweep[0]
    x_sweep_right = x_sweep[~0]

    x_static_left = x_static[0]
    x_static_right = x_static[~0]

    if x_sweep_left < x_sweep_right:
        reversed_sweep = False
    else:
        reversed_sweep = True
        x_sweep = x_sweep[::-1]

    if x_static_left < x_static_right:
        reversed_static = False
    else:
        reversed_static = True
        x_static = x_static[::-1]

    reversed_input = reversed_sweep
    reversed_output = reversed_static

    x_static_left = x_static[0]
    x_static_right = x_static[~0]

    length_input = _grids.cell_length(x_sweep)

    index_sweep = 0

    point_1 = x_sweep[index_sweep]

    if x_static_left == point_1:
        sweep_is_outside_static = False
        index_static = 0
    elif x_static_left < point_1 < x_static_right:
        sweep_is_outside_static = False
        index_static = _grids.index_of_point(
            point=point_1,
            grid=x_static,
        )
        index_static = index_static - 1
    else:
        sweep_is_outside_static = True
        index_static = sys.maxsize

    while index_sweep < (num_sweep - 1):

        index_sweep_new = index_sweep + 1

        point_2 = x_sweep[index_sweep_new]

        line = point_1, point_2

        if sweep_is_outside_static:

            line, index_sweep, index_static = _step_outside_static(
                line=line,
                index_sweep=index_sweep,
                index_static=index_static,
                grid_static=x_static,
            )

            if index_static < sys.maxsize:
                sweep_is_outside_static = False

        else:

            line, index_sweep, index_static = _step_inside_static(
                line=line,
                index_sweep=index_sweep,
                index_static=index_static,
                grid_static=x_static,
                length_input=length_input,
                weights=weights,
                reversed_input=reversed_input,
                reversed_output=reversed_output
            )

            if not (0 <= index_static < (num_static - 1)):
                break

        point_1 = line[1]

    return weights


@numba.njit(cache=True)
def _step_outside_static(
    line: tuple[float, float],
    index_sweep: int,
    index_static: int,
    grid_static: np.ndarray,
) -> tuple[
    tuple[float, float],
    int,
    int,
]:
    """
    Check if the current line segment crosses into the boundary of the static
    grid.
    If it does, return the point of intersection and the index of the static
    grid where the crossing occurs.

    Parameters
    ----------
    line
        The current line segment of the sweep grid
    index_sweep
        The index of the current vertex in the sweep grid
    index_static
        The index of the current vertex in the static grid.
        Since this function is only called outside the static grid,
        this will usually be an invalid index.
    grid_static
        The vertices of the static grid.
    """

    point_1, point_2 = line

    point_static = grid_static[0]

    if point_1 < point_static < point_2:
        index_static = 0
        point_2 = point_static
    elif point_static == point_2:
        index_sweep = index_sweep + 1
        index_static = 0
    else:
        index_sweep = index_sweep + 1

    return (point_1, point_2), index_sweep, index_static


@numba.njit(cache=True)
def _step_inside_static(
    line: tuple[float, float],
    index_sweep: int,
    index_static: int,
    grid_static: np.ndarray,
    length_input: np.ndarray,
    weights: numba.typed.List[tuple[int, int, float]],
    reversed_input: bool,
    reversed_output: bool,
) -> tuple[
    tuple[float, float],
    int,
    int,
]:
    """
    Check if the current line segment crosses any vertex of the current
    cell in the static grid.
    If it does, update the line segment and compute the index of the new
    point in the static grid.
    Otherwise, increment the index of the sweep grid.
    In any case, add the current line segment to the list of weights

    Parameters
    ----------
    line
        The current line segment of the sweep grid
    index_sweep
        The index of the current vertex in the sweep grid
    index_static
        The index of the current vertex in the static grid.
        Since this function is only called outside the static grid,
        this will usually be an invalid index.
    grid_static
        The vertices of the static grid.
    length_input
        The length of each cell in the current input grid.
    weights
        The current list of weights to which new weights will be appended.
    """

    point_1, point_2 = line

    index_input = index_sweep
    index_output = index_static

    if reversed_input:
        index_input = ~index_input

    if reversed_output:
        index_output = ~index_output

    point_static = grid_static[index_static + 1]

    if point_1 < point_static < point_2:
        index_static = index_static + 1
        point_2 = point_static
    elif point_static == point_2:
        index_static = index_static + 1
        index_sweep = index_sweep + 1
    else:
        index_sweep = index_sweep + 1

    length = point_2 - point_1

    ratio = length / length_input[index_input]

    weight = (index_input, index_output, ratio)

    weights.append(weight)

    line = point_1, point_2

    return line, index_sweep, index_static

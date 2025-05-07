import pytest
import numpy as np
import numba
from . import _intercepts


@pytest.mark.parametrize(
    argnames="line, point, result_expected",
    argvalues=[
        (
            ((-1, -1, 0), (1, 1, 0)),
            (-1, 1, 0),
            (0, 0, 0),
        )
    ],
)
def test_line_point_closest_approach(
    line: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    point: tuple[float, float, float],
    result_expected: tuple[float, float, float],
):
    t = _intercepts._line_point_closest_approach_parameter(line, point)

    result = _intercepts._line_point_closest_approach(line, t)

    assert result == result_expected


@pytest.mark.parametrize(
    argnames="grid_input,grid_output,volume_input",
    argvalues=[
        (
            np.meshgrid(
                np.arange(3),
                np.arange(4),
                np.arange(5),
                indexing="ij",
            ),
            np.meshgrid(
                np.arange(3) + 0.5,
                np.arange(4) + 0.5,
                np.arange(5) + 0.5,
                indexing="ij",
            ),
            np.ones((2, 3, 4)),
        )
    ],
)
def test_sweep(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
):

    weights = _test_sweep_compiled(grid_input, grid_output, volume_input)

    assert len(weights) > 0


@numba.njit
def _test_sweep_compiled(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume_input: np.ndarray,
):
    intercepts = _intercepts.empty(
        shape_input=grid_input[0].shape,
        shape_output=grid_output[0].shape,
    )

    intercepts[0][0][0][0].append((0, 0, (0.05, 0.15, 0.35)))
    intercepts[0][0][0][0].append((0, 1, (0.85, 0.25, 0.25)))

    weights = numba.typed.List()
    for _ in range(0):
        weights.append((0, 0, 0.))

    _intercepts.sweep(
        intercepts=intercepts,
        weights=weights,
        grid_input=grid_input,
        grid_output=grid_output,
        volume_input=volume_input,
    )

    return weights

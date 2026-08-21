from typing import Literal
import pytest
import numpy as np
import regridding


@pytest.mark.parametrize(
    argnames="coordinates_input,coordinates_output",
    argvalues=[
        (
            (np.linspace(-1, 1, num=32),),
            (np.linspace(-1, 1, num=64),),
        ),
        (
            (np.linspace(-1, 1, num=32),),
            (np.linspace(-2, 2, num=64),),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "brute",
        "searchsorted",
        pytest.param("invalid method", marks=pytest.mark.xfail),
    ],
)
def test_find_indices_1d(
    coordinates_input: tuple[np.ndarray],
    coordinates_output: tuple[np.ndarray],
    method: Literal["brute", "searchsorted"],
):
    result = regridding.find_indices(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        method=method,
    )

    (coordinates_input_x,) = coordinates_input
    (coordinates_output_x,) = coordinates_output
    (result_x,) = result

    where = coordinates_output_x <= coordinates_input_x.max()
    where &= coordinates_output_x > coordinates_input_x.min()

    assert np.all(
        coordinates_input_x[result_x[where] + 0] <= coordinates_output_x[where]
    )
    assert np.all(
        coordinates_input_x[result_x[where] + 1] >= coordinates_output_x[where]
    )


@pytest.mark.parametrize("method", ["brute", "searchsorted"])
def test_find_indices_outside_grid(method: str):
    """
    An output point outside the input grid must be marked with `fill_value`.

    The `searchsorted` finder used to store its result in an `int32` array,
    which cannot hold the default `numpy.iinfo(int).max` sentinel: it
    truncated to -1, a valid negative index that silently addressed the last
    vertex of the grid. A point above the grid was not caught at all, since
    the bound was compared against the number of vertices rather than against
    the largest usable cell index.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x_output = np.array([-1.0, 0.0, 0.5, 3.5, 4.0, 5.0])

    fill_value = np.iinfo(int).max

    (result,) = regridding.find_indices(
        coordinates_input=(x,),
        coordinates_output=(x_output,),
        method=method,
    )

    expected = np.array([fill_value, 0, 0, 3, 3, fill_value])

    assert np.array_equal(result, expected)


def test_find_indices_outside_grid_fill_value():
    """A caller-supplied `fill_value` should be used verbatim."""
    x = np.array([0.0, 1.0, 2.0])
    (result,) = regridding.find_indices(
        coordinates_input=(x,),
        coordinates_output=(np.array([-1.0, 0.5, 7.0]),),
        method="searchsorted",
        fill_value=-99,
    )
    assert np.array_equal(result, np.array([-99, 0, -99]))

"""
Weights which interpolate an array sampled on a logically-rectangular grid
of any number of dimensions at any set of points.
"""

from typing import Literal, Sequence
import numpy as np
import numba
from regridding import _util
from . import _cells, _rectilinear, _curvilinear

__all__ = [
    "weights_multilinear",
]


def _locate_curvilinear(
    grids: tuple[np.ndarray, ...],
    points: np.ndarray,
    extrapolate: bool,
    num_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the cell of a curvilinear grid which contains each point, for each
    element.

    Each grid is searched through its own hierarchy of bounding boxes.  If
    every element shares one grid, the points of every element are searched
    for together.

    Parameters
    ----------
    grids
        The coordinates of the vertices of the grid of each element, each an
        array with an axis for the elements, of length one if every element
        shares the grid, followed by an axis for each axis of the grid.
    points
        The points to find, of shape
        ``(number of elements, N, number of points)``.
    extrapolate
        Whether to find the cell to extrapolate from for a point outside the
        grid.
    num_chunk
        The number of pieces to divide the work into.
    """
    num_element, num_axis, num_point = points.shape
    num_grid = grids[0].shape[0]
    shape = np.array(grids[0].shape[1:], dtype=np.int64)

    cell = np.empty((num_element, num_point, num_axis), dtype=np.int64)
    xi = np.empty((num_element, num_point, num_axis), dtype=np.float64)
    inside = np.empty((num_element, num_point), dtype=np.bool_)

    for g in range(num_grid):
        vertices = np.stack([np.ravel(c[g]) for c in grids], dtype=np.float64)
        tree = _curvilinear.tree(vertices, shape, num_chunk)
        if num_grid == 1:
            index = slice(None)
            num_row = num_element
            points_g = points.transpose(1, 0, 2).reshape(
                num_axis, num_element * num_point
            )
        else:
            index = slice(g, g + 1)
            num_row = 1
            points_g = points[g]
        cell_g, xi_g, inside_g = _curvilinear.locate(
            vertices,
            shape,
            *tree,
            np.ascontiguousarray(points_g),
            extrapolate,
            num_chunk,
        )
        cell[index] = cell_g.reshape(num_row, num_point, num_axis)
        xi[index] = xi_g.reshape(num_row, num_point, num_axis)
        inside[index] = inside_g.reshape(num_row, num_point)

    return cell, xi, inside


def weights_multilinear(
    coordinates_input: tuple[np.ndarray, ...],
    coordinates_output: tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    weights_input: None | np.ndarray = None,
    bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    perturb: None | bool = False,
    seed: "None | int | np.random.Generator" = _util._seed_default,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """
    Weights which interpolate an array sampled on the vertices of a
    logically-rectangular grid at a set of points.

    The grid may have any number of dimensions, and the points any number of
    axes, which do not have to match.  Each point is interpolated from the
    corners of the cell which contains it, so it has :math:`2^N` weights.

    A rectilinear grid is searched along each axis on its own, and any other
    grid through a hierarchy of bounding boxes.  The two find the same
    cells, but the first is exact and needs nothing built.  The elements
    along the orthogonal axes are all searched in the same call, so that
    many small grids, such as a spectrum for each pixel of an image, cost
    little more than one large one.

    Parameters
    ----------
    coordinates_input
        The coordinates of the vertices of the input grid.
    coordinates_output
        The coordinates of the points.
    axis_input
        The logical axes of the input grid.
    axis_output
        The logical axes of the points.  If :obj:`None`, every axis of the
        points, and there must be as many as in `axis_input`, which keeps a
        forgotten `axis_output` from passing for points arranged in a
        different number of axes.
    weights_input
        Weights to multiply each vertex of the input grid by.
    bounds
        How to treat a point outside the input grid.
    perturb
        Whether to perturb the points by a small random amount.
    seed
        The seed of the random perturbation.
    """

    if bounds not in ("extrapolate", "nan", "raise"):
        raise ValueError(
            f"Unrecognized {bounds=}, "
            f"expected one of ('extrapolate', 'nan', 'raise')."
        )

    if perturb is None:
        perturb = False

    (
        coordinates_input,
        coordinates_output,
        axis_input_,
        axis_output_,
        shape_input,
        shape_output,
        shape_orthogonal,
    ) = _util._normalize_input_output_coordinates(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=axis_input,
        axis_output=axis_output,
        perturb=perturb,
        seed=seed,
        match_axes=axis_output is None,
    )

    shape_input_broadcasted = coordinates_input[0].shape
    num_orthogonal = len(shape_orthogonal)
    num_element = int(np.prod(shape_orthogonal, dtype=int))

    # Move the resampled axes behind the orthogonal ones, in ascending order,
    # which is the order of the flat indices of the weights.
    source_input = sorted(axis_input_)
    source_output = sorted(axis_output_)
    destination_input = range(-len(source_input), 0)
    destination_output = range(-len(source_output), 0)
    coordinates_input = tuple(
        np.moveaxis(c, source_input, destination_input) for c in coordinates_input
    )
    coordinates_output = tuple(
        np.moveaxis(c, source_output, destination_output) for c in coordinates_output
    )

    shape = np.array(coordinates_input[0].shape[num_orthogonal:], dtype=np.int64)
    if np.any(shape < 2):
        raise ValueError(
            f"The input grid must have at least two vertices along each of "
            f"its resampled axes, got {tuple(shape)}"
        )

    num_point = int(np.prod(coordinates_output[0].shape[num_orthogonal:], dtype=int))

    result = np.empty(shape_orthogonal, dtype=object)
    if num_element == 0:
        return result, shape_input, shape_output

    # The input grid is usually broadcast across the orthogonal axes, so
    # that every element shares it and it is searched once.  A zero stride,
    # or a single element, guarantees that.
    shared = all(
        c.strides[ax] == 0 or c.shape[ax] == 1
        for c in coordinates_input
        for ax in range(num_orthogonal)
    )
    if shared:
        grids = tuple(c[(0,) * num_orthogonal][np.newaxis] for c in coordinates_input)
    else:
        grids = tuple(np.reshape(c, (num_element, *shape)) for c in coordinates_input)

    points = np.stack(
        [np.reshape(c, (num_element, num_point)) for c in coordinates_output],
        axis=1,
        dtype=np.float64,
    )

    num_chunk = numba.get_num_threads()

    rectilinear = _rectilinear.profiles(grids)
    if rectilinear is not None:
        cell, xi, inside = _rectilinear.locate(*rectilinear, points)
    else:
        cell, xi, inside = _locate_curvilinear(
            grids=grids,
            points=points,
            extrapolate=bounds == "extrapolate",
            num_chunk=num_chunk,
        )

    if bounds == "raise" and not np.all(inside):
        raise ValueError(
            f"{np.count_nonzero(~inside)} of the output points fall outside "
            f"the input grid, and {bounds=}."
        )

    if weights_input is None:
        weights_input = np.empty((0, 0), dtype=np.float64)
    else:
        num_vertex = int(np.prod(shape))
        weights_input = np.broadcast_to(weights_input, shape_input_broadcasted)
        weights_input = np.moveaxis(weights_input, source_input, destination_input)
        if all(
            weights_input.strides[ax] == 0 or weights_input.shape[ax] == 1
            for ax in range(num_orthogonal)
        ):
            weights_input = np.reshape(
                weights_input[(0,) * num_orthogonal], (1, num_vertex)
            )
        else:
            weights_input = np.reshape(weights_input, (num_element, num_vertex))
        weights_input = np.ascontiguousarray(weights_input, dtype=np.float64)

    indices_input, indices_output, values = _cells.weights(
        cell,
        xi,
        inside,
        shape,
        weights_input,
        bounds == "nan",
    )

    flat = result.reshape(-1)
    for e in range(num_element):
        flat[e] = (indices_input[e], indices_output[e], values[e])

    return result, shape_input, shape_output

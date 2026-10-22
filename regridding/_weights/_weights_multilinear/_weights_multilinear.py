"""
Weights which interpolate an array sampled on a logically-rectangular grid
of any number of dimensions at any set of points.
"""

from typing import Callable, Literal, Sequence
import numpy as np
import numba
from regridding import _util
from . import _cells, _rectilinear, _curvilinear

__all__ = [
    "weights_multilinear",
]


def _index(
    index: tuple[int, ...],
    axis: tuple[int, ...],
) -> tuple[int | slice, ...]:
    """
    The index of one orthogonal element of an array, which takes the element
    `index` along the orthogonal axes and the whole of the resampled axes.

    Parameters
    ----------
    index
        The index of the element along the orthogonal axes.
    axis
        The resampled axes, as negative numbers.
    """
    result: list[int | slice] = list(reversed(index))
    for ax in sorted(axis, reverse=True):
        result.insert(~ax, slice(None))
    return tuple(reversed(result))


def _locator(
    coordinates: tuple[np.ndarray, ...],
    num_chunk: int,
) -> Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare to find the cells of a grid which contain a set of points.

    A rectilinear grid is searched along each axis on its own, and any other
    grid through a hierarchy of bounding boxes, which is built here.  The
    two give the same cells, but the first is exact and needs nothing built.

    Returns a function of the points, of shape ``(N, number of points)``,
    and of whether to extrapolate, which returns the cell of each point, its
    local coordinates in that cell, and whether it is inside the grid.

    Parameters
    ----------
    coordinates
        The coordinates of the vertices of the grid, each an array with an
        axis for each axis of the grid.
    num_chunk
        The number of pieces to divide the work into.
    """
    rectilinear = _rectilinear.profiles(coordinates)

    if rectilinear is not None:
        profile, start, coordinate, descending = rectilinear

        def locate_rectilinear(
            points: np.ndarray,
            extrapolate: bool,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # the search along each axis finds the cell to extrapolate from
            # either way
            del extrapolate
            return _rectilinear.locate(profile, start, coordinate, descending, points)

        return locate_rectilinear

    shape = np.array(coordinates[0].shape, dtype=np.int64)
    vertices = np.stack([np.ravel(c) for c in coordinates]).astype(np.float64)
    lower, upper, start, shapes = _curvilinear.tree(vertices, shape, num_chunk)

    def locate_curvilinear(
        points: np.ndarray,
        extrapolate: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _curvilinear.locate(
            vertices,
            shape,
            lower,
            upper,
            start,
            shapes,
            points,
            extrapolate,
            num_chunk,
        )

    return locate_curvilinear


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

    Parameters
    ----------
    coordinates_input
        The coordinates of the vertices of the input grid.
    coordinates_output
        The coordinates of the points.
    axis_input
        The logical axes of the input grid.
    axis_output
        The logical axes of the points.
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
        match_axes=False,
    )

    shape_input_broadcasted = coordinates_input[0].shape

    shape = np.array(
        [shape_input_broadcasted[ax] for ax in sorted(axis_input_)],
        dtype=np.int64,
    )
    if np.any(shape < 2):
        raise ValueError(
            f"The input grid must have at least two vertices along each of "
            f"its resampled axes, got {tuple(shape)}"
        )

    if weights_input is not None:
        weights_input = np.broadcast_to(weights_input, shape_input_broadcasted)

    # The input grid is usually broadcast across the orthogonal axes, so
    # every element shares it and the work of preparing to search it is done
    # once.  A zero stride, or a single element, guarantees that.
    axis_input_orthogonal = tuple(
        ax for ax in range(-len(shape_input_broadcasted), 0) if ax not in axis_input_
    )
    shared = all(
        coordinate.strides[ax] == 0 or coordinate.shape[ax] == 1
        for coordinate in coordinates_input
        for ax in axis_input_orthogonal
    )

    num_chunk = numba.get_num_threads()

    result = np.empty(shape_orthogonal, dtype=object)
    locate = None

    for index in np.ndindex(*shape_orthogonal):

        index_input = _index(index, axis_input_)
        index_output = _index(index, axis_output_)

        if locate is None or not shared:
            locate = _locator(
                tuple(c[index_input] for c in coordinates_input),
                num_chunk,
            )

        points = np.stack(
            [np.ravel(c[index_output]) for c in coordinates_output],
        ).astype(np.float64)

        cell, xi, inside = locate(points, bounds == "extrapolate")

        if bounds == "raise" and not np.all(inside):
            raise ValueError(
                f"{np.count_nonzero(~inside)} of the output points fall outside "
                f"the input grid, and {bounds=}."
            )

        if weights_input is None:
            weights_input_index = np.empty(0, dtype=np.float64)
        else:
            weights_input_index = np.ravel(weights_input[index_input]).astype(
                np.float64
            )

        result[index] = _cells.weights(
            cell,
            xi,
            inside,
            shape,
            weights_input_index,
            bounds == "nan",
        )

    return result, shape_input, shape_output

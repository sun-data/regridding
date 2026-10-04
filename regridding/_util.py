from typing import Any, Sequence
import numpy as np

_seed_default = 42
"""
The default seed used to perturb the output coordinates.

Fixed so that repeated calls on the same grids return identical results.
"""


def _unbroadcast(
    arrays: tuple[np.ndarray, ...],
    axis: tuple[int, ...] = (),
) -> tuple[np.ndarray, ...]:
    """
    Undo broadcasting arrays.

    An axis along which every array has been broadcast, which is to say has
    a stride of zero, is reduced to a length of one, as a view.  The strides
    rather than any shape the arrays were given with are what is looked at,
    so arrays which a caller broadcast before passing them are found as well.

    Parameters
    ----------
    arrays
        Arrays of the same shape, such as the coordinates of a grid.
    axis
        Axes to leave whole even if they have been broadcast, such as the
        resampled axes of an array defined on the cells of a grid, which is
        needed on all of them.
    """
    ndim = arrays[0].ndim
    whole = [ax % ndim for ax in axis]
    index = tuple(
        (
            slice(0, 1)
            if i not in whole and all(a.strides[i] == 0 for a in arrays)
            else slice(None)
        )
        for i in range(ndim)
    )
    return tuple(a[index] for a in arrays)


def _dimensionless(
    a: Any,
    strict: bool = True,
    name: str = "the array",
) -> np.ndarray:
    """
    Reduce an array to the plain numbers it stands for, in double precision.

    `regridding` does not depend on `astropy`, so a
    :class:`astropy.units.Quantity` is recognized by duck typing.  A
    dimensionless one, such as a percentage, is scaled to the number it
    stands for.  Its value is converted to double precision before it is
    scaled, so that a single-precision percentage is scaled exactly as a
    double-precision one is.

    An array which has been broadcast is converted as it was before being
    broadcast, and broadcast again, rather than copied out to its full
    shape, so a caller need not undo the broadcasting first.

    Parameters
    ----------
    a
        An array, which may be a quantity.
    strict
        Whether a unit with dimensions raises.  If :obj:`False`, it is
        dropped instead, leaving the value in that unit.
    name
        What to call `a` if it raises.

    Raises
    ------
    ValueError
        If `strict` is set and `a` has a unit with dimensions.
    """
    unit = getattr(a, "unit", None)
    value = getattr(a, "value", a)

    shape = None
    if isinstance(value, np.ndarray):
        shape = value.shape
        (value,) = _unbroadcast((value,))

    value = np.asarray(value, dtype=np.float64)

    if unit is not None:
        try:
            scale = unit.to("")
        except (TypeError, ValueError) as error:
            if strict:
                raise ValueError(f"{name} must be dimensionless, got {unit}") from error
            scale = 1
        if scale != 1:
            value = value * scale

    if shape is not None and value.shape != shape:
        value = np.broadcast_to(value, shape)

    return value


def _normalize_axis(
    axis: None | int | Sequence[int],
    ndim: int,
) -> tuple[int, ...]:
    if axis is None:
        axis = tuple(range(ndim))
    axis = np.lib.array_utils.normalize_axis_tuple(axis, ndim=ndim)
    axis = tuple(~(~np.array(axis) % ndim))
    return axis


def _normalize_input_output_coordinates(
    coordinates_input: np.ndarray | tuple[np.ndarray, ...],
    coordinates_output: np.ndarray | tuple[np.ndarray, ...],
    axis_input: None | int | Sequence[int] = None,
    axis_output: None | int | Sequence[int] = None,
    perturb: bool = False,
    seed: "None | int | np.random.Generator" = _seed_default,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    if isinstance(coordinates_input, np.ndarray):
        coordinates_input = (coordinates_input,)

    if isinstance(coordinates_output, np.ndarray):
        coordinates_output = (coordinates_output,)

    # If the output coordinates carry a unit, convert the input coordinates to
    # match before the two are compared. `regridding` does not depend on
    # `astropy`, so a united quantity is recognized by duck typing.
    coords_input = []
    for coord_input, coord_output in zip(coordinates_input, coordinates_output):
        unit_output = getattr(coord_output, "unit", None)
        if unit_output is not None:
            coord_input = coord_input << unit_output
        coords_input.append(coord_input)
    coordinates_input = tuple(coords_input)

    shape_coordinates_input = np.broadcast(*coordinates_input).shape
    shape_coordinates_output = np.broadcast(*coordinates_output).shape

    ndim_input = len(shape_coordinates_input)
    ndim_output = len(shape_coordinates_output)

    axis_input = _normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _normalize_axis(axis_output, ndim=ndim_output)

    axis_input = tuple(sorted(axis_input, reverse=True))
    axis_output = tuple(sorted(axis_output, reverse=True))

    if len(axis_output) != len(axis_input):
        raise ValueError(
            f"The number of axes in `axis_output`, {axis_output}, "
            f"must match the number of axes in `axis_input`, {axis_input}"
        )

    if len(coordinates_input) != len(axis_input):
        raise ValueError(
            f"The number of elements in `coordinates_input`, {len(coordinates_input)}, "
            f"should match the number of axes in `axis_input`, {axis_input}"
        )

    if len(coordinates_output) != len(coordinates_input):
        raise ValueError(
            f"The number of elements in `coordinates_output`, {len(coordinates_output)}, "
            f"should match the number of elements in `coordinates_input`, {len(coordinates_input)}"
        )

    axis_input_orthogonal = tuple(
        ax for ax in _normalize_axis(None, ndim_input) if ax not in axis_input
    )
    axis_output_orthogonal = tuple(
        ax for ax in _normalize_axis(None, ndim_output) if ax not in axis_output
    )

    shape_input_orthogonal = tuple(
        shape_coordinates_input[ax] for ax in axis_input_orthogonal
    )
    shape_output_orthogonal = tuple(
        shape_coordinates_output[ax] for ax in axis_output_orthogonal
    )

    shape_orthogonal = np.broadcast_shapes(
        shape_input_orthogonal, shape_output_orthogonal
    )

    shape_input = list(reversed(shape_orthogonal))
    for ax in axis_input:
        shape_input.insert(~ax, shape_coordinates_input[ax])
    shape_input = tuple(reversed(shape_input))

    shape_output = list(reversed(shape_orthogonal))
    for ax in axis_output:
        shape_output.insert(~ax, shape_coordinates_output[ax])
    shape_output = tuple(reversed(shape_output))

    coordinates_input = tuple(
        np.broadcast_to(coord, shape_input) for coord in coordinates_input
    )
    coordinates_output = tuple(
        np.broadcast_to(coord, shape_output) for coord in coordinates_output
    )

    if perturb:
        epsilon = 1e-9
        rng = np.random.default_rng(seed)
        _coordinates_output = []
        for coord in coordinates_output:
            ptp = np.ptp(coord, axis=axis_output, keepdims=True)
            coord = rng.normal(coord, ptp * epsilon)
            _coordinates_output.append(coord)
        coordinates_output = tuple(_coordinates_output)

    return (
        coordinates_input,
        coordinates_output,
        axis_input,
        axis_output,
        shape_coordinates_input,
        shape_coordinates_output,
        shape_orthogonal,
    )

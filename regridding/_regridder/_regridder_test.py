from typing import Any
import dataclasses
import gc
import weakref
import pytest
import numpy as np
import astropy.units as u
import regridding

Weights = tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]


def _grids(
    layout: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[int, ...],
    tuple[int, ...],
]:
    """
    Two orthogonal axes of input grids, rotated along the first and scaled
    along the second, and one output lattice which all of them are resampled
    onto.

    With the ``"trailing"`` layout, the resampled axes come last.  With the
    ``"interleaved"`` layout, they alternate with the orthogonal axes, and
    differently in the input and the output.
    """
    t = np.linspace(-4, 4, 9)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.array([0.1, 0.7])[:, np.newaxis, np.newaxis, np.newaxis]
    scale = np.array([1, 0.8, 0.9])[np.newaxis, :, np.newaxis, np.newaxis]
    grid_input = (
        scale * (x * np.cos(angle) - y * np.sin(angle)),
        scale * (x * np.sin(angle) + y * np.cos(angle)),
    )

    x, y = np.meshgrid(np.linspace(-6, 6, 13), np.linspace(-6, 5, 12), indexing="ij")
    grid_output = (x[np.newaxis, np.newaxis], y[np.newaxis, np.newaxis])

    if layout == "trailing":
        return grid_input, grid_output, (2, 3), (2, 3)

    grid_input = (np.moveaxis(grid_input[0], 2, 1), np.moveaxis(grid_input[1], 2, 1))
    grid_output = (
        np.moveaxis(grid_output[0], (2, 3), (0, 2)),
        np.moveaxis(grid_output[1], (2, 3), (0, 2)),
    )
    return grid_input, grid_output, (1, 3), (0, 2)


def _orthogonal(ndim: int, axis: tuple[int, ...]) -> tuple[int, ...]:
    """The axes of an array with `ndim` axes which are not in `axis`."""
    return tuple(ax for ax in range(ndim) if ax not in axis)


def _shape(
    shape: tuple[int, ...],
    axis: tuple[int, ...],
    orthogonal: tuple[int, ...],
) -> tuple[int, ...]:
    """`shape` with its orthogonal axes replaced by the lengths `orthogonal`."""
    result = list(shape)
    for ax, n in zip(_orthogonal(len(shape), axis), orthogonal):
        result[ax] = n
    return tuple(result)


def _sum(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `a` along the axes `shape` has a length of one along and it does not."""
    axis = tuple(ax for ax in range(a.ndim) if shape[ax] == 1 and a.shape[ax] > 1)
    return a.sum(axis=axis, keepdims=True)


roles = dict(
    diagonal=((2, 3), (2, 3)),
    shared=((1, 3), (2, 3)),
    summed=((2, 3), (2, 1)),
    ctis=((1, 3), (2, 1)),
    both=((1, 3), (1, 3)),
    all=((1, 1), (1, 1)),
)
"""
The lengths of the input and output along the two orthogonal axes, along
which the weights have lengths of 2 and 3, to test each role an orthogonal
axis can have.  ``"ctis"`` shares the input along the first, as a scene is
shared by the channels, and sums the second, as an image sums wavelengths.
"""


@pytest.mark.parametrize(argnames="layout", argvalues=["trailing", "interleaved"])
@pytest.mark.parametrize(argnames="role", argvalues=list(roles))
class TestRoles:
    """Each role an orthogonal axis can have, in each layout of the axes."""

    def _regridder(
        self,
        layout: str,
        role: str,
    ) -> tuple[regridding.Regridder, Weights, tuple[Any, ...]]:
        """The operator for `role`, the weights it is built from, and the grids."""
        grid_input, grid_output, axis_input, axis_output = _grids(layout)
        weights = regridding.weights(
            grid_input,
            grid_output,
            axis_input=axis_input,
            axis_output=axis_output,
            method="conservative",
        )
        default = regridding.Regridder.from_weights(weights, axis_input, axis_output)
        orthogonal_input, orthogonal_output = roles[role]
        regridder = regridding.Regridder.from_weights(
            weights,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_input=_shape(default.shape_input, axis_input, orthogonal_input),
            shape_output=_shape(default.shape_output, axis_output, orthogonal_output),
        )
        grids = (grid_input, grid_output, axis_input, axis_output)
        return regridder, weights, grids

    def test_call(self, layout: str, role: str) -> None:
        """
        The operator gives what resampling each element of the weights does,
        once the input is broadcast and the output summed.  Applying each
        element to its own input is the same calculation in the same order,
        so it gives exactly the same numbers.
        """
        regridder, weights, (_, _, axis_input, axis_output) = self._regridder(
            layout, role
        )
        default = regridding.Regridder.from_weights(weights, axis_input, axis_output)

        values = np.random.default_rng(1).normal(size=regridder.shape_input)
        result = regridder(values)

        expected = regridding.regrid_from_weights(
            *weights,
            values_input=np.broadcast_to(values, default.shape_input),
            axis_input=axis_input,
            axis_output=axis_output,
        )
        expected = _sum(expected, regridder.shape_output)

        assert result.shape == regridder.shape_output
        if role == "diagonal":
            assert np.array_equal(result, expected)
        else:
            assert np.allclose(result, expected, rtol=1e-13, atol=1e-15)

    def test_transpose(self, layout: str, role: str) -> None:
        """
        The transpose swaps the shapes, is the adjoint of the operator, and
        transposes back to the operator itself.
        """
        regridder, _, _ = self._regridder(layout, role)
        transposed = regridder.T

        assert transposed.shape_input == regridder.shape_output
        assert transposed.shape_output == regridder.shape_input
        assert transposed.T is regridder

        rng = np.random.default_rng(2)
        x = rng.normal(size=regridder.shape_input)
        y = rng.normal(size=regridder.shape_output)
        assert np.isclose(np.vdot(regridder(x), y), np.vdot(x, transposed(y)))

    def test_transpose_conservative(self, layout: str, role: str) -> None:
        """
        The conservative transpose gives what the conservatively transposed
        weights do, once broadcast and summed the other way around.  Without
        any sharing or summing, it gives exactly the same numbers.
        """
        regridder, weights, grids = self._regridder(layout, role)
        grid_input, grid_output, axis_input, axis_output = grids

        if role in ("both", "all"):
            # the input grids vary along both orthogonal axes, so an axis
            # which is shared and then summed loses which cells an entry has
            with pytest.raises(NotImplementedError, match="is lost"):
                regridder.transpose_conservative(grid_input, grid_output)
            return

        result = regridder.transpose_conservative(grid_input, grid_output)
        assert result.shape_input == regridder.shape_output
        assert result.shape_output == regridder.shape_input
        assert result.indices is regridder.T.indices

        weights_transposed = regridding.transpose_weights_conservative(
            weights,
            grid_input,
            grid_output,
            axis_input=axis_input,
            axis_output=axis_output,
        )
        default = regridding.Regridder.from_weights(
            weights_transposed, axis_output, axis_input
        )

        values = np.random.default_rng(3).normal(size=result.shape_input)
        expected = regridding.regrid_from_weights(
            *weights_transposed,
            values_input=np.broadcast_to(values, default.shape_input),
            axis_input=axis_output,
            axis_output=axis_input,
        )
        expected = _sum(expected, result.shape_output)

        if role == "diagonal":
            assert np.array_equal(result(values), expected)
        else:
            assert np.allclose(result(values), expected, rtol=1e-13, atol=1e-15)


def _weights_trailing() -> tuple[Weights, tuple[Any, ...]]:
    """The weights between the trailing grids, and the grids."""
    grid_input, grid_output, axis_input, axis_output = _grids("trailing")
    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )
    return weights, (grid_input, grid_output, axis_input, axis_output)


def test_batch() -> None:
    """
    An orthogonal axis the weights do not vary along is a batch axis, which
    the operator is applied along, as it is along any leading axes of the
    values.
    """
    grid_input, grid_output, axis_input, axis_output = _grids("trailing")
    grid_input = tuple(c[:, :1] for c in grid_input)
    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )
    regridder = regridding.Regridder.from_weights(
        weights,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=(2, 4, 8, 8),
        shape_output=(2, 4, 12, 11),
    )

    values = np.random.default_rng(4).normal(size=(5, 2, 4, 8, 8))
    result = regridder(values)

    expected = regridding.regrid_from_weights(
        *weights,
        values_input=values,
        axis_input=axis_input,
        axis_output=axis_output,
    )

    assert np.array_equal(result, expected)


def test_broadcast() -> None:
    """Values which broadcast to the input, like a scalar, are accepted."""
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)

    result = regridder(np.ones((8, 8)))

    assert np.array_equal(result, regridder(np.ones(regridder.shape_input)))
    assert np.array_equal(result, regridder(1))


dtypes = [
    (np.float64, np.float64, np.float64),
    (np.float32, np.float64, np.float64),
    (np.float64, np.float32, np.float64),
    (np.float32, np.float32, np.float32),
    (np.int32, np.float32, np.float64),
]
"""
The types of the values and of the matrix, and of the product they give, to
test the precision of the products.
"""


def _ctis(dtype: type) -> regridding.Regridder:
    """
    The ``"ctis"`` operator of :class:`TestRoles`, which shares its input
    and sums its output, with a matrix of type `dtype`.
    """
    operator, weights, (_, _, axis_input, axis_output) = TestRoles()._regridder(
        "interleaved", "ctis"
    )
    return regridding.Regridder.from_weights(
        weights,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=operator.shape_input,
        shape_output=operator.shape_output,
        dtype=dtype,
    )


def _double(regridder: regridding.Regridder) -> regridding.Regridder:
    """`regridder` with the same entries, held in double precision."""
    return dataclasses.replace(regridder, data=regridder.data.astype(np.float64))


@pytest.mark.parametrize(
    argnames="dtype_values,dtype_matrix,dtype_result",
    argvalues=dtypes,
)
def test_call_dtype(
    dtype_values: type,
    dtype_matrix: type,
    dtype_result: type,
) -> None:
    """
    The product is in single precision if both the values and the matrix
    are, and in double precision otherwise.  Each row is summed in double
    precision, where the product of two numbers in single precision is
    exact, so the result is the product of the same numbers in double
    precision, rounded once.
    """
    regridder = _ctis(dtype_matrix)
    rng = np.random.default_rng(8)

    for operator in (regridder, regridder.T):
        values = 10 * rng.normal(size=(3,) + operator.shape_input)
        values = values.astype(dtype_values)

        result = operator(values)
        expected = _double(operator)(values.astype(np.float64))

        assert result.dtype == dtype_result
        assert np.array_equal(result, expected.astype(dtype_result))


def _with_unit(weights: Weights, unit: Any) -> Weights:
    """`weights` with `unit` attached to the values of every element."""
    array, shape_input, shape_output = weights
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(*array.shape):
        indices_input, indices_output, values = array[index]
        result[index] = (indices_input, indices_output, values * unit)
    return result, shape_input, shape_output


def test_units() -> None:
    """The units of the values and of the weights multiply."""
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    weights_unit = _with_unit(weights, u.s)

    regridder = regridding.Regridder.from_weights(weights_unit, axis_input, axis_output)
    assert regridder.unit == u.s

    values = u.Quantity(
        np.random.default_rng(5).normal(size=regridder.shape_input), u.ph
    )
    result = regridder(values)

    assert result.unit.is_equivalent(u.ph * u.s)
    expected = regridding.regrid_from_weights(
        *weights,
        values_input=values.value,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    assert np.array_equal(result.to_value(u.ph * u.s), expected)


@pytest.mark.parametrize(
    argnames="weights_input",
    argvalues=[
        np.random.default_rng(6).uniform(0.5, 1.5, size=(8, 8)),
        np.random.default_rng(7).uniform(0.5, 1.5, size=(2, 3, 8, 8)),
    ],
    ids=["shared", "varying"],
)
def test_transpose_conservative_weights_input(weights_input: np.ndarray) -> None:
    """The weights of the input cells are inverted as the weights do it."""
    grid_input, grid_output, axis_input, axis_output = _grids("trailing")
    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
        weights_input=weights_input,
    )
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    result = regridder.transpose_conservative(
        grid_input, grid_output, weights_input=weights_input
    )

    weights_transposed = regridding.transpose_weights_conservative(
        weights,
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
    )

    values = np.random.default_rng(8).normal(size=result.shape_input)
    expected = regridding.regrid_from_weights(
        *weights_transposed,
        values_input=values,
        axis_input=axis_output,
        axis_output=axis_input,
    )

    assert np.array_equal(result(values), expected)


@pytest.mark.parametrize(
    argnames="kwargs, match",
    argvalues=[
        (dict(shape_input=(3, 3, 8, 8)), "either 1 or 2"),
        (dict(shape_output=(2, 2, 12, 11)), "either 1 or 3"),
        (dict(shape_input=(2, 3, 8, 7)), "resampled axis 3"),
        (dict(shape_output=(2, 3, 12, 11, 1)), "should have 4 axes"),
        (dict(device="cpu"), "not 'cpu'"),
    ],
)
def test_from_weights_invalid(kwargs: dict[str, Any], match: str) -> None:
    """Shapes which do not fit the weights, and unknown devices, raise."""
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    kwargs = dict(axis_input=axis_input, axis_output=axis_output) | kwargs
    with pytest.raises(ValueError, match=match):
        regridding.Regridder.from_weights(weights, **kwargs)


def test_from_weights_batch_invalid() -> None:
    """Along an axis the weights do not vary along, nothing can be summed."""
    grid_input, grid_output, axis_input, axis_output = _grids("trailing")
    grid_input = tuple(c[:, :1] for c in grid_input)
    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )
    with pytest.raises(ValueError, match="the same length along it"):
        regridding.Regridder.from_weights(
            weights,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_input=(2, 4, 8, 8),
            shape_output=(2, 1, 12, 11),
        )


def test_to_invalid() -> None:
    """An unknown device raises, rather than being taken for the GPU."""
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    with pytest.raises(ValueError, match="not 'cuda:1'"):
        regridder.to("cuda:1")


@pytest.mark.parametrize(
    argnames="shape_points",
    argvalues=[(7,), (2, 3, 4)],
)
def test_multilinear(shape_points: tuple[int, ...]) -> None:
    """
    Multilinear weights onto scattered points, whose output has a different
    number of resampled axes from the input, give what
    :func:`regridding.regrid_from_weights` does, and transpose.
    """
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 2, 6), indexing="ij")
    rng = np.random.default_rng(14)
    points = (
        rng.uniform(0, 1, size=shape_points),
        rng.uniform(0, 2, size=shape_points),
    )
    axis_input = (0, 1)
    axis_output = tuple(range(len(shape_points)))
    weights = regridding.weights(
        (x, y),
        points,
        axis_input=axis_input,
        axis_output=axis_output,
        method="multilinear",
    )
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)

    values = x**2 + y
    expected = regridding.regrid_from_weights(
        *weights,
        values_input=values,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    assert np.array_equal(regridder(values), expected)

    output = rng.normal(size=shape_points)
    assert np.isclose(
        np.vdot(regridder(values), output),
        np.vdot(values, regridder.T(output)),
    )


def _weights_empty() -> Weights:
    """Weights with no elements, from a grid with an orthogonal axis of length 0."""
    grid_input = np.broadcast_to(np.linspace(0, 10, 11), (0, 11))
    grid_output = np.linspace(-1, 11, 7)
    return regridding.weights(
        (grid_input,),
        (grid_output,),
        axis_input=-1,
        axis_output=-1,
        method="conservative",
    )


def test_empty() -> None:
    """
    Weights with no elements give an operator with no entries, which can
    still be applied.
    """
    regridder = regridding.Regridder.from_weights(_weights_empty(), -1, -1)

    assert regridder.shape_matrix == (0, 0)
    assert regridder(np.ones(regridder.shape_input)).shape == (0, 6)
    assert regridder.T(np.ones(regridder.shape_output)).shape == (0, 10)


def test_transpose_freed() -> None:
    """
    An operator and its transpose do not keep each other alive, so dropping
    the operator frees it without waiting for a collection of cycles, and
    its transpose computes the operator again if it is asked for.
    """
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    transposed = regridder.T
    assert transposed.T is regridder

    reference = weakref.ref(regridder)
    enabled = gc.isenabled()
    gc.disable()
    try:
        del regridder
        assert reference() is None
    finally:
        if enabled:
            gc.enable()

    # transposed twice, each row holds its entries in the order of the
    # columns rather than of the elements, so it sums them in another order
    again = transposed.T
    assert again.T is transposed
    values = np.random.default_rng(15).normal(size=again.shape_input)
    expected = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    assert np.allclose(again(values), expected(values), rtol=1e-13, atol=1e-15)


def test_from_weights_single() -> None:
    """
    Weights in single precision give a matrix of their own values, in either
    precision.
    """
    grid_input, grid_output, axis_input, axis_output = _grids("trailing")
    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
        dtype_values=np.float32,
    )
    double = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    single = regridding.Regridder.from_weights(
        weights, axis_input, axis_output, dtype=np.float32
    )

    assert double.data.dtype == np.float64
    assert single.data.dtype == np.float32
    assert np.array_equal(single.data.astype(np.float64), double.data)


def test_call_invalid() -> None:
    """Values which do not broadcast to the input raise."""
    weights, (_, _, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(
        weights,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=(1, 3, 8, 8),
    )
    with pytest.raises(ValueError, match="cannot be broadcast"):
        regridder(np.ones((2, 3, 8, 8)))


def test_transpose_conservative_grids_invalid() -> None:
    """Grids which do not have the cells of the weights raise."""
    weights, (grid_input, grid_output, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    with pytest.raises(ValueError, match="cells"):
        regridder.transpose_conservative(
            tuple(c[..., :-1] for c in grid_input),
            grid_output,
        )


@pytest.mark.parametrize(
    argnames="unit",
    argvalues=[u.percent, u.s],
    ids=["percent", "seconds"],
)
def test_transpose_conservative_units(unit: Any) -> None:
    """
    A unitless unit with a scale, such as a percentage, scales the
    conservative transpose by the number it stands for, and any other unit
    is dropped, as `transpose_weights_conservative()` does.
    """
    weights, (grid_input, grid_output, axis_input, axis_output) = _weights_trailing()
    weights = _with_unit(weights, unit)

    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    result = regridder.transpose_conservative(grid_input, grid_output)

    weights_transposed = regridding.transpose_weights_conservative(
        weights,
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
    )
    values = np.random.default_rng(9).normal(size=result.shape_input)
    expected = regridding.regrid_from_weights(
        *weights_transposed,
        values_input=values,
        axis_input=axis_output,
        axis_output=axis_input,
    )

    assert result.unit is None
    assert np.array_equal(result(values), expected)


def test_transpose_conservative_orthogonal_invalid() -> None:
    """Grids whose orthogonal axes do not fit the weights raise."""
    weights, (grid_input, grid_output, axis_input, axis_output) = _weights_trailing()
    regridder = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    grid_input = tuple(np.concatenate([c, c[:, :1]], axis=1) for c in grid_input)
    with pytest.raises(ValueError, match="do not fit"):
        regridder.transpose_conservative(grid_input, grid_output)

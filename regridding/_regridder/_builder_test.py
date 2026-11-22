from typing import Any, NoReturn
import dataclasses
import functools
import pytest
import numpy as np
import astropy.units as u
import regridding
from regridding._regridder import _builder as _builder_module
from regridding._regridder import _regridder_test
from regridding._regridder._cuda_test import requires_cuda

Weights = _regridder_test.Weights

Index = tuple[slice, ...]

splits: dict[str, list[Index]] = dict(
    whole=[(slice(0, 2), slice(0, 3))],
    first=[(slice(0, 1), slice(0, 3)), (slice(1, 2), slice(0, 3))],
    second=[(slice(0, 2), slice(0, 2)), (slice(0, 2), slice(2, 3))],
    elements=[(slice(i, i + 1), slice(j, j + 1)) for i in range(2) for j in range(3)],
)
"""
Ways to cut weights with lengths of 2 and 3 along their two orthogonal axes
into pieces, listed in the order they are counted and placed: whole, across
the first axis, across the second, and into single elements.
"""


@dataclasses.dataclass(frozen=True)
class _Case:
    """
    The weights of one role and layout of the tests of
    :class:`~regridding.Regridder`, with exact zeros in them, and their grids.
    """

    weights: Weights
    """The weights, whose input cells are partly blocked by a field stop."""

    grid_input: tuple[np.ndarray, ...]
    """The vertices of the input grid."""

    grid_output: tuple[np.ndarray, ...]
    """The vertices of the output grid."""

    axis_input: tuple[int, ...]
    """The resampled axes of the input."""

    axis_output: tuple[int, ...]
    """The resampled axes of the output."""

    weights_input: np.ndarray
    """
    The weights of the input cells, which the conservative transpose inverts,
    without the field stop, as a model of an instrument inverts its effective
    area but not its field stop.
    """

    shape_input: tuple[int, ...]
    """The shape of the arrays the operator accepts."""

    shape_output: tuple[int, ...]
    """The shape of the arrays the operator returns."""


def _stopped(weights: Weights) -> Weights:
    """
    `weights` with the entries of the first two rows of input cells set to
    exactly zero, as a field stop folded into the weights of the input sets
    them.

    The flat index of an input cell counts the resampled axes in ascending
    order, so its row along the first of them is the index over the 8 cells
    of a row.
    """
    array, shape_input, shape_output = weights
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(*array.shape):
        indices_input, indices_output, values = array[index]
        values = np.where(indices_input // 8 < 2, 0.0, values)
        result[index] = (indices_input, indices_output, values)
    return result, shape_input, shape_output


@functools.cache
def _case(role: str, layout: str) -> _Case:
    """
    The weights and grids of `role` in `layout`, see :class:`_Case`.

    Each case is computed once, since computing the weights is most of the
    time these tests take with the JIT of Numba disabled, so no test may
    change one in place.
    """
    grid_input, grid_output, axis_input, axis_output = _regridder_test._grids(layout)

    weights_input = np.random.default_rng(16).uniform(0.5, 1.5, size=(2, 3, 8, 8))
    if layout == "interleaved":
        weights_input = np.moveaxis(weights_input, 2, 1)

    weights = regridding.weights(
        grid_input,
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )
    weights = _stopped(weights)

    default = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    orthogonal_input, orthogonal_output = _regridder_test.roles[role]

    return _Case(
        weights=weights,
        grid_input=grid_input,
        grid_output=grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        weights_input=weights_input,
        shape_input=_regridder_test._shape(
            default.shape_input, axis_input, orthogonal_input
        ),
        shape_output=_regridder_test._shape(
            default.shape_output, axis_output, orthogonal_output
        ),
    )


def _case_uniform(role: str) -> _Case:
    """
    The weights and grids of `role` in the trailing layout, see
    :class:`_Case`, with the same input grid, and the same weights of the
    input cells, for every element.
    """
    case = _case(role, "trailing")
    grid_input = tuple(np.broadcast_to(c[:1, :1], c.shape) for c in case.grid_input)
    weights = regridding.weights(
        grid_input,
        case.grid_output,
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        method="conservative",
    )
    weights_input = case.weights_input
    return dataclasses.replace(
        case,
        weights=_stopped(weights),
        grid_input=grid_input,
        weights_input=np.broadcast_to(weights_input[:1, :1], weights_input.shape),
    )


def _drop_zeros(weights: Weights) -> Weights:
    """`weights` with the slots of their exact zeros emptied."""
    array, shape_input, shape_output = weights
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(*array.shape):
        indices_input, indices_output, values = array[index]
        zero = np.asarray(values == 0)
        result[index] = (
            np.where(zero, -1, indices_input),
            np.where(zero, -1, indices_output),
            values,
        )
    return result, shape_input, shape_output


def _piece(weights: Weights, index: Index) -> Weights:
    """The elements `index` of `weights`."""
    array, shape_input, shape_output = weights
    return array[index], shape_input, shape_output


def _start(index: Index) -> tuple[int, ...]:
    """The index of the first element of a piece."""
    return tuple(int(s.start) for s in index)


def _piece_grid(a: np.ndarray, axis: tuple[int, ...], index: Index) -> np.ndarray:
    """
    The part of an array on the vertices or cells of a grid with the resampled
    axes `axis` which the elements `index` of the weights use.
    """
    slices: list[Any] = [slice(None)] * a.ndim
    for ax, s in zip(_regridder_test._orthogonal(a.ndim, axis), index):
        if a.shape[ax] > 1:
            slices[ax] = s
    return a[tuple(slices)]


def _builder(
    case: _Case,
    conservative: bool = False,
    dtype: type = np.float64,
) -> regridding.RegridderBuilder:
    """A builder of the operator of `case`."""
    return regridding.RegridderBuilder(
        shape_input=case.shape_input,
        shape_output=case.shape_output,
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        shape_weights=case.weights[0].shape,
        conservative=conservative,
        dtype=dtype,
    )


def _grids(case: _Case, index: Index) -> dict[str, Any]:
    """The grids of the elements `index` of the weights of `case`."""
    return dict(
        coordinates_input=tuple(
            _piece_grid(c, case.axis_input, index) for c in case.grid_input
        ),
        coordinates_output=tuple(
            _piece_grid(c, case.axis_output, index) for c in case.grid_output
        ),
        weights_input=_piece_grid(case.weights_input, case.axis_input, index),
    )


def _build(
    case: _Case,
    pieces: list[Index],
    conservative: bool = False,
    dtype: type = np.float64,
) -> tuple[regridding.Regridder, None | regridding.Regridder]:
    """Build the operator of `case` from `pieces` of its weights."""
    builder = _builder(case, conservative=conservative, dtype=dtype)
    for index in pieces:
        builder.count(_piece(case.weights, index), start=_start(index))
    for index in pieces:
        grids = _grids(case, index) if conservative else {}
        builder.place(_piece(case.weights, index), start=_start(index), **grids)
    return builder.build()


def _expected(case: _Case, dtype: type = np.float64) -> regridding.Regridder:
    """The operator of `case`, from all of its weights with their zeros dropped."""
    return regridding.Regridder.from_weights(
        _drop_zeros(case.weights),
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        shape_input=case.shape_input,
        shape_output=case.shape_output,
        dtype=dtype,
    )


def _expected_conservative(
    case: _Case,
    expected: regridding.Regridder,
) -> regridding.Regridder:
    """The conservative transpose of `expected`, from the grids of all its weights."""
    return expected.transpose_conservative(
        case.grid_input,
        case.grid_output,
        weights_input=case.weights_input,
    )


def _transposed_elements(
    case: _Case,
    values: np.ndarray,
    shape_output: tuple[int, ...],
) -> np.ndarray:
    """
    What the conservative transposes of the elements of the weights of
    `case` give, each from the cells of its own element, for `values`
    broadcast to every element, summed along the axes `shape_output` has a
    length of one along.
    """
    weights = regridding.transpose_weights_conservative(
        case.weights,
        case.grid_input,
        case.grid_output,
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        weights_input=case.weights_input,
    )
    default = regridding.Regridder.from_weights(
        weights, case.axis_output, case.axis_input
    )
    result = regridding.regrid_from_weights(
        *weights,
        values_input=np.broadcast_to(values, default.shape_input),
        axis_input=case.axis_output,
        axis_output=case.axis_input,
    )
    return _regridder_test._sum(result, shape_output)


def _assert_same(
    result: None | regridding.Regridder,
    expected: regridding.Regridder,
) -> None:
    """Check that `result` is `expected`, bit for bit."""
    assert result is not None
    for name in (
        "shape_input",
        "shape_output",
        "axis_input",
        "axis_output",
        "shape_weights",
        "unit",
        "device",
    ):
        assert getattr(result, name) == getattr(expected, name), name
    for name in ("indptr", "indices", "data"):
        a = getattr(result, name)
        b = getattr(expected, name)
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


def _assert_regridder(
    result: regridding.Regridder,
    expected: regridding.Regridder,
    role: str,
    split: str,
) -> None:
    """
    Check that an operator built from the pieces `split` of the weights of
    `role` is `expected`, bit for bit, unless pieces which put entries in the
    same rows were placed out of the order of their elements.  Then check
    that it gives the same products, to within the rounding of sums taken in
    another order.
    """
    if role == "all" and split == "second":
        # every element sums into the same rows, and the first piece holds
        # elements which come after some of the second's
        assert result.num_entries == expected.num_entries
        values = np.random.default_rng(17).normal(size=result.shape_input)
        assert np.allclose(result(values), expected(values), rtol=1e-13)
        return

    _assert_same(result, expected)


def _out_of_memory(*args: Any, **kwargs: Any) -> NoReturn:
    """Run out of memory, in place of a function which makes room for arrays."""
    raise MemoryError("out of memory, as a test")


@pytest.mark.parametrize(argnames="dtype", argvalues=[np.float64, np.float32])
@pytest.mark.parametrize(argnames="split", argvalues=list(splits))
@pytest.mark.parametrize(argnames="layout", argvalues=["trailing", "interleaved"])
@pytest.mark.parametrize(argnames="role", argvalues=list(_regridder_test.roles))
class TestBuilder:
    """Each role an orthogonal axis can have, in each layout, cut each way."""

    def test_regridder(self, role: str, layout: str, split: str, dtype: type) -> None:
        """
        The operator is the one assembled from all the weights at once with
        their zeros dropped, bit for bit, unless pieces which put entries in
        the same rows are placed out of the order of their elements.  Then it
        gives the same products, to within the rounding of sums taken in
        another order.
        """
        case = _case(role, layout)
        result, transposed = _build(case, splits[split], dtype=dtype)

        assert transposed is None
        _assert_regridder(result, _expected(case, dtype), role, split)

    def test_conservative(
        self,
        role: str,
        layout: str,
        split: str,
        dtype: type,
    ) -> None:
        """
        The conservative transpose is the one computed from the grids of all
        the weights at once, bit for bit.  Where an orthogonal axis is summed
        after the input is shared along it, which
        `Regridder.transpose_conservative()` refuses for these grids, since
        they vary along it, it gives what the conservative transposes of the
        elements give together, to within the rounding of sums taken in
        another order.
        """
        case = _case(role, layout)
        result, transposed = _build(case, splits[split], conservative=True, dtype=dtype)
        expected = _expected(case, dtype)

        _assert_regridder(result, expected, role, split)

        if role in ("both", "all"):
            assert transposed is not None
            values = np.random.default_rng(19).normal(size=transposed.shape_input)
            expected_values = _transposed_elements(
                case, values, transposed.shape_output
            )
            tolerance = 1e-13 if dtype == np.float64 else 1e-6
            assert np.allclose(
                transposed(values),
                expected_values,
                rtol=tolerance,
                atol=tolerance * np.abs(expected_values).max(),
            )
            return

        _assert_same(transposed, _expected_conservative(case, expected))


@pytest.mark.parametrize(argnames="split", argvalues=["whole", "first", "elements"])
@pytest.mark.parametrize(argnames="role", argvalues=["both", "all"])
def test_conservative_shared_summed(role: str, split: str) -> None:
    """
    Where an orthogonal axis is summed after the input is shared along it,
    and the grids do not vary along it, the conservative transpose is the
    one `Regridder.transpose_conservative()` computes, bit for bit.
    """
    case = _case_uniform(role)
    result, transposed = _build(case, splits[split], conservative=True)
    expected = _expected(case)

    _assert_same(result, expected)
    _assert_same(transposed, _expected_conservative(case, expected))


def test_zeros() -> None:
    """
    Dropping the exact zeros leaves fewer entries, and exactly the same
    products.
    """
    case = _case("ctis", "interleaved")
    result, _ = _build(case, splits["first"])
    with_zeros = regridding.Regridder.from_weights(
        case.weights,
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        shape_input=case.shape_input,
        shape_output=case.shape_output,
    )

    assert result.num_entries < with_zeros.num_entries

    values = np.random.default_rng(18).normal(size=result.shape_input)
    assert np.array_equal(result(values), with_zeros(values))


def test_zeros_not_finite() -> None:
    """
    A NaN in an input cell behind the field stop reaches no output, where
    the zeros of the stop make NaN of the outputs they lead to.
    """
    case = _case("diagonal", "trailing")
    result, _ = _build(case, splits["whole"])
    with_zeros = regridding.Regridder.from_weights(
        case.weights,
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        shape_input=case.shape_input,
        shape_output=case.shape_output,
    )

    values = np.ones(result.shape_input)
    # the first row of input cells, which `_stopped` blocks
    values[..., 0, :] = np.nan

    assert np.isnan(with_zeros(values)).any()
    assert np.isfinite(result(values)).all()


def test_reversed() -> None:
    """
    Pieces which put their entries in rows of their own can be counted and
    placed in any order.
    """
    case = _case("ctis", "interleaved")
    result, transposed = _build(case, splits["first"][::-1], conservative=True)
    expected = _expected(case)

    _assert_same(result, expected)
    _assert_same(transposed, _expected_conservative(case, expected))


@pytest.mark.parametrize(
    argnames="unit",
    argvalues=[u.percent, u.s],
    ids=["percent", "seconds"],
)
def test_units(unit: Any) -> None:
    """
    The operator carries the unit of the weights, and the conservative
    transpose is scaled by the number a dimensionless one stands for, as
    :meth:`~regridding.Regridder.transpose_conservative` scales it.
    """
    case = _case("diagonal", "trailing")
    case = dataclasses.replace(
        case,
        weights=_regridder_test._with_unit(case.weights, unit),
    )
    result, transposed = _build(case, splits["elements"], conservative=True)
    expected = _expected(case)

    assert result.unit == unit
    _assert_same(result, expected)
    _assert_same(transposed, _expected_conservative(case, expected))


def test_units_invalid() -> None:
    """A piece with values in a unit other than those before it raises."""
    case = _case("diagonal", "trailing")
    first, second = splits["first"]
    builder = _builder(case)
    builder.count(
        _piece(_regridder_test._with_unit(case.weights, u.s), first),
        start=_start(first),
    )
    with pytest.raises(ValueError, match="pieces before it are in s"):
        builder.count(
            _piece(_regridder_test._with_unit(case.weights, u.m), second),
            start=_start(second),
        )


def test_units_piece_invalid() -> None:
    """A piece whose elements have values in different units raises."""
    case = _case("diagonal", "trailing")
    array, shape_input, shape_output = _regridder_test._with_unit(case.weights, u.s)
    indices_input, indices_output, values = array[0, 1]
    array[0, 1] = (indices_input, indices_output, values.value * u.m)
    (whole,) = splits["whole"]
    with pytest.raises(ValueError, match="should have one unit"):
        _builder(case).count((array, shape_input, shape_output), start=_start(whole))


def test_uncovered() -> None:
    """The elements no piece covers have no entries."""
    case = _case("diagonal", "trailing")
    first, _ = splits["first"]
    result, _ = _build(case, [first])

    array, shape_input, shape_output = _drop_zeros(case.weights)
    for index in np.ndindex(*array.shape):
        if index[0] == 1:
            indices_input, indices_output, values = array[index]
            array[index] = (
                np.full_like(indices_input, -1),
                np.full_like(indices_output, -1),
                values,
            )
    expected = regridding.Regridder.from_weights(
        (array, shape_input, shape_output),
        axis_input=case.axis_input,
        axis_output=case.axis_output,
        shape_input=case.shape_input,
        shape_output=case.shape_output,
    )

    _assert_same(result, expected)


def test_no_orthogonal() -> None:
    """Weights with no orthogonal axes are one piece, which starts nowhere."""
    t = np.linspace(-4, 4, 9)
    grid_input = tuple(np.meshgrid(t, t, indexing="ij"))
    grid_output = tuple(
        np.meshgrid(np.linspace(-6, 6, 13), np.linspace(-6, 5, 12), indexing="ij")
    )
    weights = regridding.weights(
        grid_input,
        grid_output,
        method="conservative",
    )
    expected = regridding.Regridder.from_weights(_drop_zeros(weights))

    builder = regridding.RegridderBuilder(
        shape_input=expected.shape_input,
        shape_output=expected.shape_output,
        axis_input=None,
        axis_output=None,
        shape_weights=(),
        conservative=True,
    )
    builder.count(weights, start=())
    builder.place(
        weights,
        start=(),
        coordinates_input=grid_input,
        coordinates_output=grid_output,
    )
    result, transposed = builder.build()

    _assert_same(result, expected)
    _assert_same(transposed, expected.transpose_conservative(grid_input, grid_output))


def test_empty() -> None:
    """Weights with no elements, given in no pieces, give operators with no entries."""
    expected = regridding.Regridder.from_weights(
        _regridder_test._weights_empty(), -1, -1
    )
    builder = regridding.RegridderBuilder(
        shape_input=expected.shape_input,
        shape_output=expected.shape_output,
        axis_input=-1,
        axis_output=-1,
        shape_weights=expected.shape_weights,
        conservative=True,
    )
    result, transposed = builder.build()

    _assert_same(result, expected)
    _assert_same(transposed, expected.T)


@pytest.mark.parametrize(
    argnames="kwargs, match",
    argvalues=[
        (dict(shape_input=(3, 3, 8, 8)), "either 1 or 2"),
        (dict(shape_output=(2, 2, 12, 11)), "either 1 or 3"),
        (dict(shape_output=(2, 3, 12, 11, 1)), "should have 4 axes"),
        (dict(shape_weights=(2, 3, 1)), "should have 5 axes"),
    ],
)
def test_init_invalid(kwargs: dict[str, Any], match: str) -> None:
    """Shapes which do not fit each other raise, as for `from_weights()`."""
    case = _case("diagonal", "trailing")
    kwargs = (
        dict(
            shape_input=case.shape_input,
            shape_output=case.shape_output,
            axis_input=case.axis_input,
            axis_output=case.axis_output,
            shape_weights=(2, 3),
        )
        | kwargs
    )
    with pytest.raises(ValueError, match=match):
        regridding.RegridderBuilder(**kwargs)


@pytest.mark.parametrize(
    argnames="piece, start, match",
    argvalues=[
        ("first", (0,), "should have 2 orthogonal axes"),
        ("first", (1, 1), "does not fit"),
        ("grid", (0, 0), "resampled axis 3"),
    ],
)
def test_count_invalid(piece: str, start: tuple[int, ...], match: str) -> None:
    """A piece which does not fit the operator raises."""
    case = _case("diagonal", "trailing")
    weights = _piece(case.weights, splits["first"][0])
    if piece == "grid":
        array, shape_input, shape_output = weights
        weights = (array, shape_input[:-1] + (7,), shape_output)
    with pytest.raises(ValueError, match=match):
        _builder(case).count(weights, start=start)


def test_count_overlap() -> None:
    """A piece which covers an element another piece covers raises."""
    case = _case("diagonal", "trailing")
    builder = _builder(case)
    (whole,) = splits["whole"]
    first, _ = splits["first"]
    builder.count(_piece(case.weights, whole), start=_start(whole))
    with pytest.raises(ValueError, match="covers elements"):
        builder.count(_piece(case.weights, first), start=_start(first))


def test_count_after_place() -> None:
    """Placing a piece ends the first pass."""
    case = _case("diagonal", "trailing")
    first, second = splits["first"]
    builder = _builder(case)
    builder.count(_piece(case.weights, first), start=_start(first))
    builder.place(_piece(case.weights, first), start=_start(first))
    with pytest.raises(ValueError, match="ends the first pass"):
        builder.count(_piece(case.weights, second), start=_start(second))


def test_place_invalid() -> None:
    """
    The second pass has to give the pieces of the first in the same order,
    and no more of them.
    """
    case = _case("diagonal", "trailing")
    first, second = splits["first"]
    builder = _builder(case)
    builder.count(_piece(case.weights, first), start=_start(first))
    builder.count(_piece(case.weights, second), start=_start(second))

    with pytest.raises(ValueError, match="same pieces in the same order"):
        builder.place(_piece(case.weights, second), start=_start(second))

    builder.place(_piece(case.weights, first), start=_start(first))
    builder.place(_piece(case.weights, second), start=_start(second))
    with pytest.raises(ValueError, match="have all been placed"):
        builder.place(_piece(case.weights, second), start=_start(second))


def test_place_grids_invalid() -> None:
    """
    A conservative build needs the grids of each piece, and any other refuses
    them.
    """
    case = _case("diagonal", "trailing")
    (whole,) = splits["whole"]

    builder = _builder(case, conservative=True)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    with pytest.raises(ValueError, match="needs the grids"):
        builder.place(_piece(case.weights, whole), start=_start(whole))

    builder = _builder(case)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    with pytest.raises(ValueError, match="not asked for"):
        builder.place(
            _piece(case.weights, whole),
            start=_start(whole),
            **_grids(case, whole),
        )


def _changed(weights: Weights, change: str) -> Weights:
    """
    `weights` with an element whose entries changed between the passes: the
    values of the first element zeroed, an entry of it moved to the row of
    its first entry, or an entry of the last element moved to the last row
    of the matrix.

    The rows of an element follow its output indices, in the same order, so
    the last row of the last element is the last row of the matrix with any
    entries.
    """
    array, shape_input, shape_output = weights
    array = array.copy()
    index = (0, 0) if change != "last" else tuple(n - 1 for n in array.shape)
    indices_input, indices_output, values = array[index]
    if change == "zeroed":
        values = np.zeros_like(values)
    else:
        valid = np.flatnonzero((indices_input >= 0) & (values != 0))
        if change == "moved":
            target = indices_output[valid[0]]
        else:
            target = indices_output[valid].max()
        other = valid[indices_output[valid] != target]
        indices_output = indices_output.copy()
        indices_output[other[~0]] = target
    array[index] = (indices_input, indices_output, values)
    return array, shape_input, shape_output


def test_place_changed() -> None:
    """
    A piece with fewer entries in the second pass than the first raises,
    and the builder cannot go on, so it lets go of the matrix.
    """
    case = _case("diagonal", "trailing")
    (whole,) = splits["whole"]
    builder = _builder(case)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    with pytest.raises(ValueError, match=r"has \d+ entries, but \d+ were counted"):
        builder.place(_changed(case.weights, "zeroed"), start=_start(whole))
    assert builder._matrix is None
    with pytest.raises(ValueError, match="cannot go on"):
        builder.build()


def test_place_overrun() -> None:
    """
    A piece which gives the last row of the matrix more entries than were
    counted for it, with as many in all, raises as it is placed, rather than
    writing past the end of the matrix.
    """
    case = _case("diagonal", "trailing")
    (whole,) = splits["whole"]
    builder = _builder(case)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    with pytest.raises(ValueError, match="gives a row more entries"):
        builder.place(_changed(case.weights, "last"), start=_start(whole))


def test_build_rows_changed() -> None:
    """
    A piece whose entries moved to other rows between the passes, with as
    many in all, is found once every piece has been placed, and the builder
    lets go of the matrix.
    """
    case = _case("diagonal", "trailing")
    (whole,) = splits["whole"]
    builder = _builder(case)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    builder.place(_changed(case.weights, "moved"), start=_start(whole))
    with pytest.raises(ValueError, match="rows of the matrix"):
        builder.build()
    assert builder._matrix is None


def test_place_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A builder which runs out of memory as it makes room for the entries is
    left as it was, so the piece can be placed again.
    """
    case = _case("diagonal", "trailing")
    first, second = splits["first"]
    builder = _builder(case)
    builder.count(_piece(case.weights, first), start=_start(first))
    builder.count(_piece(case.weights, second), start=_start(second))

    with monkeypatch.context() as patch:
        patch.setattr(_builder_module, "_index_dtype", _out_of_memory)
        with pytest.raises(MemoryError):
            builder.place(_piece(case.weights, first), start=_start(first))

    builder.place(_piece(case.weights, first), start=_start(first))
    builder.place(_piece(case.weights, second), start=_start(second))
    result, _ = builder.build()

    _assert_same(result, _expected(case))


def test_build_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A builder which runs out of memory as it transposes the scaled values
    keeps the matrix, so it can build the operators again.
    """
    case = _case("diagonal", "trailing")
    (whole,) = splits["whole"]
    builder = _builder(case, conservative=True)
    builder.count(_piece(case.weights, whole), start=_start(whole))
    builder.place(
        _piece(case.weights, whole),
        start=_start(whole),
        **_grids(case, whole),
    )

    with monkeypatch.context() as patch:
        patch.setattr(regridding.Regridder, "_transpose", _out_of_memory)
        with pytest.raises(MemoryError):
            builder.build()

    result, transposed = builder.build()
    expected = _expected(case)

    _assert_same(result, expected)
    _assert_same(transposed, _expected_conservative(case, expected))


def test_build_invalid() -> None:
    """
    The operators are built once every piece has been placed, and only once.
    """
    case = _case("diagonal", "trailing")
    first, second = splits["first"]
    builder = _builder(case)
    builder.count(_piece(case.weights, first), start=_start(first))
    builder.count(_piece(case.weights, second), start=_start(second))
    builder.place(_piece(case.weights, first), start=_start(first))
    with pytest.raises(ValueError, match="1 of the 2 pieces"):
        builder.build()

    builder.place(_piece(case.weights, second), start=_start(second))
    builder.build()
    with pytest.raises(ValueError, match="built its operators already"):
        builder.build()


@requires_cuda
def test_device_pieces() -> None:
    """
    Pieces of weights built on a device are brought to the host, and give the
    operators which the same weights give from the host.
    """
    from regridding._weights._weights_transposed import _cuda_test

    kwargs: dict[str, Any] = dict(method="conservative", **_cuda_test.grids)
    weights = regridding.weights(device="cuda", **kwargs)
    expected = regridding.Regridder.from_weights(
        _drop_zeros(_cuda_test._to_host(weights)),
        **_cuda_test.axes,
    )

    builder = regridding.RegridderBuilder(
        shape_input=expected.shape_input,
        shape_output=expected.shape_output,
        shape_weights=expected.shape_weights,
        conservative=True,
        **_cuda_test.axes,
    )
    pieces = [(slice(0, 1),), (slice(1, 2),)]
    for index in pieces:
        builder.count(_piece(weights, index), start=_start(index))
    for index in pieces:
        builder.place(
            _piece(weights, index),
            start=_start(index),
            coordinates_input=tuple(
                _piece_grid(c, _cuda_test.axes["axis_input"], index)
                for c in _cuda_test.grid_input
            ),
            coordinates_output=_cuda_test.grid_output,
        )
    result, transposed = builder.build()

    _assert_same(result, expected)
    _assert_same(
        transposed,
        expected.transpose_conservative(_cuda_test.grid_input, _cuda_test.grid_output),
    )

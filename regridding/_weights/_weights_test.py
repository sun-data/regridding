from typing import Any, NoReturn
import pytest
import numpy as np
import astropy.units as u
import regridding
from regridding._weights import _weights_conservative as _conservative

x_input = np.linspace(-1, 1, num=13)
y_input = np.linspace(-1, 1, num=12)
x_input_broadcasted, y_input_broadcasted = np.meshgrid(
    x_input,
    y_input,
    indexing="ij",
)

angle = 0.2
x_output_broadcasted = x_input_broadcasted * np.cos(
    angle
) - y_input_broadcasted * np.sin(angle)
y_output_broadcasted = x_input_broadcasted * np.sin(
    angle
) + y_input_broadcasted * np.cos(angle)

values_input = np.random.default_rng(42).random((12, 11))


def _weights_conservative(
    **kwargs: Any,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    return regridding.weights(
        coordinates_input=(x_input_broadcasted, y_input_broadcasted),
        coordinates_output=(x_output_broadcasted, y_output_broadcasted),
        method="conservative",
        **kwargs,
    )


def _flat(
    weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, ...]:
    """Concatenate the per-element flat arrays into one comparable triple."""
    elements = weights[0].reshape(-1)
    return tuple(np.concatenate([element[i] for element in elements]) for i in range(3))


class TestSeed:
    """
    The output grid of a conservative build is perturbed to break degenerate
    overlaps, so the weights are only reproducible if that perturbation is
    seeded.
    """

    def test_default_is_deterministic(self) -> None:
        result = _flat(_weights_conservative())
        result_expected = _flat(_weights_conservative())

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_int(self) -> None:
        result = _flat(_weights_conservative(seed=12345))
        result_expected = _flat(_weights_conservative(seed=12345))

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_generator(self) -> None:
        result = _flat(_weights_conservative(seed=np.random.default_rng(12345)))
        result_expected = _flat(
            _weights_conservative(seed=np.random.default_rng(12345))
        )

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_different(self) -> None:
        _, _, values = _flat(_weights_conservative(seed=0))
        _, _, values_expected = _flat(_weights_conservative(seed=1))

        # a different seed moves the weights, but only in the last few digits
        assert not np.array_equal(values, values_expected)
        assert np.isclose(values.sum(), values_expected.sum(), rtol=1e-6)

    def test_seed_none(self) -> None:
        _, _, values = _flat(_weights_conservative(seed=None))
        _, _, values_expected = _flat(_weights_conservative(seed=None))

        # an unseeded generator draws a fresh perturbation for every call
        assert not np.array_equal(values, values_expected)

    def test_seed_unperturbed(self) -> None:
        """`seed` is inert when the grid is not perturbed."""
        result = _flat(_weights_conservative(perturb=False, seed=0))
        result_expected = _flat(_weights_conservative(perturb=False, seed=1))

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_regrid_deterministic(self) -> None:
        kwargs = dict(
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
            values_input=values_input,
            method="conservative",
        )

        result = regridding.regrid(**kwargs)
        result_expected = regridding.regrid(**kwargs)

        assert np.array_equal(result, result_expected, equal_nan=True)


class TestBounds:
    """
    Behavior of `regrid` for output points that fall outside the input grid.

    These points used to address the wrong cell of the input grid: the
    out-of-grid sentinel from `find_indices` truncated to -1, so a point below
    the grid was extrapolated from the *last* cell rather than the first.
    """

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = x**2
    x_output = np.array([-0.5, 0.5, 3.5, 4.5])

    def _regrid(self, **kwargs: Any) -> np.ndarray:
        return regridding.regrid(
            coordinates_input=(self.x,),
            coordinates_output=(self.x_output,),
            values_input=self.values,
            method="multilinear",
            **kwargs,
        )

    def test_extrapolate(self) -> None:
        """The default extrapolates from the nearest cell of the input grid."""
        result = self._regrid()
        # Slope 1 in the first cell and 7 in the last.
        expected = np.array([-0.5, 0.5, 12.5, 19.5])
        assert np.allclose(result, expected)

    def test_nan(self) -> None:
        result = self._regrid(bounds="nan")
        assert np.isnan(result[0])
        assert np.isnan(result[~0])
        assert np.allclose(result[1:~0], [0.5, 12.5])

    def test_raise(self) -> None:
        with pytest.raises(ValueError, match="fall outside the input grid"):
            self._regrid(bounds="raise")

    def test_raise_inside_grid(self) -> None:
        """A grid entirely inside the input grid must not raise."""
        result = regridding.regrid(
            coordinates_input=(self.x,),
            coordinates_output=(np.array([0.5, 3.5]),),
            values_input=self.values,
            method="multilinear",
            bounds="raise",
        )
        assert np.allclose(result, [0.5, 12.5])

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized bounds="):
            self._regrid(bounds="foo")


class TestCoalesce:
    """
    The conservative builders emit several fragments per distinct
    ``(input, output)`` pair.  Merging them is an optimization for weights
    that get reused, not a change to what the weights mean.
    """

    def test_fewer_triples(self) -> None:
        """Merging shrinks the result."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))
        assert merged[0].size < raw[0].size

    def test_unique_pairs(self) -> None:
        """Every pair appears exactly once after merging, and only then."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))

        def num_unique(triple: tuple[np.ndarray, ...]) -> int:
            indices_input, indices_output, _ = triple
            pairs = np.stack([indices_input, indices_output], axis=~0)
            return np.unique(pairs, axis=0).shape[0]

        assert num_unique(merged) == merged[0].size
        assert num_unique(raw) < raw[0].size

    def test_same_total_weight(self) -> None:
        """Merging preserves each input cell's total weight exactly."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))

        num = max(raw[0].max(), merged[0].max()) + 1

        total_raw = np.zeros(num)
        np.add.at(total_raw, raw[0], raw[2])
        total_merged = np.zeros(num)
        np.add.at(total_merged, merged[0], merged[2])

        assert np.allclose(total_raw, total_merged)

    def test_same_result_when_applied(self) -> None:
        """Both forms regrid a scene to the same answer."""
        results = []
        for coalesce in (False, True):
            weights, shape_input, shape_output = _weights_conservative(
                coalesce=coalesce,
            )
            results.append(
                regridding.regrid_from_weights(
                    weights=weights,
                    shape_input=shape_input,
                    shape_output=shape_output,
                    values_input=values_input,
                )
            )

        assert np.allclose(results[0], results[1])


class TestClippingApplicable:
    """
    A 2D conservative build uses the clipping kernel when every output grid
    is a uniform, axis-aligned lattice, and the sweep otherwise.
    """

    @staticmethod
    def _grid(
        num_x: int,
        num_y: int,
        nonuniform: bool = False,
        curvilinear: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-1, 1, num_x)
        y = np.linspace(-1, 1, num_y)
        if nonuniform:
            x = np.sign(x) * np.square(x)
        x, y = np.meshgrid(x, y, indexing="ij")
        if curvilinear:
            x = x + 0.1 * y
        return x, y

    def test_uniform_lattice(self) -> None:
        assert _conservative._clipping_applicable(
            coordinates_output=self._grid(6, 7),
            axis_output=(0, 1),
            shape_orthogonal=(),
        )

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(nonuniform=True),
            dict(curvilinear=True),
        ],
    )
    def test_disqualified(self, kwargs: dict) -> None:
        assert not _conservative._clipping_applicable(
            coordinates_output=self._grid(6, 7, **kwargs),
            axis_output=(0, 1),
            shape_orthogonal=(),
        )

    def test_not_2d(self) -> None:
        assert not _conservative._clipping_applicable(
            coordinates_output=self._grid(6, 7),
            axis_output=(0,),
            shape_orthogonal=(),
        )

    def test_curvilinear_output_still_works(self) -> None:
        """The sweep still handles an output grid the clipping kernel can't."""
        grid_input = self._grid(9, 9)
        grid_output = self._grid(6, 7, curvilinear=True)
        weights, shape_input, shape_output = regridding.weights(
            coordinates_input=grid_input,
            coordinates_output=grid_output,
            method="conservative",
        )
        values = np.random.default_rng(0).random(
            (grid_input[0].shape[0] - 1, grid_input[0].shape[1] - 1)
        )
        result = regridding.regrid_from_weights(
            weights=weights,
            shape_input=shape_input,
            shape_output=shape_output,
            values_input=values,
        )
        assert np.all(np.isfinite(result))


class TestDtype:
    """
    The weights are computed in double precision regardless; only the stored
    result is narrowed, which halves what a large operator costs to keep.
    """

    def _triple(self, **kwargs: Any) -> tuple[np.ndarray, ...]:
        return _flat(_weights_conservative(**kwargs))

    def test_default_is_wide(self) -> None:
        indices_input, indices_output, values = self._triple()
        assert indices_input.dtype == np.int64
        assert indices_output.dtype == np.int64
        assert values.dtype == np.float64

    def test_narrowed(self) -> None:
        indices_input, indices_output, values = self._triple(
            dtype_indices=np.int32,
            dtype_values=np.float32,
        )
        assert indices_input.dtype == np.int32
        assert indices_output.dtype == np.int32
        assert values.dtype == np.float32

    def test_indices_only(self) -> None:
        indices_input, _, values = self._triple(dtype_indices=np.int32)
        assert indices_input.dtype == np.int32
        assert values.dtype == np.float64

    def test_values_only(self) -> None:
        indices_input, _, values = self._triple(dtype_values=np.float32)
        assert indices_input.dtype == np.int64
        assert values.dtype == np.float32

    def test_indices_unchanged(self) -> None:
        """Narrowing the indices must not move them."""
        wide = self._triple()
        narrow = self._triple(dtype_indices=np.int32)
        assert np.array_equal(wide[0], narrow[0])
        assert np.array_equal(wide[1], narrow[1])

    def test_conserved(self) -> None:
        """
        Narrowing costs each input cell only single-precision rounding.

        The comparison is against the wide weights rather than against one,
        since a cell at the edge of the output grid is only partly covered
        and its total is legitimately less than one.
        """

        def totals(**kwargs: Any) -> np.ndarray:
            indices_input, _, values = self._triple(**kwargs)
            total = np.zeros(indices_input.max() + 1)
            np.add.at(total, indices_input, values.astype(np.float64))
            return total

        wide = totals()
        narrow = totals(dtype_values=np.float32)

        assert np.allclose(narrow, wide, rtol=1e-6, atol=0)
        # and the fully covered cells still sum to one
        full = np.isclose(wide, 1, atol=1e-12)
        assert full.any()
        assert np.allclose(narrow[full], 1, atol=1e-6)

    def test_same_result_when_applied(self) -> None:
        """Narrowed weights regrid a scene to the same answer."""
        results = []
        for kwargs in (dict(), dict(dtype_indices=np.int32, dtype_values=np.float32)):
            weights, shape_input, shape_output = _weights_conservative(**kwargs)
            results.append(
                regridding.regrid_from_weights(
                    weights=weights,
                    shape_input=shape_input,
                    shape_output=shape_output,
                    values_input=values_input,
                )
            )
        assert np.allclose(results[0], results[1], rtol=1e-6)

    def test_index_overflow(self) -> None:
        """An index which does not fit raises instead of wrapping around."""
        with pytest.raises(ValueError, match="does not fit in int8"):
            _weights_conservative(dtype_indices=np.int8)


def _grids_unit() -> dict[str, Any]:
    x = np.linspace(-1, 1, num=11)
    y = np.linspace(-1, 1, num=11)
    return dict(
        coordinates_input=np.meshgrid(x, y, indexing="ij"),
        coordinates_output=np.meshgrid(1.1 * x + 0.01, 1.2 * y + 0.01, indexing="ij"),
        method="conservative",
    )


@pytest.mark.parametrize("unit", [u.dimensionless_unscaled, u.percent, u.mm])
def test_weights_input_unit_kept(unit: u.UnitBase) -> None:
    """A ``weights_input`` with any unit carries it onto the weights."""
    result, _, _ = regridding.weights(
        weights_input=np.ones((10, 10)) * unit,
        **_grids_unit(),
    )
    assert all(getattr(w[2], "unit", None) == unit for w in result.reshape(-1))


@pytest.mark.cuda
def test_weights_input_dimensionless_quantity_on_device() -> None:
    """
    On a device a dimensionless ``weights_input`` is applied as the number it
    stands for, so that a percentage is applied as a fraction, and the
    weights come back as plain floats.
    """
    weights_input = np.random.default_rng(0).uniform(0.5, 1.5, size=(10, 10))
    kwargs = dict(_grids_unit(), coalesce=False, device="cuda")
    reference, _, _ = regridding.weights(weights_input=weights_input, **kwargs)
    values_reference = reference.reshape(-1)[0][2].copy_to_host()

    for weights_quantity, factor in (
        (weights_input * u.dimensionless_unscaled, 1),
        (weights_input * u.percent, 0.01),
    ):
        result, _, _ = regridding.weights(weights_input=weights_quantity, **kwargs)
        assert np.allclose(
            result.reshape(-1)[0][2].copy_to_host(), factor * values_reference
        )


@pytest.mark.cuda
def test_weights_input_unit_on_device() -> None:
    """A ``weights_input`` with a real unit cannot be built on a device."""
    with pytest.raises(ValueError, match="cannot be built on a device"):
        regridding.weights(
            weights_input=np.ones((10, 10)) * u.mm,
            **dict(_grids_unit(), coalesce=False, device="cuda"),
        )


def _assert_same_weights(
    result: np.ndarray,
    expected: np.ndarray,
    device: bool = False,
) -> None:
    """
    Assert that two arrays of weights have exactly the same indices and
    values for every element, copying them from the device if `device`.
    """
    assert result.shape == expected.shape
    for index in np.ndindex(*expected.shape):
        for a, b in zip(result[index], expected[index]):
            if device:
                a, b = a.copy_to_host(), b.copy_to_host()
            assert np.array_equal(a, b)


def _grids_interleaved(
    curvilinear: bool,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    """
    Input grids of 4 by 4 cells, rotated along the first of two orthogonal
    axes and scaled along the second, a weight for each of their cells, and
    an output lattice, bent if `curvilinear` so that the sweep resamples onto
    it instead of the clipping kernel.

    The second orthogonal axis has as many elements as the input has cells
    along a resampled axis, so that reading the weights of the cells of an
    element with its orthogonal index alone, as if the resampled axes were
    the last ones, finds cells of the right shape, but the wrong ones.
    """
    t = np.linspace(-2, 2, 5)
    x, y = np.meshgrid(t, t, indexing="ij")
    angle = np.array([0.1, 0.7])[:, np.newaxis, np.newaxis, np.newaxis]
    scale = np.linspace(0.8, 1.1, 4)[np.newaxis, :, np.newaxis, np.newaxis]
    grid_input = (
        scale * (x * np.cos(angle) - y * np.sin(angle)),
        scale * (x * np.sin(angle) + y * np.cos(angle)),
    )

    x_output, y_output = np.meshgrid(
        np.linspace(-3, 3, 7),
        np.linspace(-3, 3, 8),
        indexing="ij",
    )
    if curvilinear:
        x_output = x_output + 0.1 * y_output
    grid_output = (x_output[np.newaxis, np.newaxis], y_output[np.newaxis, np.newaxis])

    weights_input = np.random.default_rng(3).uniform(0.5, 1.5, size=(2, 4, 4, 4))

    return grid_input, grid_output, weights_input


@pytest.mark.parametrize(
    argnames="kernel",
    argvalues=[
        "clipping",
        "sweep",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_weights_input_interleaved(kernel: str) -> None:
    """
    The weights of the input cells of each element are read from the cells of
    its own input grid, so moving the resampled axes of the input in among
    its orthogonal axes gives the same weights.
    """
    grid_input, grid_output, weights_input = _grids_interleaved(
        curvilinear=kernel == "sweep",
    )
    kwargs: dict[str, Any] = dict(
        coordinates_output=grid_output,
        axis_output=(2, 3),
        method="conservative",
    )
    if kernel == "cuda":
        kwargs |= dict(coalesce=False, device="cuda")

    trailing, _, _ = regridding.weights(
        coordinates_input=grid_input,
        axis_input=(2, 3),
        weights_input=weights_input,
        **kwargs,
    )

    # orthogonal, resampled, orthogonal, resampled
    order = (0, 2, 1, 3)
    interleaved, _, _ = regridding.weights(
        coordinates_input=tuple(np.transpose(c, order) for c in grid_input),
        axis_input=(1, 3),
        weights_input=np.transpose(weights_input, order),
        **kwargs,
    )

    _assert_same_weights(interleaved, trailing, device=kernel == "cuda")


def _grids_broadcast(
    case: str,
    ndim: int,
    curvilinear: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    The arguments of a conservative build onto three output grids along an
    orthogonal axis, from an input grid which the output grids broadcast
    along that axis, and the same arguments with the input grid broadcast by
    hand.

    The input grid has 4 cells along each of its `ndim` resampled axes.  In
    the ``"shared"`` case it has no orthogonal axis and a weight for each of
    its cells; in ``"length 1"`` it has the orthogonal axis, but only one
    element along it; and in ``"per element"`` it has no orthogonal axis,
    but different weights for each element.  The output grids are lattices
    shifted along the first resampled axis, and bent if `curvilinear`, so
    that the sweep resamples onto them instead of the clipping kernel.
    """
    t = np.linspace(-2, 2, 5)
    grid_input = tuple(np.meshgrid(*ndim * [t], indexing="ij"))

    t_output = (np.linspace(-3, 3, 7), np.linspace(-3, 3, 8))[:ndim]
    grid_output = list(np.meshgrid(*t_output, indexing="ij"))
    if curvilinear:
        grid_output[0] = grid_output[0] + 0.1 * grid_output[1]
    shift = np.array([0, 0.1, 0.2]).reshape((3,) + ndim * (1,))
    grid_output = (grid_output[0] + shift,) + tuple(
        np.broadcast_to(c, shift.shape[:1] + c.shape) for c in grid_output[1:]
    )

    rng = np.random.default_rng(4)
    shape_cells = ndim * (4,)
    if case == "shared":
        weights_input = rng.uniform(0.5, 1.5, size=shape_cells)
    elif case == "length 1":
        grid_input = tuple(c[np.newaxis] for c in grid_input)
        weights_input = rng.uniform(0.5, 1.5, size=(1,) + shape_cells)
    else:
        weights_input = rng.uniform(0.5, 1.5, size=(3,) + shape_cells)

    ndim_input = grid_input[0].ndim
    axis_output = tuple(range(1, ndim + 1))
    kwargs: dict[str, Any] = dict(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        axis_input=tuple(range(ndim_input - ndim, ndim_input)),
        axis_output=axis_output,
        weights_input=weights_input,
        method="conservative",
    )
    kwargs_broadcast = kwargs | dict(
        coordinates_input=tuple(
            np.broadcast_to(c, (3,) + c.shape[-ndim:]) for c in grid_input
        ),
        axis_input=axis_output,
    )

    return kwargs, kwargs_broadcast


@pytest.mark.parametrize(
    argnames="kernel",
    argvalues=[
        "1d",
        "clipping",
        "sweep",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    argnames="case",
    argvalues=["shared", "length 1", "per element"],
)
def test_weights_input_broadcast(case: str, kernel: str) -> None:
    """
    The weights of the input cells are broadcast against the output grid
    along with the input grid, so an input grid which the output grid
    broadcasts gives the same weights as one broadcast by hand.
    """
    kwargs, kwargs_broadcast = _grids_broadcast(
        case=case,
        ndim=1 if kernel == "1d" else 2,
        curvilinear=kernel == "sweep",
    )
    if kernel == "cuda":
        kwargs |= dict(coalesce=False, device="cuda")
        kwargs_broadcast |= dict(coalesce=False, device="cuda")

    result, _, _ = regridding.weights(**kwargs)
    expected, _, _ = regridding.weights(**kwargs_broadcast)

    _assert_same_weights(result, expected, device=kernel == "cuda")


def test_weights_1d_thread_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    An error in one of the threads which build 1D weights is raised, instead
    of leaving the elements of that thread without weights.
    """

    def _fail(**kwargs: Any) -> NoReturn:
        raise RuntimeError("the thread failed")

    monkeypatch.setattr(_conservative, "weights_conservative_1d", _fail)

    with pytest.raises(RuntimeError, match="the thread failed"):
        regridding.weights(
            coordinates_input=(np.linspace(-2, 2, 5),),
            coordinates_output=(np.linspace(-3, 3, 7),),
            method="conservative",
            coalesce=False,
        )

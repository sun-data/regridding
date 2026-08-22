import pytest
import numpy as np
import regridding

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


def _weights_conservative(**kwargs):
    return regridding.weights(
        coordinates_input=(x_input_broadcasted, y_input_broadcasted),
        coordinates_output=(x_output_broadcasted, y_output_broadcasted),
        method="conservative",
        **kwargs,
    )


def _flat(weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]):
    """Concatenate the per-element flat arrays into one comparable triple."""
    elements = weights[0].reshape(-1)
    return tuple(np.concatenate([element[i] for element in elements]) for i in range(3))


class TestSeed:
    """
    The output grid of a conservative build is perturbed to break degenerate
    overlaps, so the weights are only reproducible if that perturbation is
    seeded.
    """

    def test_default_is_deterministic(self):
        result = _flat(_weights_conservative())
        result_expected = _flat(_weights_conservative())

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_int(self):
        result = _flat(_weights_conservative(seed=12345))
        result_expected = _flat(_weights_conservative(seed=12345))

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_generator(self):
        result = _flat(_weights_conservative(seed=np.random.default_rng(12345)))
        result_expected = _flat(
            _weights_conservative(seed=np.random.default_rng(12345))
        )

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_seed_different(self):
        _, _, values = _flat(_weights_conservative(seed=0))
        _, _, values_expected = _flat(_weights_conservative(seed=1))

        # a different seed moves the weights, but only in the last few digits
        assert not np.array_equal(values, values_expected)
        assert np.isclose(values.sum(), values_expected.sum(), rtol=1e-6)

    def test_seed_none(self):
        _, _, values = _flat(_weights_conservative(seed=None))
        _, _, values_expected = _flat(_weights_conservative(seed=None))

        # an unseeded generator draws a fresh perturbation for every call
        assert not np.array_equal(values, values_expected)

    def test_seed_unperturbed(self):
        """`seed` is inert when the grid is not perturbed."""
        result = _flat(_weights_conservative(perturb=False, seed=0))
        result_expected = _flat(_weights_conservative(perturb=False, seed=1))

        for array, array_expected in zip(result, result_expected):
            assert np.array_equal(array, array_expected)

    def test_regrid_deterministic(self):
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

    def _regrid(self, **kwargs):
        return regridding.regrid(
            coordinates_input=(self.x,),
            coordinates_output=(self.x_output,),
            values_input=self.values,
            method="multilinear",
            **kwargs,
        )

    def test_extrapolate(self):
        """The default extrapolates from the nearest cell of the input grid."""
        result = self._regrid()
        # Slope 1 in the first cell and 7 in the last.
        expected = np.array([-0.5, 0.5, 12.5, 19.5])
        assert np.allclose(result, expected)

    def test_nan(self):
        result = self._regrid(bounds="nan")
        assert np.isnan(result[0])
        assert np.isnan(result[~0])
        assert np.allclose(result[1:~0], [0.5, 12.5])

    def test_raise(self):
        with pytest.raises(ValueError, match="fall outside the input grid"):
            self._regrid(bounds="raise")

    def test_raise_inside_grid(self):
        """A grid entirely inside the input grid must not raise."""
        result = regridding.regrid(
            coordinates_input=(self.x,),
            coordinates_output=(np.array([0.5, 3.5]),),
            values_input=self.values,
            method="multilinear",
            bounds="raise",
        )
        assert np.allclose(result, [0.5, 12.5])

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized bounds="):
            self._regrid(bounds="foo")


class TestCoalesce:
    """
    The conservative builders emit several fragments per distinct
    ``(input, output)`` pair.  Merging them is an optimization for weights
    that get reused, not a change to what the weights mean.
    """

    def test_fewer_triples(self):
        """Merging shrinks the result."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))
        assert merged[0].size < raw[0].size

    def test_unique_pairs(self):
        """Every pair appears exactly once after merging, and only then."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))

        def num_unique(triple):
            indices_input, indices_output, _ = triple
            pairs = np.stack([indices_input, indices_output], axis=~0)
            return np.unique(pairs, axis=0).shape[0]

        assert num_unique(merged) == merged[0].size
        assert num_unique(raw) < raw[0].size

    def test_same_total_weight(self):
        """Merging preserves each input cell's total weight exactly."""
        raw = _flat(_weights_conservative(coalesce=False))
        merged = _flat(_weights_conservative(coalesce=True))

        num = max(raw[0].max(), merged[0].max()) + 1

        total_raw = np.zeros(num)
        np.add.at(total_raw, raw[0], raw[2])
        total_merged = np.zeros(num)
        np.add.at(total_merged, merged[0], merged[2])

        assert np.allclose(total_raw, total_merged)

    def test_same_result_when_applied(self):
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

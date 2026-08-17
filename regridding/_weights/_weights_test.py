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

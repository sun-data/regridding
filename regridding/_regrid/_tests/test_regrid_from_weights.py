import pickle

import numpy as np
import pytest

import regridding

x_input = np.linspace(-1, 1, num=11)
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

values_input = np.random.default_rng(42).random((10, 11))


@pytest.mark.parametrize(
    argnames="weights",
    argvalues=[
        regridding.weights(
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
            method="conservative",
        ),
    ],
)
class TestWeightsArrayFormat:
    def test_format(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        for element in weights[0].reshape(-1):
            indices_input, indices_output, values = element
            assert indices_input.ndim == 1
            assert indices_input.shape == indices_output.shape == values.shape
            assert np.issubdtype(indices_input.dtype, np.integer)
            assert np.issubdtype(indices_output.dtype, np.integer)
            assert np.issubdtype(values.dtype, np.floating)

    def test_weights_input_quantity(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        u = pytest.importorskip("astropy.units")

        weights_quantity = regridding.weights(
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
            weights_input=2 * np.ones((10, 11)) * u.cm**2,
            method="conservative",
        )

        result = regridding.regrid_from_weights(
            *weights_quantity,
            values_input=values_input * u.ph,
        )
        result_expected = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        assert result.unit == u.ph * u.cm**2
        assert np.allclose(result.value, 2 * result_expected, rtol=1e-3, atol=1e-6)

    def test_pairs_unique(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        for element in weights[0].reshape(-1):
            indices_input, indices_output, values = element
            key = indices_input * (indices_output.max() + 1) + indices_output
            assert np.unique(key).size == key.size

    def test_pickle(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        unpickled = pickle.loads(pickle.dumps(weights))

        result = regridding.regrid_from_weights(
            *unpickled,
            values_input=values_input,
        )
        result_expected = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        assert np.array_equal(result, result_expected)

    def test_regrid_from_weights(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        result = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )

        result_expected = regridding.regrid(
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
            values_input=values_input,
            method="conservative",
        )
        assert result.shape == result_expected.shape
        # regrid() builds its own weights, but the grid perturbation is seeded,
        # so an identical grid gives a bitwise-identical result
        assert np.array_equal(result, result_expected)

    def test_transpose_weights(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        result = regridding.transpose_weights(weights)

        assert result[1] == weights[2]
        assert result[2] == weights[1]
        for transposed, original in zip(
            result[0].reshape(-1),
            weights[0].reshape(-1),
        ):
            assert transposed[0] is original[1]
            assert transposed[1] is original[0]
            assert transposed[2] is original[2]

    def test_transpose_weights_conservative(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        weights_transposed = regridding.transpose_weights_conservative(
            weights,
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
        )

        assert weights_transposed[1] == weights[2]
        assert weights_transposed[2] == weights[1]

        values_output = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        values_transposed = regridding.regrid_from_weights(
            *weights_transposed,
            values_input=values_output,
        )

        assert values_transposed.shape == values_input.shape
        # the rotated output grid clips corner flux, so the round trip can
        # only lose flux at the boundary, never gain it
        assert values_transposed.sum() <= values_input.sum() * (1 + 1e-6)
        assert values_transposed.sum() >= 0.8 * values_input.sum()

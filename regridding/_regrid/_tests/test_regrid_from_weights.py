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
class TestWeightsArrays:
    def test_weights_to_arrays(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        result = regridding.weights_to_arrays(weights)

        assert result[1] == weights[1]
        assert result[2] == weights[2]
        for element_arrays, element_triples in zip(
            result[0].reshape(-1),
            weights[0].reshape(-1),
        ):
            indices_input, indices_output, values = element_arrays
            assert indices_input.shape[0] == len(element_triples)
            assert indices_output.shape[0] == len(element_triples)
            assert values.shape[0] == len(element_triples)

    def test_weights_from_arrays(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        result = regridding.weights_from_arrays(regridding.weights_to_arrays(weights))

        assert result[1] == weights[1]
        assert result[2] == weights[2]
        for element_result, element_expected in zip(
            result[0].reshape(-1),
            weights[0].reshape(-1),
        ):
            assert list(element_result) == list(element_expected)

    def test_pickle(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        arrays = regridding.weights_to_arrays(weights)

        unpickled = pickle.loads(pickle.dumps(arrays))

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
            *regridding.weights_to_arrays(weights),
            values_input=values_input,
        )
        result_expected = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )
        assert np.array_equal(result, result_expected)

    def test_regrid_from_weights_transposed(
        self,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
    ):
        weights_transposed = regridding.transpose_weights_conservative(
            weights,
            coordinates_input=(x_input_broadcasted, y_input_broadcasted),
            coordinates_output=(x_output_broadcasted, y_output_broadcasted),
        )
        values = regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )

        result = regridding.regrid_from_weights(
            *regridding.weights_to_arrays(weights_transposed),
            values_input=values,
        )
        result_expected = regridding.regrid_from_weights(
            *weights_transposed,
            values_input=values,
        )
        assert np.array_equal(result, result_expected)

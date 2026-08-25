import pytest
import numpy as np
import astropy.units as u
import regridding

try:
    from numba import cuda

    _available = cuda.is_available()
except ImportError:  # pragma: nocover
    _available = False

requires_cuda = pytest.mark.skipif(
    not _available,
    reason="a CUDA device is needed to apply weights on the device",
)


def _grids(num_input: int = 20, num_output: int = 12):
    """A distorted input grid and the lattice it is resampled onto."""
    t = np.linspace(-0.7, 0.7, num_input + 1)
    u_ = t[:, np.newaxis] * np.ones(num_input + 1)
    v = np.ones(num_input + 1)[:, np.newaxis] * t
    angle = 0.3
    x = (u_ * np.cos(angle) - v * np.sin(angle) + 0.15 * u_ * v) * 4 + 6.3
    y = (u_ * np.sin(angle) + v * np.cos(angle) + 0.10 * u_ * u_) * 4 + 6.1
    lattice = (
        np.arange(num_output + 1, dtype=float)[:, np.newaxis] * np.ones(num_output + 1),
        np.ones(num_output + 1)[:, np.newaxis] * np.arange(num_output + 1, dtype=float),
    )
    return (x, y), lattice


def _weights(device: None | str = None):
    grid_input, grid_output = _grids()
    return regridding.weights(
        coordinates_input=grid_input,
        coordinates_output=grid_output,
        method="conservative",
        coalesce=False,
        device=device,
    )


def _scene(num_input: int = 20):
    return np.random.default_rng(42).random((num_input, num_input))


class TestRegridFromWeightsCuda:

    @requires_cuda
    def test_matches_host(self):
        """Applying the weights on the device agrees with the host."""
        scene = _scene()

        expected = regridding.regrid_from_weights(*_weights(), values_input=scene)
        result = regridding.regrid_from_weights(
            *_weights(device="cuda"),
            values_input=scene,
        )

        assert cuda.is_cuda_array(result)
        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-12)

    @requires_cuda
    def test_values_input_on_device(self):
        """The scene may already be on the device, and then is not copied."""
        scene = _scene()

        expected = regridding.regrid_from_weights(*_weights(), values_input=scene)
        result = regridding.regrid_from_weights(
            *_weights(device="cuda"),
            values_input=cuda.to_device(np.ascontiguousarray(scene)),
        )

        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-12)

    @requires_cuda
    def test_values_output(self):
        """A device array may be given to place the result in."""
        scene = _scene()
        weights = _weights(device="cuda")

        expected = regridding.regrid_from_weights(*_weights(), values_input=scene)

        given = cuda.to_device(np.zeros(weights[2]))
        result = regridding.regrid_from_weights(
            *weights,
            values_input=scene,
            values_output=given,
        )

        assert result is given
        assert np.allclose(given.copy_to_host(), expected, rtol=0, atol=1e-12)

    @requires_cuda
    def test_dtype_values(self):
        """Weights built in single precision resample in single precision."""
        grid_input, grid_output = _grids()
        weights = regridding.weights(
            coordinates_input=grid_input,
            coordinates_output=grid_output,
            method="conservative",
            coalesce=False,
            device="cuda",
            dtype_values=np.float32,
        )

        scene = _scene().astype(np.float32)
        result = regridding.regrid_from_weights(*weights, values_input=scene)

        assert result.dtype == np.float32

        expected = regridding.regrid_from_weights(*_weights(), values_input=_scene())
        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-5)


class TestRegridFromWeightsCudaRejected:
    """The cases the device path cannot serve are refused, not mishandled."""

    @requires_cuda
    def test_quantity(self):
        with pytest.raises(ValueError, match="cannot be resampled on a device"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=_scene() << u.electron,
            )

    @requires_cuda
    def test_axis_input_not_trailing(self):
        with pytest.raises(ValueError, match="has to name the last"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=_scene(),
                axis_input=0,
            )

    @requires_cuda
    def test_axis_output_not_trailing(self):
        with pytest.raises(ValueError, match="has to name the last"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=_scene(),
                axis_output=0,
            )

    @requires_cuda
    def test_values_input_wrong_shape(self):
        with pytest.raises(ValueError, match="cannot be broadcast"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=_scene()[:5],
            )

    @requires_cuda
    def test_values_output_wrong_shape(self):
        weights = _weights(device="cuda")
        with pytest.raises(ValueError, match="should be equal to"):
            regridding.regrid_from_weights(
                *weights,
                values_input=_scene(),
                values_output=cuda.to_device(np.zeros((3, 3))),
            )

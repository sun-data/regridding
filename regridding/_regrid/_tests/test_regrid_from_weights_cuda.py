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


def _grids_interleaved(num: int = 3, num_input: int = 20, num_output: int = 12):
    """The same grids, with an orthogonal axis between the resampled ones."""
    (x, y), (x_output, y_output) = _grids(num_input, num_output)
    grid_input = (
        np.ascontiguousarray(
            np.broadcast_to(x[:, np.newaxis, :], (num_input + 1, num, num_input + 1))
        ),
        np.ascontiguousarray(
            np.broadcast_to(y[:, np.newaxis, :], (num_input + 1, num, num_input + 1))
        ),
    )
    grid_output = (
        np.ascontiguousarray(
            np.broadcast_to(
                x_output[:, :1, np.newaxis],
                (num_output + 1, num, num_output + 1),
            )
        ),
        np.ascontiguousarray(
            np.broadcast_to(
                y_output[:1, :, np.newaxis].reshape(1, 1, -1),
                (num_output + 1, num, num_output + 1),
            )
        ),
    )
    return grid_input, grid_output


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

        # seeded with something other than zero, since the weights are
        # scattered into it with an atomic add and the host path clears it
        given = cuda.to_device(np.full(weights[2], 7.0))
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


class TestAxesArbitrary:
    """The resampled axes may sit anywhere, not only at the end."""

    @pytest.mark.parametrize("shape_values", [(3, 20, 20), (1, 20, 20), (4, 3, 20, 20)])
    @requires_cuda
    def test_orthogonal_axes_leading(self, shape_values: tuple[int, ...]):
        """Axes the weights do not touch are broadcast, without a copy."""
        values = np.random.default_rng(7).random(shape_values)

        expected = regridding.regrid_from_weights(*_weights(), values_input=values)
        result = regridding.regrid_from_weights(
            *_weights(device="cuda"),
            values_input=values,
        )

        assert result.shape == expected.shape
        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-12)

    @requires_cuda
    def test_orthogonal_axis_between(self):
        """A grid whose resampled axes are not the trailing ones."""
        grid_input, grid_output = _grids_interleaved()
        axis = (0, 2)

        def weights(device):
            return regridding.weights(
                coordinates_input=grid_input,
                coordinates_output=grid_output,
                axis_input=axis,
                axis_output=axis,
                method="conservative",
                coalesce=False,
                device=device,
            )

        values = np.random.default_rng(11).random((20, 3, 20))

        expected = regridding.regrid_from_weights(
            *weights(None),
            values_input=values,
            axis_input=axis,
            axis_output=axis,
        )
        result = regridding.regrid_from_weights(
            *weights("cuda"),
            values_input=values,
            axis_input=axis,
            axis_output=axis,
        )

        assert result.shape == expected.shape == (12, 3, 12)
        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("shape_values", [(20, 20), (1, 20, 20)])
    @requires_cuda
    def test_values_on_device_broadcast(self, shape_values: tuple[int, ...]):
        """
        A device scene is broadcast across the orthogonal axis, not copied.

        Either by having fewer axes than the weights or by having one of
        length one, which are the two ways :func:`numpy.broadcast_to` would
        do it on the host.
        """
        num = 3
        x, y = _grids()[0]
        grid_input = (
            np.ascontiguousarray(np.broadcast_to(x, (num,) + x.shape)),
            np.ascontiguousarray(np.broadcast_to(y, (num,) + y.shape)),
        )
        grid_output = _grids()[1]

        def weights(device):
            return regridding.weights(
                coordinates_input=grid_input,
                coordinates_output=grid_output,
                axis_input=(-2, -1),
                method="conservative",
                coalesce=False,
                device=device,
            )

        # one scene, shared across the orthogonal axis of the weights
        values = np.random.default_rng(13).random(shape_values)

        expected = regridding.regrid_from_weights(
            *weights(None),
            values_input=values,
            axis_input=(-2, -1),
        )
        result = regridding.regrid_from_weights(
            *weights("cuda"),
            values_input=cuda.to_device(np.array(values, order="C")),
            axis_input=(-2, -1),
        )

        assert result.shape == expected.shape
        assert np.allclose(result.copy_to_host(), expected, rtol=0, atol=1e-12)


class TestNoOverlap:
    """An input grid which misses the output grid entirely."""

    @staticmethod
    def _weights(device):
        t = np.linspace(0, 1, 9)
        u = t[:, np.newaxis] * np.ones(9)
        v = np.ones(9)[:, np.newaxis] * t
        lattice = (
            np.arange(5, dtype=float)[:, np.newaxis] * np.ones(5),
            np.ones(5)[:, np.newaxis] * np.arange(5, dtype=float),
        )
        return regridding.weights(
            coordinates_input=(u + 50.0, v + 50.0),
            coordinates_output=lattice,
            method="conservative",
            coalesce=False,
            device=device,
        )

    @requires_cuda
    def test_weights_are_empty(self):
        """No overlap means no weights, rather than a kernel launched on none."""
        indices_input, indices_output, values = self._weights("cuda")[0].reshape(-1)[0]
        assert indices_input.size == 0
        assert indices_output.size == 0
        assert values.size == 0

    @requires_cuda
    def test_applies_to_zeros(self):
        """Applying them gives zeros, as it does on the host."""
        scene = np.ones((8, 8))

        expected = regridding.regrid_from_weights(
            *self._weights(None),
            values_input=scene,
        )
        result = regridding.regrid_from_weights(
            *self._weights("cuda"),
            values_input=scene,
        )

        assert np.array_equal(result.copy_to_host(), expected)
        assert not result.copy_to_host().any()


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
    def test_values_input_wrong_shape(self):
        with pytest.raises(ValueError, match="could not be broadcast"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=_scene()[:5],
            )

    @requires_cuda
    def test_values_input_on_device_wrong_shape(self):
        """A device array is checked, since stride-zero would take any shape."""
        bad = cuda.to_device(np.ascontiguousarray(np.zeros((5, 20))))
        with pytest.raises(ValueError, match="cannot be broadcast to"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=bad,
            )

    @requires_cuda
    def test_values_input_not_contiguous(self):
        wide = cuda.to_device(np.zeros((20, 40)))
        with pytest.raises(ValueError, match="has to be contiguous"):
            regridding.regrid_from_weights(
                *_weights(device="cuda"),
                values_input=wide[:, ::2],
            )

    @requires_cuda
    def test_values_output_not_contiguous(self):
        weights = _weights(device="cuda")
        given = cuda.to_device(np.zeros((weights[2][0], 2 * weights[2][1])))
        with pytest.raises(ValueError, match="has to be contiguous"):
            regridding.regrid_from_weights(
                *weights,
                values_input=_scene(),
                values_output=given[:, ::2],
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

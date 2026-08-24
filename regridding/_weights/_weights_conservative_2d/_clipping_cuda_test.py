import pytest
import numpy as np
import regridding
from ._clipping import weights_conservative_2d_clipping

try:
    from numba import cuda

    _available = cuda.is_available()
except ImportError:  # pragma: nocover
    _available = False

requires_cuda = pytest.mark.skipif(
    not _available,
    reason="a CUDA device is needed to run the device kernel",
)


def _lattice(num_x, num_y, start=0, stop=1):
    x = np.linspace(start, stop, num_x)[:, np.newaxis] * np.ones(num_y)
    y = np.ones(num_x)[:, np.newaxis] * np.linspace(start, stop, num_y)
    return x, y


def _distorted(num, angle=0.3):
    t = np.linspace(0.15, 0.85, num)
    u = t[:, np.newaxis] * np.ones(num)
    v = np.ones(num)[:, np.newaxis] * t
    x = u * np.cos(angle) - v * np.sin(angle) + 0.15 * u * v + 0.4
    y = u * np.sin(angle) + v * np.cos(angle) + 0.10 * u * u + 0.1
    return x, y


def _host(triple):
    """Bring a device triple back, dropping the slots which saw no overlap."""
    indices_input, indices_output, values = (a.copy_to_host() for a in triple)
    keep = indices_input >= 0
    return indices_input[keep], indices_output[keep], values[keep]


class TestWeightsConservative2dClippingCuda:

    @requires_cuda
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_matches_host(self, dtype):
        """The device kernel agrees with the host kernel it was ported from."""
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        grid_input = _distorted(21)
        grid_output = _lattice(9, 11)

        indices_input, indices_output, values = _host(
            weights_conservative_2d_clipping_cuda(grid_input, grid_output, dtype=dtype)
        )
        expected = weights_conservative_2d_clipping(grid_input, grid_output)

        num_output = (grid_output[0].shape[0] - 1) * (grid_output[0].shape[1] - 1)

        def dense(triple):
            result = np.zeros((20 * 20, num_output))
            np.add.at(
                result,
                (np.asarray(triple[0], np.int64), np.asarray(triple[1], np.int64)),
                np.asarray(triple[2], float),
            )
            return result

        tolerance = 1e-12 if dtype is np.float64 else 1e-6
        assert np.allclose(
            dense((indices_input, indices_output, values)),
            dense(expected),
            atol=tolerance,
        )

    @requires_cuda
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_conserved(self, dtype):
        """
        Each cell distributes the same total area as on the host.

        The comparison is against the host kernel rather than against one,
        since a cell reaching past the edge of the output grid is only
        partly covered and its total is legitimately less.
        """
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        grid_input = _distorted(21)
        grid_output = _lattice(9, 11)

        def totals(triple):
            total = np.zeros(20 * 20)
            np.add.at(
                total,
                np.asarray(triple[0], np.int64),
                np.asarray(triple[2], float),
            )
            return total

        result = totals(
            _host(
                weights_conservative_2d_clipping_cuda(
                    grid_input, grid_output, dtype=dtype
                )
            )
        )
        expected = totals(weights_conservative_2d_clipping(grid_input, grid_output))

        tolerance = 1e-12 if dtype is np.float64 else 1e-6
        assert np.allclose(result, expected, atol=tolerance)

        # and the cells wholly inside the output grid still sum to one
        full = np.isclose(expected, 1, atol=1e-12)
        assert full.any()
        assert np.allclose(result[full], 1, atol=tolerance)

    @requires_cuda
    def test_on_device(self):
        """The result is left in device memory."""
        from numba import cuda
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        triple = weights_conservative_2d_clipping_cuda(_distorted(9), _lattice(5, 5))
        for array in triple:
            assert cuda.is_cuda_array(array)

    @requires_cuda
    def test_weights_input(self):
        """`weights_input` scales each cell's row."""
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        weights_input = np.arange(1, 20 * 20 + 1, dtype=float).reshape(20, 20)
        indices_input, _, values = _host(
            weights_conservative_2d_clipping_cuda(
                _distorted(21), _lattice(9, 11), weights_input=weights_input
            )
        )

        total = np.zeros(20 * 20)
        np.add.at(total, indices_input, values.astype(float))

        # compare against the host kernel given the same scaling, so that
        # cells reaching past the edge of the output grid are not expected
        # to carry their whole weight
        indices_expected, _, values_expected = weights_conservative_2d_clipping(
            _distorted(21), _lattice(9, 11), weights_input
        )
        expected = np.zeros(20 * 20)
        np.add.at(expected, indices_expected, values_expected)

        assert np.allclose(total, expected, atol=1e-12)
        assert expected.max() > 1


class TestDeviceRejected:
    """The cases the device path cannot serve are refused, not mishandled."""

    grid_input = _distorted(9)
    grid_output = _lattice(5, 5)

    def test_multilinear(self):
        with pytest.raises(ValueError, match="only supported by the conservative"):
            regridding.weights(
                coordinates_input=self.grid_input,
                coordinates_output=self.grid_output,
                method="multilinear",
                device="cuda",
            )

    def test_coalesce(self):
        with pytest.raises(ValueError, match="needs `coalesce=False`"):
            regridding.weights(
                coordinates_input=self.grid_input,
                coordinates_output=self.grid_output,
                method="conservative",
                device="cuda",
            )

    def test_output_not_a_lattice(self):
        with pytest.raises(ValueError, match="uniform, axis-aligned lattice"):
            regridding.weights(
                coordinates_input=self.grid_input,
                coordinates_output=_distorted(5),
                method="conservative",
                coalesce=False,
                device="cuda",
            )

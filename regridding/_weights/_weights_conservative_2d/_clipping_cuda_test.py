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

    @requires_cuda
    def test_dtype_values(self):
        """`dtype_values` is what the device builds in, not a later cast."""
        import regridding

        weights = regridding.weights(
            coordinates_input=_distorted(9),
            coordinates_output=_lattice(5, 5),
            method="conservative",
            coalesce=False,
            device="cuda",
            dtype_values=np.float32,
        )

        indices_input, indices_output, values = weights[0].reshape(-1)[0]

        assert values.dtype == np.float32
        assert cuda.is_cuda_array(values)

        # and the weights are still the ones the host would have built
        expected = weights_conservative_2d_clipping(_distorted(9), _lattice(5, 5))
        total = np.zeros(8 * 8)
        got = _host((indices_input, indices_output, values))
        np.add.at(total, got[0], got[2].astype(float))
        expected_total = np.zeros(8 * 8)
        np.add.at(expected_total, expected[0], expected[2])
        assert np.allclose(total, expected_total, rtol=0, atol=1e-6)

    @requires_cuda
    def test_grid_input_on_device(self):
        """A grid already on the device is used where it is, in cell units."""
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        grid_input = _distorted(21)
        grid_output = _lattice(9, 11)

        x_output, y_output = grid_output
        origin_x = float(x_output[0, 0])
        origin_y = float(y_output[0, 0])
        step_x = float(x_output[1, 0] - x_output[0, 0])
        step_y = float(y_output[0, 1] - y_output[0, 0])
        resident = (
            cuda.to_device(np.ascontiguousarray((grid_input[0] - origin_x) / step_x)),
            cuda.to_device(np.ascontiguousarray((grid_input[1] - origin_y) / step_y)),
        )

        result = _host(weights_conservative_2d_clipping_cuda(resident, grid_output))
        expected = _host(weights_conservative_2d_clipping_cuda(grid_input, grid_output))

        for got, want in zip(result, expected):
            assert np.array_equal(got, want)

    @requires_cuda
    def test_weights_input_on_device(self):
        """`weights_input` already on the device is used where it is."""
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        grid_input = _distorted(21)
        grid_output = _lattice(9, 11)
        weights_input = np.arange(1, 20 * 20 + 1, dtype=float).reshape(20, 20)

        result = _host(
            weights_conservative_2d_clipping_cuda(
                grid_input,
                grid_output,
                weights_input=cuda.to_device(np.ascontiguousarray(weights_input)),
            )
        )
        expected = _host(
            weights_conservative_2d_clipping_cuda(
                grid_input,
                grid_output,
                weights_input=weights_input,
            )
        )

        for got, want in zip(result, expected):
            assert np.array_equal(got, want)

    @requires_cuda
    def test_slots_without_overlap(self):
        """
        A slot which sees no overlap holds the sentinel and zeros.

        The host kernel builds its result with :func:`numpy.zeros`, so a
        reader which forgets to drop the sentinel slots sees zeros there
        rather than whatever the allocation happened to contain.
        """
        from ._clipping_cuda import weights_conservative_2d_clipping_cuda

        # rotated hard enough that many bounding boxes cover cells the
        # quadrilateral itself misses
        num = 21
        t = np.linspace(-0.7, 0.7, num)
        u = t[:, np.newaxis] * np.ones(num)
        v = np.ones(num)[:, np.newaxis] * t
        angle = 0.785
        grid_input = (
            (u * np.cos(angle) - v * np.sin(angle)) * 4 + 6.3,
            (u * np.sin(angle) + v * np.cos(angle)) * 4 + 6.1,
        )
        grid_output = (
            np.arange(13, dtype=float)[:, np.newaxis] * np.ones(13),
            np.ones(13)[:, np.newaxis] * np.arange(13, dtype=float),
        )

        indices_input, indices_output, values = (
            a.copy_to_host()
            for a in weights_conservative_2d_clipping_cuda(grid_input, grid_output)
        )

        empty = indices_input < 0
        assert empty.any()
        assert np.array_equal(indices_output[empty], np.zeros(empty.sum(), np.int64))
        assert np.array_equal(values[empty], np.zeros(empty.sum()))


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

    def test_dtype_indices(self):
        with pytest.raises(ValueError, match="addresses its slots"):
            regridding.weights(
                coordinates_input=self.grid_input,
                coordinates_output=self.grid_output,
                method="conservative",
                coalesce=False,
                device="cuda",
                dtype_indices=np.int32,
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

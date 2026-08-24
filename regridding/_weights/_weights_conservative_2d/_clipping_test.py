import pytest
import numpy as np
from ._weights_conservative_2d import weights_conservative_2d
from ._clipping import (
    grid_is_uniform_rectilinear,
    weights_conservative_2d_clipping,
)


def _lattice(
    num_x: int,
    num_y: int,
    start: float = -1,
    stop: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a uniform, axis-aligned grid of cell vertices."""
    x = np.linspace(start, stop, num_x)[:, np.newaxis] * np.ones(num_y)
    y = np.ones(num_x)[:, np.newaxis] * np.linspace(start, stop, num_y)
    return x, y


def _distorted(
    num: int,
    angle: float = 0.3,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a rotated and quadratically-distorted grid of cell vertices."""
    t = np.linspace(-0.7, 0.7, num)
    u = t[:, np.newaxis] * np.ones(num)
    v = np.ones(num)[:, np.newaxis] * t
    x = u * np.cos(angle) - v * np.sin(angle) + 0.15 * u * v
    y = u * np.sin(angle) + v * np.cos(angle) + 0.10 * u * u
    if flip:
        x, y = x[::-1], y[::-1]
    return x, y


def _dense(
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_input: int,
    num_output: int,
) -> np.ndarray:
    """Accumulate a triple of weights into a dense matrix."""
    indices_input, indices_output, values = weights
    result = np.zeros((num_input, num_output))
    np.add.at(
        result,
        (
            np.asarray(indices_input, dtype=np.int64),
            np.asarray(indices_output, dtype=np.int64),
        ),
        np.asarray(values, dtype=float),
    )
    return result


def _num_cells(grid: tuple[np.ndarray, np.ndarray]) -> int:
    x, _ = grid
    return (x.shape[0] - 1) * (x.shape[1] - 1)


class TestGridIsUniformRectilinear:

    @pytest.mark.parametrize(
        argnames="grid,result_expected",
        argvalues=[
            (_lattice(6, 6), True),
            (_lattice(6, 9), True),
            (_lattice(6, 6, start=1, stop=-1), True),
            (_distorted(6), False),
        ],
    )
    def test_uniform(
        self,
        grid: tuple[np.ndarray, np.ndarray],
        result_expected: bool,
    ):
        assert grid_is_uniform_rectilinear(grid) == result_expected

    def test_nonuniform_spacing_x(self):
        x, y = _lattice(6, 6)
        assert not grid_is_uniform_rectilinear((x * x, y))

    def test_nonuniform_spacing_y(self):
        x, y = _lattice(6, 6)
        assert not grid_is_uniform_rectilinear((x, y * y))

    def test_varies_along_wrong_axis(self):
        """`y` must vary along the second axis, not the first."""
        x, _ = _lattice(6, 6)
        assert not grid_is_uniform_rectilinear((x, x))

    def test_not_finite(self):
        x, y = _lattice(6, 6)
        x = x.copy()
        x[0] = np.inf
        assert not grid_is_uniform_rectilinear((x, y))


class TestWeightsConservative2dClipping:

    @pytest.mark.parametrize(
        argnames="grid_input",
        argvalues=[
            _distorted(9),
            _distorted(9, angle=-0.2),
            _distorted(12, angle=0.9),
        ],
    )
    def test_matches_sweep(
        self,
        grid_input: tuple[np.ndarray, np.ndarray],
    ):
        """
        The clipping kernel agrees with the sweep on non-degenerate grids.

        Grids which share vertices or have collinear edges are excluded
        here: the sweep cannot resolve them unless the coordinates are
        perturbed, so it is not a usable reference.  Those cases are
        covered against analytic expectations instead, by
        :meth:`test_identical_grids` and :meth:`test_refinement`.
        """
        grid_output = _lattice(6, 6)

        num_input = _num_cells(grid_input)
        num_output = _num_cells(grid_output)

        result = _dense(
            weights_conservative_2d_clipping(grid_input, grid_output),
            num_input,
            num_output,
        )
        expected = _dense(
            weights_conservative_2d(grid_input, grid_output, None),
            num_input,
            num_output,
        )

        assert np.allclose(result, expected, atol=1e-12)

    def test_identical_grids(self):
        """
        Resampling a grid onto itself is the identity.

        Every vertex is shared, which the sweep cannot resolve without
        perturbing the coordinates, but clipping handles exactly.
        """
        grid = _lattice(6, 6)
        num = _num_cells(grid)

        result = _dense(
            weights_conservative_2d_clipping((grid[0].copy(), grid[1].copy()), grid),
            num,
            num,
        )

        assert np.allclose(result, np.eye(num), atol=1e-12)

    def test_refinement(self):
        """
        A grid refined by exactly two maps each input cell into one output
        cell, which is another exactly-degenerate case for the sweep.
        """
        grid_input = _lattice(11, 11)
        grid_output = _lattice(6, 6)

        num_input = _num_cells(grid_input)
        num_output = _num_cells(grid_output)

        result = _dense(
            weights_conservative_2d_clipping(grid_input, grid_output),
            num_input,
            num_output,
        )

        index_input_x, index_input_y = np.indices((10, 10))
        index_output = (index_input_x // 2) * 5 + (index_input_y // 2)
        expected = np.zeros((num_input, num_output))
        expected[np.arange(num_input), index_output.reshape(-1)] = 1

        assert np.allclose(result, expected, atol=1e-12)

    def test_inverted_orientation(self):
        """
        A grid wound in the opposite sense gives the same weights, relabelled.

        Reversing one axis of the vertex array does not move any cell, it
        only renumbers the cells and flips the sign of their areas, so the
        weights must simply permute.
        """
        grid_output = _lattice(6, 6)
        num_output = _num_cells(grid_output)

        grid = _distorted(9)
        grid_flipped = _distorted(9, flip=True)
        num_input = _num_cells(grid)

        result = _dense(
            weights_conservative_2d_clipping(grid_flipped, grid_output),
            num_input,
            num_output,
        )

        num_x = grid[0].shape[0] - 1
        num_y = grid[0].shape[1] - 1
        permutation = (
            np.arange(num_x)[::-1][:, np.newaxis] * num_y + np.arange(num_y)
        ).reshape(-1)
        expected = _dense(
            weights_conservative_2d_clipping(grid, grid_output),
            num_input,
            num_output,
        )[permutation]

        assert np.allclose(result, expected, atol=1e-12)

    def test_conservation(self):
        """Every fully-covered input cell distributes exactly its own area."""
        grid_input = _distorted(9)
        grid_output = _lattice(6, 6)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
        )

        total = np.zeros(_num_cells(grid_input))
        np.add.at(total, indices_input, values)

        assert np.allclose(total, 1, atol=1e-12)

    @pytest.mark.parametrize("num_output", [50, 500])
    def test_conservation_on_a_large_output_grid(self, num_output: int):
        """
        Conservation does not decay as the output grid is refined.

        The kernel works in output-cell units, so a cell of a finely divided
        output grid sits at a large coordinate.  The shoelace formula sums
        products of coordinates and then cancels almost all of the total
        away, so evaluating it at that scale would lose precision as the
        square of the number of output cells.  Each cell is therefore shifted
        onto the block of output cells it touches before being clipped, which
        keeps the arithmetic at the scale of a cell.

        Without that shift a 500 by 500 output grid conserves area only to
        about ``1e-10``, and a 5000 by 5000 one only to about ``1e-8``.
        """
        span = 2.0

        # a small distorted patch, sitting near the far corner of the output
        # grid where the coordinates are largest
        x, y = _distorted(9)
        scale = 3 * span / num_output
        grid_input = (x * scale + 0.9 * span, y * scale + 0.9 * span)

        grid_output = _lattice(num_output + 1, num_output + 1, start=0, stop=span)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
        )

        total = np.zeros(_num_cells(grid_input))
        np.add.at(total, indices_input, values)

        assert np.allclose(total, 1, rtol=0, atol=1e-14)

    def test_weights_input(self):
        """`weights_input` scales each input cell's row."""
        grid_input = _distorted(9)
        grid_output = _lattice(6, 6)

        shape = (grid_input[0].shape[0] - 1, grid_input[0].shape[1] - 1)
        weights_input = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
            weights_input,
        )

        total = np.zeros(_num_cells(grid_input))
        np.add.at(total, indices_input, values)

        assert np.allclose(total, weights_input.reshape(-1), atol=1e-12)

    def test_partial_coverage(self):
        """Cells hanging outside the output grid contribute only their overlap."""
        grid_input = _lattice(3, 3, start=0, stop=2)
        grid_output = _lattice(3, 3, start=0, stop=1)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
        )

        total = np.zeros(_num_cells(grid_input))
        np.add.at(total, indices_input, values)

        # only the lower-left input cell lies inside the output grid
        assert np.isclose(total[0], 1)
        assert np.allclose(total[1:], 0)

    def test_grid_output_not_rectilinear(self):
        with pytest.raises(ValueError):
            weights_conservative_2d_clipping(_lattice(6, 6), _distorted(6))

    def test_degenerate_cell(self):
        """A cell with zero area contributes nothing and does not divide by it."""
        x, y = _lattice(4, 4, start=0, stop=3)
        x = x.copy()

        # collapse the first column of vertices onto the second, so every
        # cell in the first column has zero area
        x[0] = x[1]

        indices_input, _, values = weights_conservative_2d_clipping(
            (x, y),
            _lattice(4, 4, start=0, stop=3),
        )

        total = np.zeros(9)
        np.add.at(total, indices_input, values)

        assert np.all(np.isfinite(values))
        assert np.allclose(total[:3], 0)
        assert np.allclose(total[3:], 1)

    def test_hangs_off_lower_corner(self):
        """Cells reaching past the lower corner keep only their overlap."""
        grid_input = _lattice(3, 3, start=-1, stop=1)
        grid_output = _lattice(3, 3, start=0, stop=2)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
        )

        total = np.zeros(4)
        np.add.at(total, indices_input, values)

        # only the upper-right input cell lies inside the output grid
        assert np.allclose(total[:3], 0)
        assert np.isclose(total[3], 1)

    def test_bounding_box_larger_than_overlap(self):
        """
        A thin diagonal cell touches far fewer output cells than its
        bounding box spans, so most candidates clip away to nothing.
        """
        num = 9
        t = np.linspace(0, 8, num)
        u = t[:, np.newaxis] * np.ones(num)
        v = np.ones(num)[:, np.newaxis] * t

        # a narrow band running diagonally across the output grid
        x = u + 0.02 * v
        y = 0.9 * u + 0.05 * v + 0.5

        grid_output = _lattice(9, 9, start=0, stop=8)

        indices_input, _, values = weights_conservative_2d_clipping(
            (x, y),
            grid_output,
        )

        total = np.zeros((num - 1) * (num - 1))
        np.add.at(total, indices_input, values)

        assert np.all(np.isfinite(values))
        assert np.all(total <= 1 + 1e-12)


class TestNonConvexCells:
    """
    A cell which is not convex, but whose edges do not cross, is a legitimate
    shape with a well-defined area, and a strong enough distortion produces
    one.  It needs more vertex slots than a convex cell does.
    """

    @staticmethod
    def _grid_with_dart() -> tuple[np.ndarray, np.ndarray]:
        """A lattice with one vertex pulled inside its neighbours."""
        x, y = _lattice(4, 4, start=0, stop=3)
        x, y = x.copy(), y.copy()
        # drag the middle vertex far enough that the four cells sharing it
        # become non-convex without any edges crossing
        x[1, 1] = 1.72
        y[1, 1] = 1.72
        return x, y

    def test_conserved(self):
        """Every fully-covered cell still distributes exactly its own area."""
        grid_input = self._grid_with_dart()
        grid_output = _lattice(7, 7, start=0, stop=3)

        indices_input, _, values = weights_conservative_2d_clipping(
            grid_input,
            grid_output,
        )

        total = np.zeros(_num_cells(grid_input))
        np.add.at(total, indices_input, values)

        assert np.all(np.isfinite(values))
        assert np.allclose(total, 1, atol=1e-12)

    def test_matches_sweep(self):
        """The sweep agrees, which it can only do if the clip is complete."""
        grid_input = self._grid_with_dart()
        grid_output = _lattice(7, 7, start=-0.13, stop=3.11)

        num_input = _num_cells(grid_input)
        num_output = _num_cells(grid_output)

        result = _dense(
            weights_conservative_2d_clipping(grid_input, grid_output),
            num_input,
            num_output,
        )
        expected = _dense(
            weights_conservative_2d(grid_input, grid_output, None),
            num_input,
            num_output,
        )

        assert np.allclose(result, expected, atol=1e-12)

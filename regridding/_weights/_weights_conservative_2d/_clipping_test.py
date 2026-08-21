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

    def test_nonuniform_spacing(self):
        x, y = _lattice(6, 6)
        x = x * x
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

import pytest
import numpy as np
from . import _rectilinear


def _flat(grid: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The vertices of a grid, flattened, and its shape."""
    vertices = np.stack([np.ravel(c) for c in np.broadcast_arrays(*grid)])
    return vertices.astype(float), np.array(np.broadcast(*grid).shape)


x = np.array([0.0, 0.5, 2.0, 2.25])
y = np.array([1.0, 3.0, 4.0])
z = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])


class TestProfiles:

    def test_meshgrid(self) -> None:
        vertices, shape = _flat(np.meshgrid(x, y, z, indexing="ij"))
        result = _rectilinear.profiles(vertices, shape)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, np.concatenate([x, y, z]))
        np.testing.assert_array_equal(start, [0, 4, 7, 12])
        np.testing.assert_array_equal(coordinate, [0, 1, 2])
        np.testing.assert_array_equal(descending, [False, False, False])

    def test_coordinates_out_of_order(self) -> None:
        # the first coordinate varies along the second axis, and the second
        # along the first
        yy, xx = np.meshgrid(y, x, indexing="ij")
        vertices, shape = _flat((xx, yy))
        result = _rectilinear.profiles(vertices, shape)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, np.concatenate([y, x]))
        np.testing.assert_array_equal(coordinate, [1, 0])

    def test_descending(self) -> None:
        vertices, shape = _flat(np.meshgrid(x, y[::-1], indexing="ij"))
        result = _rectilinear.profiles(vertices, shape)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, np.concatenate([x, y]))
        np.testing.assert_array_equal(descending, [False, True])

    @pytest.mark.parametrize(
        argnames="grid",
        argvalues=[
            # rotated
            (
                np.add.outer(x, 0.1 * y),
                np.add.outer(-0.1 * x, y),
            ),
            # not monotonic
            np.meshgrid(np.array([0.0, 1.0, 0.5]), y, indexing="ij"),
            # both coordinates vary along the first axis only
            (
                np.broadcast_to(x[:, np.newaxis], (4, 3)),
                np.broadcast_to(2 * x[:, np.newaxis], (4, 3)),
            ),
            # a coordinate which does not vary at all
            (
                np.broadcast_to(x[:, np.newaxis], (4, 3)),
                np.zeros((4, 3)),
            ),
            # a coordinate which is NaN somewhere
            np.meshgrid(np.array([0.0, np.nan, 1.0]), y, indexing="ij"),
        ],
    )
    def test_not_rectilinear(self, grid: tuple[np.ndarray, ...]) -> None:
        vertices, shape = _flat(grid)
        assert _rectilinear.profiles(vertices, shape) is None


class TestLocate:

    @pytest.mark.parametrize("reverse", [False, True])
    def test_locate(self, reverse: bool) -> None:
        # the second axis may run backwards, which has to give the same
        # cells counted from the other end
        y_ = y[::-1] if reverse else y
        vertices, shape = _flat(np.meshgrid(x, y_, indexing="ij"))
        result = _rectilinear.profiles(vertices, shape)
        assert result is not None
        points = np.array(
            [
                # inside, on a vertex, on the edges, outside, NaN
                [0.25, 2.0, 0.0, 2.25, -1.0, 3.0, 1.0],
                [2.0, 3.0, 1.0, 4.0, 2.0, 5.0, np.nan],
            ]
        )
        cell, xi, inside = _rectilinear.locate(*result, points)

        expected_cell_x = [0, 1, 0, 2, 0, 2, 1]
        expected_xi_x = [0.5, 1.0, 0.0, 1.0, -2.0, 4.0, 1 / 3]
        expected_cell_y = np.array([0, 0, 0, 1, 0, 1, 1])
        expected_xi_y = np.array([0.5, 1.0, 0.0, 1.0, 0.5, 2.0, np.nan])
        if reverse:
            expected_cell_y = 1 - expected_cell_y
            expected_xi_y = 1 - expected_xi_y

        np.testing.assert_array_equal(cell[:, 0], expected_cell_x)
        np.testing.assert_array_equal(cell[:, 1], expected_cell_y)
        np.testing.assert_allclose(xi[:, 0], expected_xi_x, rtol=1e-15)
        np.testing.assert_allclose(xi[:, 1], expected_xi_y, rtol=1e-15)
        np.testing.assert_array_equal(
            inside, [True, True, True, True, False, False, False]
        )

import pytest
import numpy as np
from . import _rectilinear


def _grid(grid: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """
    The coordinates of one grid, broadcast against each other, with an axis
    for a single element in front.
    """
    return tuple(c[np.newaxis] for c in np.broadcast_arrays(*grid))


x = np.array([0.0, 0.5, 2.0, 2.25])
y = np.array([1.0, 3.0, 4.0])
z = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])


class TestProfiles:

    def test_meshgrid(self) -> None:
        coordinates = _grid(np.meshgrid(x, y, z, indexing="ij"))
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, [np.concatenate([x, y, z])])
        np.testing.assert_array_equal(start, [0, 4, 7, 12])
        np.testing.assert_array_equal(coordinate, [0, 1, 2])
        np.testing.assert_array_equal(descending, [[False, False, False]])

    def test_broadcast(self) -> None:
        # a grid broadcast from one dimension along each axis is recognized
        # from its strides, without its values being compared
        coordinates = _grid(np.meshgrid(x, y, indexing="ij", sparse=True))
        assert all(0 in c.strides for c in coordinates)
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, [np.concatenate([x, y])])
        np.testing.assert_array_equal(coordinate, [0, 1])

    def test_coordinates_out_of_order(self) -> None:
        # the first coordinate varies along the second axis, and the second
        # along the first
        yy, xx = np.meshgrid(y, x, indexing="ij")
        coordinates = _grid((xx, yy))
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, [np.concatenate([y, x])])
        np.testing.assert_array_equal(coordinate, [1, 0])

    def test_descending(self) -> None:
        coordinates = _grid(np.meshgrid(x, y[::-1], indexing="ij"))
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, [np.concatenate([x, y])])
        np.testing.assert_array_equal(descending, [[False, True]])

    def test_descending_unsigned(self) -> None:
        # the difference of two unsigned integers wraps around instead of
        # going negative, which once passed this for an ascending grid
        coordinates = (np.array([[5, 4, 3, 2, 1]], dtype=np.uint8),)
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        profile, start, coordinate, descending = result
        np.testing.assert_array_equal(profile, [[1.0, 2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_array_equal(descending, [[True]])

    def test_elements(self) -> None:
        # a grid for each of three elements, the second running backwards
        # along its second axis, and every one sharing its first coordinate
        xx = np.broadcast_to(x[np.newaxis, :, np.newaxis], (3, 4, 3))
        yy = np.stack([y, 10 - y, 2 * y])[:, np.newaxis, :]
        yy = np.broadcast_to(yy, (3, 4, 3))
        result = _rectilinear.profiles((xx, yy))
        assert result is not None
        profile, start, coordinate, descending = result
        expected = [
            np.concatenate([x, y]),
            np.concatenate([x, 10 - y[::-1]]),
            np.concatenate([x, 2 * y]),
        ]
        np.testing.assert_array_equal(profile, expected)
        np.testing.assert_array_equal(
            descending,
            [[False, False], [False, True], [False, False]],
        )

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
        coordinates = _grid(grid)
        assert _rectilinear.profiles(coordinates) is None

    def test_elements_disagree(self) -> None:
        # rectilinear for one element but not monotonic for the other
        xx = np.stack([x, np.array([0.0, 1.0, 0.5, 2.0])])
        assert _rectilinear.profiles((xx,)) is None


class TestLocate:

    @pytest.mark.parametrize("reverse", [False, True])
    def test_locate(self, reverse: bool) -> None:
        # the second axis may run backwards, which has to give the same
        # cells counted from the other end
        y_ = y[::-1] if reverse else y
        coordinates = _grid(np.meshgrid(x, y_, indexing="ij"))
        result = _rectilinear.profiles(coordinates)
        assert result is not None
        points = np.array(
            [
                # inside, on a vertex, on the edges, outside, NaN
                [0.25, 2.0, 0.0, 2.25, -1.0, 3.0, 1.0],
                [2.0, 3.0, 1.0, 4.0, 2.0, 5.0, np.nan],
            ]
        )
        cell, xi, inside = _rectilinear.locate(*result, points[np.newaxis])

        expected_cell_x = [0, 1, 0, 2, 0, 2, 1]
        expected_xi_x = [0.5, 1.0, 0.0, 1.0, -2.0, 4.0, 1 / 3]
        expected_cell_y = np.array([0, 0, 0, 1, 0, 1, 1])
        expected_xi_y = np.array([0.5, 1.0, 0.0, 1.0, 0.5, 2.0, np.nan])
        if reverse:
            expected_cell_y = 1 - expected_cell_y
            expected_xi_y = 1 - expected_xi_y

        np.testing.assert_array_equal(cell[0, :, 0], expected_cell_x)
        np.testing.assert_array_equal(cell[0, :, 1], expected_cell_y)
        np.testing.assert_allclose(xi[0, :, 0], expected_xi_x, rtol=1e-15)
        np.testing.assert_allclose(xi[0, :, 1], expected_xi_y, rtol=1e-15)
        np.testing.assert_array_equal(
            inside, [[True, True, True, True, False, False, False]]
        )

    def test_elements(self) -> None:
        # each element is searched in its own grid, the second of which is
        # [4.5, 4, 1, 0], running backwards
        xx = np.stack([x, 2 * x[::-1]])
        result = _rectilinear.profiles((xx,))
        assert result is not None
        points = np.array([[[0.25, 3.0]], [[0.5, 4.25]]])
        cell, xi, inside = _rectilinear.locate(*result, points)
        np.testing.assert_array_equal(cell[..., 0], [[0, 2], [2, 0]])
        np.testing.assert_allclose(
            xi[..., 0],
            [[0.5, 4.0], [0.5, 0.5]],
            rtol=1e-15,
        )
        np.testing.assert_array_equal(inside, [[True, False], [True, True]])

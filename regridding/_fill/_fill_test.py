import pytest
import numpy as np
import regridding

_num_x = 11
_num_y = 12
_num_t = 13


@pytest.mark.parametrize(
    argnames="a,where,axis",
    argvalues=[
        (
            np.random.uniform(0, 1, size=(_num_x, _num_y)),
            np.random.uniform(0, 1, size=(_num_x, _num_y)) > 0.9,
            None,
        ),
        (
            np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y)),
            np.random.uniform(0, 1, size=(_num_t, _num_x, _num_y)) > 0.9,
            (~1, ~0),
        ),
        (
            np.sqrt(np.random.uniform(-0.1, 1, size=(_num_x, _num_t, _num_y))),
            None,
            (0, ~0),
        ),
    ],
)
@pytest.mark.parametrize("num_iterations", [11])
def test_fill_gauss_sidel_2d(
    a: np.ndarray,
    where: np.ndarray,
    axis: None | tuple[int, ...],
    num_iterations: int,
):
    result = regridding.fill(
        a=a,
        where=where,
        axis=axis,
        method="gauss_seidel",
        num_iterations=num_iterations,
    )
    if where is None:
        where = np.isnan(a)

    assert np.allclose(result[~where], a[~where])
    assert np.all(result[where] != 0)

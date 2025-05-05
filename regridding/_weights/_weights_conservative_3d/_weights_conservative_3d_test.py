import pytest
import numpy as np
from . import _weights_conservative_3d

_x = np.arange(3)
_y = np.arange(4)
_z = np.arange(5)

_x, _y, _z = np.meshgrid(_x, _y, _z, indexing="ij")


@pytest.mark.parametrize(
    argnames="grid_input,grid_output",
    argvalues=[
        (
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
            np.meshgrid(
                np.arange(2),
                np.arange(2),
                np.arange(2),
                indexing="ij",
            ),
        )
    ]
)
def test_weights_conservative_3d(
    grid_input: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    result = _weights_conservative_3d.weights_conservative_3d(
        grid_input=grid_input,
        grid_output=grid_output,
    )






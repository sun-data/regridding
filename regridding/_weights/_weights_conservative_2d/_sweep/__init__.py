"""
First-order conservative weights for a 2D grid, by sweeping its grid lines.

The general case: the grid lines of both grids are swept and the boundary
integrals accumulated, which works for any output grid but walks each line
in order and pays for the output grid whether or not the input reaches it.
Where the output grid is a uniform, axis-aligned lattice,
:mod:`~regridding._weights._weights_conservative_2d._clipping` does the same
job by clipping instead, and does it faster.
"""

from ._sweep import weights_conservative_2d

__all__ = [
    "weights_conservative_2d",
]

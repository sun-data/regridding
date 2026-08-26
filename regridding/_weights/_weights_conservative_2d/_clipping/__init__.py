"""
First-order conservative weights for a 2D grid, by clipping its cells.

Each input cell is clipped against the output cells its bounding box
touches, which needs the output grid to be a uniform, axis-aligned lattice
but makes every cell independent of every other.  That independence is what
lets the same algorithm run on the host and on a CUDA device:
:mod:`._shared` holds it once, and :mod:`._host` and :mod:`._cuda` compile
that source for their own target.
"""

from ._host import grid_is_uniform_rectilinear, weights_conservative_2d_clipping
from ._cuda import weights_conservative_2d_clipping_cuda

__all__ = [
    "grid_is_uniform_rectilinear",
    "weights_conservative_2d_clipping",
    "weights_conservative_2d_clipping_cuda",
]

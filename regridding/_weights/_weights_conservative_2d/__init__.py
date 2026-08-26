"""
First-order conservative weights for a 2D logically-rectangular grid.

There are two algorithms, which produce the same weights and which
:func:`~regridding._weights._weights_conservative._weights_conservative`
chooses between: :mod:`._clipping` where the output grid is a uniform,
axis-aligned lattice, and :mod:`._sweep` otherwise.  They share nothing but
the grid helpers beside them.
"""

from ._sweep import weights_conservative_2d
from ._clipping import (
    grid_is_uniform_rectilinear,
    weights_conservative_2d_clipping,
    weights_conservative_2d_clipping_cuda,
)

__all__ = [
    "weights_conservative_2d",
    "grid_is_uniform_rectilinear",
    "weights_conservative_2d_clipping",
    "weights_conservative_2d_clipping_cuda",
]

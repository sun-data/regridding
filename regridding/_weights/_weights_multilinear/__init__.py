"""
Multilinear weights for a logically-rectangular grid of any number of
dimensions.

:mod:`._weights_multilinear` finds the cell of the input grid which contains
each output point, with :mod:`._rectilinear` if the grid is rectilinear and
:mod:`._curvilinear` otherwise, and turns the local coordinates of the point
in that cell into a weight for each corner of the cell with :mod:`._cells`.
"""

from ._weights_multilinear import weights_multilinear

__all__ = [
    "weights_multilinear",
]

"""
Resample arrays defined on curvilinear grids.

The two entry points are :func:`regridding.regrid`, which resamples an array in
a single call, and :func:`regridding.weights`, which saves the sparse matrix
relating the two grids so that it can be applied to many arrays using
:func:`regridding.regrid_from_weights`.
"""

from . import math
from . import geometry
from ._find_indices import find_indices
from ._weights import (
    weights,
    transpose_weights,
    transpose_weights_conservative,
)
from ._interp_ndarray import ndarray_linear_interpolation
from ._regrid import regrid_from_weights, regrid
from ._fill import fill

__all__ = [
    "math",
    "geometry",
    "find_indices",
    "weights",
    "transpose_weights",
    "transpose_weights_conservative",
    "ndarray_linear_interpolation",
    "regrid_from_weights",
    "regrid",
    "fill",
]

import numpy as np


def _normalize_axis(
    axis: None | int | tuple[int, ...],
    ndim: int,
) -> tuple[int, ...]:
    if axis is None:
        axis = tuple(range(ndim))
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim=ndim)
    axis = tuple(~(~np.array(axis) % ndim))
    return axis

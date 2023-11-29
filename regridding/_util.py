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


def _normalize_input_output_vertices(
    vertices_input: tuple[np.ndarray, ...],
    vertices_output: tuple[np.ndarray, ...],
    axis_input: None | int | tuple[int, ...] = None,
    axis_output: None | int | tuple[int, ...] = None,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    shape_vertices_input = np.broadcast(*vertices_input).shape
    shape_vertices_output = np.broadcast(*vertices_output).shape

    ndim_input = len(shape_vertices_input)
    ndim_output = len(shape_vertices_output)

    axis_input = _normalize_axis(axis_input, ndim=ndim_input)
    axis_output = _normalize_axis(axis_output, ndim=ndim_output)

    axis_input = sorted(axis_input, reverse=True)
    axis_output = sorted(axis_output, reverse=True)

    if len(axis_output) != len(axis_input):
        raise ValueError(
            f"The number of axes in `axis_output`, {axis_output}, "
            f"must match the number of axes in `axis_input`, {axis_input}"
        )

    if len(vertices_input) != len(axis_input):
        raise ValueError(
            f"The number of elements in `vertices_input`, {len(vertices_input)}, "
            f"should match the number of axes in `axis_input`, {axis_input}"
        )

    if len(vertices_output) != len(vertices_input):
        raise ValueError(
            f"The number of elements in `vertices_output`, {len(vertices_output)}, "
            f"should match the number of elements in `vertices_input`, {len(vertices_input)}"
        )

    axis_input_orthogonal = tuple(
        ax for ax in _normalize_axis(None, ndim_input) if ax not in axis_input
    )
    axis_output_orthogonal = tuple(
        ax for ax in _normalize_axis(None, ndim_output) if ax not in axis_output
    )

    shape_input_orthogonal = tuple(
        shape_vertices_input[ax] for ax in axis_input_orthogonal
    )
    shape_output_orthogonal = tuple(
        shape_vertices_output[ax] for ax in axis_output_orthogonal
    )

    shape_orthogonal = np.broadcast_shapes(
        shape_input_orthogonal, shape_output_orthogonal
    )

    shape_input = list(shape_orthogonal)
    for ax in axis_input:
        shape_input.insert(ax, shape_vertices_input[ax])
    shape_input = tuple(shape_input)

    shape_output = list(shape_orthogonal)
    for ax in axis_output:
        shape_output.insert(ax, shape_vertices_output[ax])
    shape_output = tuple(shape_output)

    vertices_input = tuple(
        np.broadcast_to(component, shape_input) for component in vertices_input
    )
    vertices_output = tuple(
        np.broadcast_to(component, shape_output) for component in vertices_output
    )

    return (
        vertices_input,
        vertices_output,
        axis_input,
        axis_output,
        shape_input,
        shape_output,
        shape_orthogonal,
    )

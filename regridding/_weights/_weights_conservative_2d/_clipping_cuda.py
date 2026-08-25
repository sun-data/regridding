"""
The clipping kernel on a CUDA device, leaving the result in device memory.

The algorithm is not repeated here.  It lives in
:mod:`~regridding._weights._weights_conservative_2d._clipping_shared`, and
this module compiles that same source for the device, exactly as
:mod:`~regridding._weights._weights_conservative_2d._clipping` compiles it
for the host.  What remains here is only what a device needs and a CPU does
not: mapping threads onto cells, allocating the scratch space in local
memory, the prefix sum, and the device allocations.

Each input cell is independent and writes into its own slice of the result,
which is what makes the algorithm suit a GPU in the first place.
"""

from typing import Any, Callable
import numpy as np
import numba
from numba import cuda
import regridding as rg
from regridding import _cuda
from ._clipping_shared import num_slot as _num_slot, build as _build_shared

__all__ = [
    "weights_conservative_2d_clipping_cuda",
]

# `numba` declares several of the names below as intrinsics or with
# annotations which describe how the compiler calls them rather than how a
# kernel does, so the calls to them carry `type: ignore` comments.


def _jit(function: Callable) -> Any:
    """
    Compile one of the shared kernel bodies for a CUDA device.

    Parameters
    ----------
    function
        The plain Python function to compile.
    """
    return cuda.jit(device=True, inline=True)(function)


def _source(function: Callable) -> Callable:
    """
    Recover the undecorated source of a host function.

    A kernel cannot call a function compiled for the host, so the device has
    to compile the same source itself.  :func:`numba.njit` keeps it on the
    dispatcher as ``py_func``, except when ``NUMBA_DISABLE_JIT`` is set, in
    which case it hands back the plain function and there is nothing to
    unwrap.

    Parameters
    ----------
    function
        The host function to unwrap.
    """
    return getattr(function, "py_func", function)


def _build(ftype: Any) -> tuple[Any, Any]:
    """
    Build the kernels for a given floating-point type.

    The coordinates may be single or double precision, and the scratch space
    each thread clips in has to be declared with a concrete type, so there is
    one set of kernels per precision.

    Parameters
    ----------
    ftype
        The :mod:`numba` type of the coordinates and weights.
    """

    num_pair, clip_cell = _build_shared(
        _jit,
        _jit(_source(rg.geometry.cross_2d)),
        np.float32(1) if ftype == numba.float32 else np.float64(1),
    )

    @cuda.jit
    def count_cells(x, y, num_output_x, num_output_y, counts):  # pragma: nocover
        """
        Count the output cells each input cell can touch.

        The prefix sum of this is where each cell writes its result, which is
        what lets the cells run independently.
        """
        index = cuda.grid(1)  # type: ignore[call-arg]
        num_y = x.shape[1] - 1
        if index >= (x.shape[0] - 1) * num_y:
            return
        index_x = index // num_y
        index_y = index - index_x * num_y

        counts[index] = num_pair(x, y, index_x, index_y, num_output_x, num_output_y)

    @cuda.jit
    def clip_cells(  # pragma: nocover
        x,
        y,
        weights_input,
        num_output_x,
        num_output_y,
        offset,
        indices_input,
        indices_output,
        values,
    ):
        """
        Clip every input cell against the output cells it touches.

        Slots which receive no overlap keep the sentinel index of ``-1``
        they were initialized with, and the zeros beside it, and are
        dropped by the caller.
        """
        index = cuda.grid(1)  # type: ignore[call-arg]
        num_y = x.shape[1] - 1
        if index >= (x.shape[0] - 1) * num_y:
            return
        index_x = index // num_y
        index_y = index - index_x * num_y

        # `numba` annotates the shape of a local array as a `local`, so each
        # of these needs the annotation waived
        subject_x = cuda.local.array(_num_slot, ftype)  # type: ignore[arg-type]
        subject_y = cuda.local.array(_num_slot, ftype)  # type: ignore[arg-type]
        clipped_x = cuda.local.array(_num_slot, ftype)  # type: ignore[arg-type]
        clipped_y = cuda.local.array(_num_slot, ftype)  # type: ignore[arg-type]

        clip_cell(
            x,
            y,
            weights_input,
            num_output_x,
            num_output_y,
            index_x,
            index_y,
            index,
            offset[index],
            subject_x,
            subject_y,
            clipped_x,
            clipped_y,
            indices_input,
            indices_output,
            values,
        )

    return count_cells, clip_cells


def _allocate_result(
    num: int,
    dtype: np.typing.DTypeLike,
) -> tuple[Any, Any, Any]:
    """
    Allocate the three result arrays, initialized as the host leaves them.

    A slot which sees no overlap is never written by the clipping, so it
    keeps whatever it was allocated with.  The host builds its result with
    :func:`numpy.full` and :func:`numpy.zeros`, and this leaves the same,
    rather than leaving a reader which forgets to drop those slots looking
    at arbitrary memory.

    Parameters
    ----------
    num
        The number of slots to reserve.
    dtype
        The type of the weights.
    """
    return (
        _cuda.fill(_cuda.allocate(num, np.int64), -1),
        _cuda.zeros(num, np.int64),
        _cuda.zeros(num, dtype),
    )


_kernels = {}
"""The kernels built so far, keyed by the floating-point type."""


def _prefix_sum(counts: Any, num_cell: int) -> tuple[Any, int]:
    """
    Compute the exclusive prefix sum of the per-cell counts, on the device.

    :mod:`numba` has no scan, so this borrows :func:`torch.cumsum`, which
    shares the memory rather than copying it.

    Parameters
    ----------
    counts
        The per-cell counts, on the device.
    num_cell
        The number of input cells.
    """
    try:
        # an optional dependency, so it is absent from the environment the
        # type checker runs in
        import torch  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: nocover
        raise ImportError(
            "building weights on a device needs `torch`, which provides the "
            "prefix sum; install `regridding[cuda]`"
        ) from error

    offset = torch.zeros(num_cell + 1, dtype=torch.int64, device="cuda")
    torch.cumsum(torch.as_tensor(counts, device="cuda"), dim=0, out=offset[1:])
    return offset, int(offset[~0].item())


def weights_conservative_2d_clipping_cuda(
    grid_input: tuple[np.ndarray, np.ndarray],
    grid_output: tuple[np.ndarray, np.ndarray],
    weights_input: None | np.ndarray = None,
    dtype: np.typing.DTypeLike = np.float64,
    threads: int = 128,
) -> tuple:
    """
    Compute 2D conservative weights on a CUDA device, and leave them there.

    The result uses the same convention as the host kernel, a flat
    ``(indices_input, indices_output, values)`` triple, but the three
    arrays are :class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`.
    They expose ``__cuda_array_interface__``, so :func:`torch.as_tensor`
    wraps them without copying.

    Parameters
    ----------
    grid_input
        The vertices of the old grid.  May already be on the device, in
        which case they are expected in output-cell units and nothing is
        transferred.
    grid_output
        The vertices of the new grid, which must be a uniform,
        axis-aligned lattice.
    weights_input
        Optional weights applied to the values of the input grid before
        resampling.
    dtype
        The floating-point type of the clipping and of the result.
    threads
        The number of threads in each block.
    """

    ftype = numba.float32 if np.dtype(dtype) == np.float32 else numba.float64
    if ftype not in _kernels:
        _kernels[ftype] = _build(ftype)
    count_cells, clip_cells = _kernels[ftype]

    x_output, y_output = grid_output
    num_output_x = x_output.shape[0] - 1
    num_output_y = y_output.shape[1] - 1

    x_input, y_input = grid_input
    if cuda.is_cuda_array(x_input):
        x, y = x_input, y_input
    else:
        origin_x = float(x_output[0, 0])
        origin_y = float(y_output[0, 0])
        step_x = float(x_output[1, 0] - x_output[0, 0])
        step_y = float(y_output[0, 1] - y_output[0, 0])
        x = cuda.to_device(
            np.ascontiguousarray(
                (np.asarray(x_input, float) - origin_x) / step_x, dtype
            )
        )
        y = cuda.to_device(
            np.ascontiguousarray(
                (np.asarray(y_input, float) - origin_y) / step_y, dtype
            )
        )

    num_x = x.shape[0] - 1
    num_y = x.shape[1] - 1
    num_cell = num_x * num_y
    blocks = (num_cell + threads - 1) // threads

    if weights_input is None:
        factor = _cuda.fill(_cuda.allocate((num_x, num_y), dtype), 1, threads)
    elif cuda.is_cuda_array(weights_input):
        factor = weights_input
    else:
        factor = cuda.to_device(np.ascontiguousarray(weights_input, dtype))

    counts = _cuda.allocate(num_cell, np.int64)
    count_cells[blocks, threads](x, y, num_output_x, num_output_y, counts)  # type: ignore[index]

    offset, num_total = _prefix_sum(counts, num_cell)

    indices_input, indices_output, values = _allocate_result(num_total, dtype)

    if not num_total:
        return indices_input, indices_output, values

    clip_cells[blocks, threads](  # type: ignore[index]
        x,
        y,
        factor,
        num_output_x,
        num_output_y,
        cuda.as_cuda_array(offset),
        indices_input,
        indices_output,
        values,
    )

    return indices_input, indices_output, values

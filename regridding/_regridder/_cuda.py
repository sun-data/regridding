"""
Assembling and applying a :class:`regridding.Regridder` on a CUDA device.

The matrix is held in :mod:`torch` tensors, which do the sorting and the
bookkeeping, and is applied by :mod:`numba` kernels.  Every step gives the
same result every time: entries are only ever counted with atomic
operations, which are exact on integers, and are placed by sorting.
"""

from typing import Any
import numpy as np
from numba import cuda
from regridding import _cuda

__all__ = [
    "assemble",
    "transpose",
    "scale",
    "matmul",
]

_warp = 32
"""The number of threads in a warp, the most which sum one row of the matrix."""

_chunk = 2**26
"""
How many entries of a matrix are transposed at a time, which bounds the
memory the sort takes beyond the matrix and its transpose.
"""


def _torch() -> Any:
    """Import :mod:`torch`, see :func:`regridding._regridder._regridder._torch`."""
    from ._regridder import _torch

    return _torch()


def _dtype(dtype: type) -> Any:
    """The :mod:`torch` type which stands for a :mod:`numpy` one."""
    torch = _torch()
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


def _place(
    keys: Any,
    indices: Any,
    values: Any,
    cursor: Any,
    indices_result: Any,
    data_result: Any,
) -> None:
    """
    Place a run of entries in the rows of a matrix in CSR form, after the
    entries already there, keeping their order within each row.

    The entries are sorted by row, stably, and each goes as far past its
    row's cursor as it is past the first entry of that row in the run.

    Parameters
    ----------
    keys
        The row of each entry.
    indices
        The column of each entry.
    values
        The value of each entry.
    cursor
        Where the next entry of each row goes, which is moved past the run.
    indices_result
        The columns of the matrix.
    data_result
        The values of the matrix.
    """
    torch = _torch()

    num = keys.shape[0]
    if num == 0:
        return

    order = torch.argsort(keys, stable=True)
    keys_sorted = keys[order]

    position = torch.arange(num, device=keys.device)
    first = torch.ones(num, dtype=torch.bool, device=keys.device)
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]
    start = torch.cummax(torch.where(first, position, 0), dim=0).values

    target = cursor[keys_sorted] + (position - start)
    indices_result[target] = indices[order].to(indices_result.dtype)
    data_result[target] = values[order].to(data_result.dtype)

    cursor.index_add_(0, keys, torch.ones_like(keys))


def assemble(
    blocks: list[tuple[Any, Any, Any]],
    offsets_rows: np.ndarray,
    offsets_columns: np.ndarray,
    lookup_rows: np.ndarray,
    lookup_columns: np.ndarray,
    num_rows: int,
    dtype_indices: type,
    dtype: type,
) -> tuple[Any, Any, Any]:
    """
    Assemble the matrix in CSR form on the device, one element of the
    weights at a time, as :func:`regridding._regridder._regridder._assemble`
    does on the host.

    The weights are read where they are: weights built on the device are
    viewed rather than copied, and weights on the host are sent one element
    at a time.  They are read twice, once to count the entries of each row
    and once to place them.

    Parameters
    ----------
    blocks
        The ``(indices_input, indices_output, values)`` of each element of
        the weights, in order.
    offsets_rows
        Where the rows of each element start in the matrix.
    offsets_columns
        Where the columns of each element start in the matrix.
    lookup_rows
        Where each output index lands in the rows, after the offset.
    lookup_columns
        Where each input index lands in the columns, after the offset.
    num_rows
        The number of rows of the matrix.
    dtype_indices
        The type of the column indices.
    dtype
        The type of the entries.

    Returns
    -------
    Where each row starts, the column of each entry and its value, as
    :mod:`torch` tensors on the device.
    """
    torch = _torch()
    device = "cuda"

    lookup_rows = torch.as_tensor(lookup_rows, device=device)
    lookup_columns = torch.as_tensor(lookup_columns, device=device)

    def entries(b: int) -> tuple[Any, Any, Any]:
        """The rows, columns and values of the entries of element `b`."""
        indices_input, indices_output, values = blocks[b]
        indices_input = torch.as_tensor(indices_input, device=device)
        indices_output = torch.as_tensor(indices_output, device=device)
        values = torch.as_tensor(values, device=device)
        keep = (indices_input >= 0) & (indices_output >= 0)
        rows = int(offsets_rows[b]) + lookup_rows[indices_output[keep].long()]
        columns = int(offsets_columns[b]) + lookup_columns[indices_input[keep].long()]
        return rows, columns, values[keep]

    counts = torch.zeros(num_rows, dtype=torch.int64, device=device)
    for b in range(len(blocks)):
        rows, _, _ = entries(b)
        counts.index_add_(0, rows, torch.ones_like(rows))

    indptr = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=indptr[1:])
    del counts

    num_entries = int(indptr[~0])
    indices = torch.empty(num_entries, dtype=_dtype(dtype_indices), device=device)
    data = torch.empty(num_entries, dtype=_dtype(dtype), device=device)

    cursor = indptr[:~0].clone()
    for b in range(len(blocks)):
        rows, columns, values = entries(b)
        _place(rows, columns, values, cursor, indices, data)

    return indptr, indices, data


def transpose(
    indptr: Any,
    indices: Any,
    data: Any,
    num_columns: int,
    dtype_indices: type,
) -> tuple[Any, Any, Any]:
    """
    Transpose a matrix in CSR form on the device, keeping the entries of
    each row of the transpose in the order of the rows they came from, as
    the host does.

    The entries are taken :data:`_chunk` at a time, in order, so the memory
    the sort takes does not grow with the matrix.

    Parameters
    ----------
    indptr
        Where each row starts, as a :mod:`torch` tensor on the device.
    indices
        The column of each entry.
    data
        The value of each entry.
    num_columns
        The number of columns of the matrix.
    dtype_indices
        The type of the column indices of the transpose.
    """
    torch = _torch()
    device = indptr.device

    num_entries = indices.shape[0]

    counts = torch.zeros(num_columns, dtype=torch.int64, device=device)
    for start in range(0, num_entries, _chunk):
        columns = indices[start : start + _chunk].long()
        counts.index_add_(0, columns, torch.ones_like(columns))

    indptr_transposed = torch.zeros(num_columns + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=indptr_transposed[1:])
    del counts

    indices_transposed = torch.empty(
        num_entries, dtype=_dtype(dtype_indices), device=device
    )
    data_transposed = torch.empty_like(data)

    cursor = indptr_transposed[:~0].clone()
    for start in range(0, num_entries, _chunk):
        stop = min(start + _chunk, num_entries)
        position = torch.arange(start, stop, device=device)
        rows = torch.searchsorted(indptr, position, right=True) - 1
        _place(
            keys=indices[start:stop].long(),
            indices=rows,
            values=data[start:stop],
            cursor=cursor,
            indices_result=indices_transposed,
            data_result=data_transposed,
        )

    return indptr_transposed, indices_transposed, data_transposed


def scale(
    indptr: Any,
    indices: Any,
    data: Any,
    scale: float,
    table_rows: np.ndarray,
    table_columns: np.ndarray,
    factor: np.ndarray,
    volume: np.ndarray,
) -> Any:
    """
    Scale each entry of a transposed matrix by the factor of its input cell
    over the volume of its output cell, on the device, as
    :func:`regridding._regridder._regridder._scale` does on the host.

    Parameters
    ----------
    indptr
        Where each row starts, as a :mod:`torch` tensor on the device.
    indices
        The column of each entry.
    data
        The value of each entry.
    scale
        What a unit of the values stands for, which they are multiplied by
        first.
    table_rows
        The axes of the rows, see :func:`regridding._regridder._regridder._scale`.
    table_columns
        The axes of the columns.
    factor
        The flat factor of each input cell, on the host.
    volume
        The flat volume of each output cell, on the host.
    """
    torch = _torch()

    if scale != 1:
        data = data * scale
    result = torch.empty_like(data)

    num_rows = indptr.shape[0] - 1
    blocks = (num_rows + _cuda.threads - 1) // _cuda.threads
    if blocks:
        _scale[blocks, _cuda.threads](  # type: ignore[index]
            cuda.as_cuda_array(indptr),
            cuda.as_cuda_array(indices),
            cuda.as_cuda_array(data),
            cuda.to_device(table_rows),
            cuda.to_device(table_columns),
            cuda.to_device(factor),
            cuda.to_device(volume),
            cuda.as_cuda_array(result),
        )
    return result


def _width(num_entries: int, num_rows: int) -> int:
    """
    The number of threads which sum each row of a matrix: the number of
    entries in an average row, rounded down to a power of two, up to a warp.

    Fewer threads waste less of a warp on short rows, and more read long
    rows faster.  On an RTX 4090, this was the fastest width, or close to
    it, for the operators of MART: two threads for a backprojection, with
    about three entries a row, and a warp for the forward operators, with
    tens to thousands.

    Parameters
    ----------
    num_entries
        The number of entries of the matrix.
    num_rows
        The number of rows of the matrix.
    """
    width = 1
    while width < _warp and 2 * width * num_rows <= num_entries:
        width *= 2
    return width


def matmul(
    indptr: Any,
    indices: Any,
    data: Any,
    x: Any,
    y: Any,
) -> None:
    """
    Multiply a matrix in CSR form by a dense matrix on the device, writing
    the product to `y`.

    Each row is summed by a group of threads, as many as :func:`_width`
    gives the matrix: each thread of a group sums every so many entries of
    the row, in order, and the sums of the group are then added in a fixed
    tree.  The size of the groups depends only on the matrix, so the result
    is the same every time, and the same for each column of `x` however
    many columns there are.

    Parameters
    ----------
    indptr
        Where each row starts in `indices` and `data`, on the device.
    indices
        The column of each entry, on the device.
    data
        The value of each entry, on the device.
    x
        The dense matrix, with a row for each column of the sparse one.
    y
        The product, with a row for each row of the sparse one.
    """
    num_rows = y.shape[0]
    width = _width(indices.shape[0], num_rows)
    blocks = (num_rows * width + _cuda.threads - 1) // _cuda.threads
    if blocks:
        _matmul[blocks, _cuda.threads](  # type: ignore[index]
            indptr, indices, data, x, y, width
        )


# these run on the device, where `coverage` cannot follow them, so it reports
# their bodies as missed even when they do the work.  The arrays are device
# arrays, which `numba` ships no type for, so they are annotated as `Any`
@cuda.jit
def _matmul(
    indptr: Any,
    indices: Any,
    data: Any,
    x: Any,
    y: Any,
    width: int,
) -> None:  # pragma: nocover
    """
    Sum one row of the product for each group of `width` threads, see
    :func:`matmul`.
    """
    thread: int = cuda.grid(1)  # type: ignore[call-arg, assignment]
    row = thread // width
    lane = thread % width

    num_rows, num_batch = y.shape

    # every thread of a warp takes part in the shuffles below, so threads
    # past the last row stay, with nothing to sum, rather than leave
    valid = row < num_rows
    start = indptr[row] if valid else 0
    stop = indptr[row + 1] if valid else 0

    for t in range(num_batch):
        total = 0.0
        for k in range(start + lane, stop, width):
            total += data[k] * x[indices[k], t]
        # the first thread of each group gathers the sums of the others; the
        # shuffles of the other threads may reach into the next group, but
        # what they gather is not used
        offset = width // 2
        while offset > 0:
            total += cuda.shfl_down_sync(0xFFFFFFFF, total, offset)
            offset //= 2
        if valid and lane == 0:
            y[row, t] = total


@cuda.jit
def _scale(
    indptr: Any,
    indices: Any,
    data: Any,
    table_rows: Any,
    table_columns: Any,
    factor: Any,
    volume: Any,
    result: Any,
) -> None:  # pragma: nocover
    """Scale the entries of one row for each thread, see :func:`scale`."""
    row: int = cuda.grid(1)  # type: ignore[call-arg, assignment]
    if row >= indptr.shape[0] - 1:
        return

    offset_factor = 0
    offset_volume = 0
    for a in range(table_rows.shape[0]):
        i = (row // table_rows[a, 0]) % table_rows[a, 1]
        offset_factor += i * table_rows[a, 2]
        offset_volume += i * table_rows[a, 3]

    for k in range(indptr[row], indptr[row + 1]):
        c = indices[k]
        f = offset_factor
        v = offset_volume
        for a in range(table_columns.shape[0]):
            i = (c // table_columns[a, 0]) % table_columns[a, 1]
            f += i * table_columns[a, 2]
            v += i * table_columns[a, 3]
        result[k] = data[k] * factor[f] / volume[v]

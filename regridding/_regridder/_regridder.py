from typing import Any, Sequence
import dataclasses
import math
import numpy as np
import numba
from numba import cuda
from numba.typed.typedlist import List as TypedList
from regridding import _util
from regridding._weights._weights_transposed._weights_transposed import (
    _check_grids,
    _factors,
)

__all__ = [
    "Regridder",
]


@dataclasses.dataclass(eq=False)
class Regridder:
    r"""
    A sparse linear operator which resamples arrays from one grid onto
    another, stored as a single matrix in compressed sparse row (CSR) form.

    The weights computed by :func:`regridding.weights` hold one sparse matrix
    for each element along the orthogonal axes, the axes which are not
    resampled.  This assembles them into one matrix, which is applied in a
    single pass over its rows, with every row summed in a fixed order, so
    that the result is the same every time on the host and on a device.

    The operator maps arrays of shape :attr:`shape_input` onto arrays of
    shape :attr:`shape_output`.  Along the resampled axes, these are the
    shapes of the grids.  Along each orthogonal axis, they say what the
    operator does there:

    =======  =====  ======  ==============================================
    weights  input  output  the operator
    =======  =====  ======  ==============================================
    N        N      N       applies each element to its own input
    N        1      N       applies every element to the same input
    N        N      1       sums what the elements give
    N        1      1       applies the sum of the elements
    1        M      M       applies the same matrix to every input
    =======  =====  ======  ==============================================

    The first four are folded into the matrix, which has as many entries as
    the weights whatever the shapes.  The last is a batch axis, which the
    matrix is applied along, as it is along any leading axes the values have
    beyond :attr:`shape_input`.

    Transposing the operator swaps the two shapes, so the elements applied
    to the same input are summed instead, and vice versa.

    Build one with :meth:`from_weights`.
    """

    shape_input: tuple[int, ...]
    """The shape of the arrays this operator accepts."""

    shape_output: tuple[int, ...]
    """The shape of the arrays this operator returns."""

    axis_input: tuple[int, ...]
    """The resampled axes of the input, ascending and non-negative."""

    axis_output: tuple[int, ...]
    """The resampled axes of the output, ascending and non-negative."""

    shape_weights: tuple[int, ...]
    """The length of the weights along each orthogonal axis, in order."""

    indptr: np.ndarray
    """Where each row of the matrix starts in :attr:`indices` and :attr:`data`."""

    indices: np.ndarray
    """The column of each entry of the matrix."""

    data: np.ndarray
    """The value of each entry of the matrix."""

    unit: Any = None
    """The unit of the weights, if they have one."""

    device: None | str = None
    """
    Where the operator is applied: on the host if :obj:`None`, or on a CUDA
    device if ``"cuda"``.
    """

    _transposed: "None | Regridder" = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    _arrays_device: None | tuple[Any, Any, Any] = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    _layouts: dict[tuple[int, ...], "_Layout"] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_weights(
        cls,
        weights: tuple[np.ndarray, tuple[int, ...], tuple[int, ...]],
        axis_input: None | int | Sequence[int] = None,
        axis_output: None | int | Sequence[int] = None,
        shape_input: None | tuple[int, ...] = None,
        shape_output: None | tuple[int, ...] = None,
        device: None | str = None,
    ) -> "Regridder":
        """
        Assemble an operator from weights computed by
        :func:`regridding.weights`.

        Parameters
        ----------
        weights
            The weights computed by :func:`regridding.weights`, with the
            shapes of the grids they were computed for.  Weights in device
            memory are brought back to the host, dropping their empty slots.
        axis_input
            The resampled axes of the input, as given to
            :func:`regridding.weights`.
        axis_output
            The resampled axes of the output, as given to
            :func:`regridding.weights`.
        shape_input
            The shape of the arrays the operator accepts.  If :obj:`None`,
            the shape of the input grid, so that the operator applies each
            element of the weights to its own input.
        shape_output
            The shape of the arrays the operator returns.  If :obj:`None`,
            the shape of the output grid.
        device
            Where to apply the operator, see :attr:`device`.
        """
        array, shape_grid_input, shape_grid_output = weights

        array = np.asarray(array)
        shape_weights = array.shape
        num_orthogonal = array.ndim

        # the resampled axes are counted from the end, as
        # `regridding.regrid_from_weights` counts them, since the grids can
        # have fewer orthogonal axes than the weights, which broadcast them
        axis_input = _util._normalize_axis(axis_input, len(shape_grid_input))
        axis_output = _util._normalize_axis(axis_output, len(shape_grid_output))

        if len(axis_input) != len(axis_output):
            raise ValueError(
                f"{axis_input=} and {axis_output=} should have the same number "
                f"of axes"
            )

        ndim_input = num_orthogonal + len(axis_input)
        ndim_output = num_orthogonal + len(axis_output)

        shape_cells_input = _shape_cells(
            shape_grid_input, axis_input, shape_weights, ndim_input
        )
        shape_cells_output = _shape_cells(
            shape_grid_output, axis_output, shape_weights, ndim_output
        )

        if shape_input is None:
            shape_input = shape_cells_input
        if shape_output is None:
            shape_output = shape_cells_output

        shape_input = tuple(int(n) for n in shape_input)
        shape_output = tuple(int(n) for n in shape_output)

        axis_input = _axes(axis_input, ndim_input)
        axis_output = _axes(axis_output, ndim_output)

        for shape, shape_cells, axis, name in (
            (shape_input, shape_cells_input, axis_input, "input"),
            (shape_output, shape_cells_output, axis_output, "output"),
        ):
            if len(shape) != len(shape_cells):
                raise ValueError(
                    f"the {name} should have {len(shape_cells)} axes, one for "
                    f"each orthogonal axis of the weights and each resampled "
                    f"axis, but has a shape of {shape}"
                )
            for ax in axis:
                if shape[ax] != shape_cells[ax]:
                    raise ValueError(
                        f"the {name} has {shape[ax]} elements along the "
                        f"resampled axis {ax}, but the grid the weights were "
                        f"built for has {shape_cells[ax]}"
                    )

        orthogonal_input = _orthogonal(len(shape_input), axis_input)
        orthogonal_output = _orthogonal(len(shape_output), axis_output)

        for k in range(num_orthogonal):
            num_weights = shape_weights[k]
            num_input = shape_input[orthogonal_input[k]]
            num_output = shape_output[orthogonal_output[k]]
            if num_weights > 1:
                if num_input not in (1, num_weights) or num_output not in (
                    1,
                    num_weights,
                ):
                    raise ValueError(
                        f"the weights have {num_weights} elements along "
                        f"orthogonal axis {k}, so the input and output should "
                        f"have either 1 or {num_weights} along it, but they "
                        f"have {num_input} and {num_output}"
                    )
            elif num_input != num_output:
                raise ValueError(
                    f"the weights are the same along orthogonal axis {k}, so "
                    f"the input and output should have the same length along "
                    f"it, but they have {num_input} and {num_output}"
                )

        result = cls(
            shape_input=shape_input,
            shape_output=shape_output,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_weights=shape_weights,
            indptr=np.empty(0, dtype=np.int64),
            indices=np.empty(0, dtype=np.int64),
            data=np.empty(0),
            device=device,
        )

        strides_columns = result._strides_input
        strides_rows = result._strides_output
        lookup_columns = _lookup(shape_input, axis_input, strides_columns)
        lookup_rows = _lookup(shape_output, axis_output, strides_rows)

        # `numba.typed.List()` is declared to return a plain `list` when Numba's
        # JIT is disabled, a mode this library is not usable in.
        list_input: TypedList = TypedList()  # type: ignore[assignment]
        list_output: TypedList = TypedList()  # type: ignore[assignment]
        list_values: TypedList = TypedList()  # type: ignore[assignment]
        offsets_columns = []
        offsets_rows = []
        unit = None
        for index in np.ndindex(*shape_weights):
            indices_input, indices_output, values = array[index]
            if cuda.is_cuda_array(values):
                indices_input = indices_input.copy_to_host()
                indices_output = indices_output.copy_to_host()
                values = values.copy_to_host()
            unit_values = getattr(values, "unit", None)
            if unit_values is not None:
                unit = unit_values
            values = np.asarray(getattr(values, "value", values), dtype=np.float64)
            list_input.append(indices_input)
            list_output.append(indices_output)
            list_values.append(values)
            offsets_columns.append(
                _offset(
                    index, orthogonal_input, shape_input, strides_columns, shape_weights
                )
            )
            offsets_rows.append(
                _offset(
                    index, orthogonal_output, shape_output, strides_rows, shape_weights
                )
            )

        num_rows, num_columns = result.shape_matrix

        starts = np.zeros(len(list_values) + 1, dtype=np.int64)
        starts[1:] = np.cumsum([v.shape[0] for v in list_values])
        offsets_rows = np.array(offsets_rows, dtype=np.int64)
        offsets_columns = np.array(offsets_columns, dtype=np.int64)

        cursors = _assemble_count(
            list_input,
            list_output,
            offsets_rows,
            lookup_rows,
            starts,
            num_rows,
            _num_chunks(num_rows),
        )
        indptr = _cursors(cursors)

        num_entries = int(indptr[~0])
        indices = np.empty(num_entries, dtype=_index_dtype(num_columns))
        data = np.empty(num_entries)
        _assemble_scatter(
            list_input,
            list_output,
            list_values,
            offsets_rows,
            lookup_rows,
            offsets_columns,
            lookup_columns,
            starts,
            cursors,
            indices,
            data,
        )

        result.indptr = indptr
        result.indices = indices
        result.data = data
        result.unit = unit

        return result

    @property
    def _orthogonal_input(self) -> tuple[int, ...]:
        """The orthogonal axes of the input, in order."""
        return _orthogonal(len(self.shape_input), self.axis_input)

    @property
    def _orthogonal_output(self) -> tuple[int, ...]:
        """The orthogonal axes of the output, in order."""
        return _orthogonal(len(self.shape_output), self.axis_output)

    @property
    def _batch_input(self) -> tuple[int, ...]:
        """The axes of the input which the matrix is applied along."""
        return tuple(
            ax for ax, n in zip(self._orthogonal_input, self.shape_weights) if n == 1
        )

    @property
    def _batch_output(self) -> tuple[int, ...]:
        """The axes of the output which the matrix is applied along."""
        return tuple(
            ax for ax, n in zip(self._orthogonal_output, self.shape_weights) if n == 1
        )

    @property
    def _matrix_input(self) -> tuple[int, ...]:
        """The axes of the input which index the columns of the matrix."""
        batch = self._batch_input
        return tuple(ax for ax in range(len(self.shape_input)) if ax not in batch)

    @property
    def _matrix_output(self) -> tuple[int, ...]:
        """The axes of the output which index the rows of the matrix."""
        batch = self._batch_output
        return tuple(ax for ax in range(len(self.shape_output)) if ax not in batch)

    @property
    def _strides_input(self) -> dict[int, int]:
        """How far apart in the columns the elements along each axis are."""
        return _strides(self.shape_input, self._matrix_input)

    @property
    def _strides_output(self) -> dict[int, int]:
        """How far apart in the rows the elements along each axis are."""
        return _strides(self.shape_output, self._matrix_output)

    @property
    def shape_matrix(self) -> tuple[int, int]:
        """The number of rows and columns of the matrix."""
        num_rows = math.prod(self.shape_output[ax] for ax in self._matrix_output)
        num_columns = math.prod(self.shape_input[ax] for ax in self._matrix_input)
        return num_rows, num_columns

    @property
    def num_entries(self) -> int:
        """The number of entries stored in the matrix."""
        return int(self.indptr[~0])

    def to(self, device: None | str) -> "Regridder":
        """
        The same operator, applied on the host or on a device.

        Parameters
        ----------
        device
            Where to apply the operator, see :attr:`device`.
        """
        return dataclasses.replace(self, device=device)

    @property
    def T(self) -> "Regridder":
        """
        The exact transpose of this operator.

        It is computed once and kept, and its own transpose is this
        operator.
        """
        if self._transposed is None:
            num_rows, num_columns = self.shape_matrix
            num_chunks = _num_chunks(num_columns)
            # each chunk of rows has about as many entries as the others
            bounds = np.searchsorted(
                self.indptr,
                np.linspace(0, self.num_entries, num_chunks + 1),
                side="left",
            )
            bounds[0] = 0
            bounds[~0] = num_rows
            cursors = _transpose_count(self.indptr, self.indices, num_columns, bounds)
            indptr = _cursors(cursors)
            indices = np.empty(self.num_entries, dtype=_index_dtype(num_rows))
            data = np.empty(self.num_entries)
            _transpose_scatter(
                self.indptr, self.indices, self.data, bounds, cursors, indices, data
            )
            result = Regridder(
                shape_input=self.shape_output,
                shape_output=self.shape_input,
                axis_input=self.axis_output,
                axis_output=self.axis_input,
                shape_weights=self.shape_weights,
                indptr=indptr,
                indices=indices,
                data=data,
                unit=self.unit,
                device=self.device,
            )
            result._transposed = self
            self._transposed = result
        return self._transposed

    def transpose_conservative(
        self,
        coordinates_input: np.ndarray | tuple[np.ndarray, ...],
        coordinates_output: np.ndarray | tuple[np.ndarray, ...],
        weights_input: None | np.ndarray = None,
    ) -> "Regridder":
        """
        The transpose of this operator, normalized to conserve flux, as
        :func:`regridding.transpose_weights_conservative` computes it.

        Each entry of the exact transpose is multiplied by the volume of its
        input cell, divided by the square of that cell's weight, and divided
        by the volume of its output cell.  The cells are those of the element
        of the weights the entry came from, so this is done for each entry
        rather than as two scalings of the whole matrix: an element whose
        input is shared with others, or whose output is summed with others',
        can still have cells of its own size.  The result shares the indices
        of :attr:`T`.

        Parameters
        ----------
        coordinates_input
            The vertices of the input grid given to :func:`regridding.weights`.
        coordinates_output
            The vertices of the output grid given to :func:`regridding.weights`.
        weights_input
            The weights applied to the input values by
            :func:`regridding.weights`, if any.

        Raises
        ------
        NotImplementedError
            If an orthogonal axis is summed by the operator after its input
            is shared along it, and either grid varies along it, since the
            element an entry came from is then lost.
        """
        ndim_input = len(self.shape_input)
        ndim_output = len(self.shape_output)

        (
            coordinates_input,
            coordinates_output,
            axis_input,
            axis_output,
            _,
            _,
            _,
        ) = _util._normalize_input_output_coordinates(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=tuple(ax - ndim_input for ax in self.axis_input),
            axis_output=tuple(ax - ndim_output for ax in self.axis_output),
        )
        axis_input = tuple(sorted(axis_input))
        axis_output = tuple(sorted(axis_output))

        shape_cells_input = self._cells(self.shape_input, self._orthogonal_input)
        shape_cells_output = self._cells(self.shape_output, self._orthogonal_output)

        _check_grids(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_input=shape_cells_input,
            shape_output=shape_cells_output,
        )

        factor, volume = _factors(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_input=shape_cells_input,
            weights_input=weights_input,
        )
        factor, strides_factor = _flat(factor, shape_cells_input, "input")
        volume, strides_volume = _flat(volume, shape_cells_output, "output")

        orthogonal_input = self._orthogonal_input
        orthogonal_output = self._orthogonal_output

        # each entry of the transpose finds the cells of the element it came
        # from in its row, which is a column of this operator, and its
        # column, which is a row of this operator: the resampled axes of
        # each side, and each orthogonal axis in whichever side keeps it
        table_rows = []
        for ax, stride in self._strides_input.items():
            if self.shape_input[ax] == 1:
                continue
            coefficient_volume = 0
            if ax in orthogonal_input:
                k = orthogonal_input.index(ax)
                coefficient_volume = strides_volume[orthogonal_output[k]]
            table_rows.append(
                (stride, self.shape_input[ax], strides_factor[ax], coefficient_volume)
            )

        table_columns = []
        for ax, stride in self._strides_output.items():
            if self.shape_output[ax] == 1:
                continue
            coefficient_factor = 0
            coefficient_volume = strides_volume[ax]
            if ax in orthogonal_output:
                k = orthogonal_output.index(ax)
                if self.shape_input[orthogonal_input[k]] > 1:
                    # already found in the row
                    coefficient_volume = 0
                else:
                    coefficient_factor = strides_factor[orthogonal_input[k]]
            table_columns.append(
                (stride, self.shape_output[ax], coefficient_factor, coefficient_volume)
            )

        for k in range(len(orthogonal_input)):
            ax_input = orthogonal_input[k]
            ax_output = orthogonal_output[k]
            lost = self.shape_input[ax_input] == 1 and self.shape_output[ax_output] == 1
            varies = strides_factor[ax_input] != 0 or strides_volume[ax_output] != 0
            if lost and varies:
                raise NotImplementedError(
                    f"orthogonal axis {k} is summed after the input is shared "
                    f"along it, and the grids vary along it, so the element "
                    f"each entry came from is lost"
                )

        transposed = self.T

        data = transposed.data
        if self.unit is not None:
            data = _util._dimensionless(data << self.unit, strict=False)

        result = np.empty_like(data)
        _scale(
            transposed.indptr,
            transposed.indices,
            data,
            np.array(table_rows, dtype=np.int64).reshape(-1, 4),
            np.array(table_columns, dtype=np.int64).reshape(-1, 4),
            factor,
            volume,
            result,
        )

        return Regridder(
            shape_input=transposed.shape_input,
            shape_output=transposed.shape_output,
            axis_input=transposed.axis_input,
            axis_output=transposed.axis_output,
            shape_weights=self.shape_weights,
            indptr=transposed.indptr,
            indices=transposed.indices,
            data=result,
            device=self.device,
        )

    def _cells(
        self,
        shape: tuple[int, ...],
        orthogonal: tuple[int, ...],
    ) -> tuple[int, ...]:
        """
        The shape of the cells of a grid the weights were built for: `shape`
        along the resampled axes, and the length of the weights along the
        orthogonal ones.
        """
        result = list(shape)
        for ax, n in zip(orthogonal, self.shape_weights):
            result[ax] = n
        return tuple(result)

    def __call__(self, values: Any) -> Any:
        """
        Apply this operator to an array of values.

        Parameters
        ----------
        values
            An array which broadcasts to :attr:`shape_input`, after any
            leading axes it has beyond it, which the operator is applied
            along.  On a device, a :mod:`torch` tensor or another array which
            :func:`torch.as_tensor` accepts, and the result is of the same
            kind.
        """
        if self.device is not None:
            return self._call_device(values)

        unit = getattr(values, "unit", None)
        values = np.asarray(getattr(values, "value", values))

        layout = self._layout(self._shape_lead(values.shape))
        try:
            values = np.broadcast_to(values, layout.shape_input)
        except ValueError as error:
            raise ValueError(
                f"values of shape {values.shape} cannot be broadcast to the "
                f"input of this operator, {self.shape_input}"
            ) from error

        x = np.ascontiguousarray(values.transpose(layout.order_input), dtype=float)
        x = x.reshape(layout.num_columns, -1)
        y = np.zeros((layout.num_rows, x.shape[1]))

        _matmul(self.indptr, self.indices, self.data, x, y, numba.get_num_threads())

        result = y.reshape(layout.shape_result).transpose(layout.order_result)

        if self.unit is not None:
            unit = self.unit if unit is None else unit * self.unit
        if unit is not None:
            result = result << unit

        return result

    def _shape_lead(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The leading axes of values of `shape`, beyond :attr:`shape_input`."""
        return shape[: max(len(shape) - len(self.shape_input), 0)]

    def _layout(self, shape_lead: tuple[int, ...]) -> "_Layout":
        """
        How values with leading axes of `shape_lead` are laid out as a dense
        matrix with a row for each column of this operator, and how the
        product is laid out as the result.

        The axes which index the matrix come first, followed by the leading
        axes and the batch axes, which become the columns of the dense
        matrix.  It is worked out once for each `shape_lead`, since an
        operator is usually applied to the same shape many times.
        """
        if shape_lead in self._layouts:
            return self._layouts[shape_lead]

        num_lead = len(shape_lead)
        lead = list(range(num_lead))
        order_input = (
            [num_lead + ax for ax in self._matrix_input]
            + lead
            + [num_lead + ax for ax in self._batch_input]
        )
        order_output = (
            [num_lead + ax for ax in self._matrix_output]
            + lead
            + [num_lead + ax for ax in self._batch_output]
        )
        shape_result = (
            tuple(self.shape_output[ax] for ax in self._matrix_output)
            + shape_lead
            + tuple(self.shape_output[ax] for ax in self._batch_output)
        )
        num_rows, num_columns = self.shape_matrix
        result = _Layout(
            order_input=tuple(order_input),
            order_result=tuple(int(ax) for ax in np.argsort(order_output)),
            shape_input=shape_lead + self.shape_input,
            shape_result=shape_result,
            num_rows=num_rows,
            num_columns=num_columns,
        )
        self._layouts[shape_lead] = result
        return result

    def _call_device(self, values: Any) -> Any:
        """Apply this operator on the device, see :meth:`__call__`."""
        try:
            # an optional dependency, so it is absent from the environment the
            # type checker runs in
            import torch  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: nocover
            raise ImportError(
                "applying a `Regridder` on a device needs `torch`, which lays "
                "out the values; install `regridding[cuda]`"
            ) from error
        from . import _cuda as _cuda_regridder

        is_tensor = isinstance(values, torch.Tensor)
        values = torch.as_tensor(values, device="cuda")

        layout = self._layout(self._shape_lead(tuple(values.shape)))
        try:
            values = values.broadcast_to(layout.shape_input)
        except RuntimeError as error:
            raise ValueError(
                f"values of shape {tuple(values.shape)} cannot be broadcast to "
                f"the input of this operator, {self.shape_input}"
            ) from error

        x = values.permute(layout.order_input).to(torch.float64).contiguous()
        x = x.reshape(layout.num_columns, -1)
        y = torch.empty(
            (layout.num_rows, x.shape[1]), dtype=torch.float64, device="cuda"
        )

        if self._arrays_device is None:
            self._arrays_device = (
                cuda.to_device(self.indptr),
                cuda.to_device(self.indices),
                cuda.to_device(self.data),
            )
        indptr, indices, data = self._arrays_device

        _cuda_regridder.matmul(
            indptr,
            indices,
            data,
            cuda.as_cuda_array(x),
            cuda.as_cuda_array(y),
        )

        result = y.reshape(layout.shape_result).permute(layout.order_result)

        if is_tensor:
            return result
        return cuda.as_cuda_array(result.contiguous())


@dataclasses.dataclass(frozen=True)
class _Layout:
    """How the values given to a :class:`Regridder` are laid out, see
    :meth:`Regridder._layout`."""

    order_input: tuple[int, ...]
    """The order to transpose the values to."""

    order_result: tuple[int, ...]
    """The order to transpose the product to, to give the result."""

    shape_input: tuple[int, ...]
    """The shape the values are broadcast to, leading axes included."""

    shape_result: tuple[int, ...]
    """The shape of the product, before it is transposed."""

    num_rows: int
    """The number of rows of the matrix."""

    num_columns: int
    """The number of columns of the matrix."""


def _axes(
    axis: None | int | Sequence[int],
    ndim: int,
) -> tuple[int, ...]:
    """Normalize resampled axes to be non-negative and ascending."""
    return tuple(sorted(ax % ndim for ax in _util._normalize_axis(axis, ndim)))


def _shape_cells(
    shape_grid: tuple[int, ...],
    axis: tuple[int, ...],
    shape_weights: tuple[int, ...],
    ndim: int,
) -> tuple[int, ...]:
    """
    The shape of the cells of a grid broadcast against its weights: the
    grid's lengths along the resampled axes and the weights' along the
    orthogonal ones.

    Parameters
    ----------
    shape_grid
        The shape of the cells of the grid, as :func:`regridding.weights`
        returns it.
    axis
        The resampled axes, counted from the end.
    shape_weights
        The shape of the weights, one length for each orthogonal axis.
    ndim
        The number of axes of the result.
    """
    result = [0] * ndim
    for ax in axis:
        result[ax] = int(shape_grid[ax])
    orthogonal = _orthogonal(ndim, tuple(ax % ndim for ax in axis))
    for ax, n in zip(orthogonal, shape_weights):
        result[ax] = int(n)
    return tuple(result)


def _orthogonal(
    ndim: int,
    axis: tuple[int, ...],
) -> tuple[int, ...]:
    """The axes of an array with `ndim` axes which are not in `axis`."""
    return tuple(ax for ax in range(ndim) if ax not in axis)


def _strides(
    shape: tuple[int, ...],
    axis: tuple[int, ...],
) -> dict[int, int]:
    """
    The strides, in elements, of a C-ordered array of `shape` restricted to
    the axes `axis`, keyed by axis.
    """
    result = {}
    stride = 1
    for ax in reversed(axis):
        result[ax] = stride
        stride *= shape[ax]
    return dict(sorted(result.items()))


def _lookup(
    shape: tuple[int, ...],
    axis: tuple[int, ...],
    strides: dict[int, int],
) -> np.ndarray:
    """
    Where each flat cell index of a set of weights lands in the rows or
    columns of the matrix, given the `strides` of the resampled axes `axis`.

    The flat cell indices count the resampled axes in ascending order, as
    :func:`regridding.regrid_from_weights` does.
    """
    shape_cells = tuple(shape[ax] for ax in axis)
    result = np.zeros(shape_cells, dtype=np.int64)
    for j, ax in enumerate(axis):
        index = np.arange(shape[ax], dtype=np.int64) * strides[ax]
        result = result + index.reshape((-1,) + (1,) * (len(axis) - j - 1))
    return result.reshape(-1)


def _offset(
    index: tuple[int, ...],
    orthogonal: tuple[int, ...],
    shape: tuple[int, ...],
    strides: dict[int, int],
    shape_weights: tuple[int, ...],
) -> int:
    """
    Where the element of the weights at `index` starts in the rows or
    columns of the matrix.

    An orthogonal axis which the values have a length of one along is shared
    by every element along it, so it does not move them.
    """
    return sum(
        index[k] * strides[ax]
        for k, ax in enumerate(orthogonal)
        if shape_weights[k] > 1 and shape[ax] > 1
    )


def _index_dtype(num: int) -> type:
    """The narrowest integer type which can index `num` elements."""
    if num <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64  # pragma: nocover


def _flat(
    a: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Flatten an array defined on the cells of a grid, which may have a length
    of one along axes it does not vary along.

    Parameters
    ----------
    a
        The array, with no more axes than `shape`.
    shape
        The shape of the cells of the grid.
    name
        Which grid it is defined on, should it not fit.

    Returns
    -------
    The flat array, and the stride of each axis of `shape` in it, in
    elements, which is zero along axes it does not vary along.
    """
    a = np.asarray(a, dtype=np.float64)
    a = a.reshape((1,) * (len(shape) - a.ndim) + a.shape)
    if np.broadcast_shapes(a.shape, shape) != shape:
        raise ValueError(
            f"the {name} grid has cells of shape {a.shape}, which do not fit "
            f"the {shape} the weights were built for"
        )
    a = np.ascontiguousarray(a)
    strides = tuple(
        0 if n == 1 else s // a.itemsize for n, s in zip(a.shape, a.strides)
    )
    return a.reshape(-1), strides


_num_counts = 2**25
"""
The most counts the histograms of a parallel sort may hold between them.

Each chunk of a sort counts its own entries in every row of the result, so
the histograms grow with the number of chunks times the number of rows.
Past this, fewer chunks are used than there are threads.
"""


def _num_chunks(num_rows: int) -> int:
    """
    How many chunks to sort the entries of a matrix into `num_rows` rows
    with: one for each thread, unless the histograms would be too large.
    """
    return max(1, min(numba.get_num_threads(), _num_counts // max(num_rows, 1)))


@numba.njit(cache=True, parallel=True)
def _assemble_count(
    indices_input: TypedList,
    indices_output: TypedList,
    offsets_rows: np.ndarray,
    lookup_rows: np.ndarray,
    starts: np.ndarray,
    num_rows: int,
    num_chunks: int,
) -> np.ndarray:
    """
    Count the entries of the weights in each row of the matrix, skipping
    empty slots, separately for each of `num_chunks` equal runs of the
    entries, taken in the order of the elements of the weights.

    Parameters
    ----------
    indices_input
        The input index of each entry, for each element of the weights.
    indices_output
        The output index of each entry, for each element of the weights.
    offsets_rows
        Where the rows of each element start in the matrix.
    lookup_rows
        Where each output index lands in the rows, after the offset.
    starts
        Where each element starts among all the entries, with the total last.
    num_rows
        The number of rows of the matrix.
    num_chunks
        The number of runs to count separately.
    """
    total = starts[~0]
    counts = np.zeros((num_chunks, num_rows), dtype=np.int64)
    for c in numba.prange(num_chunks):
        start = c * total // num_chunks
        stop = (c + 1) * total // num_chunks
        b = np.searchsorted(starts, start, side="right") - 1
        g = start
        while g < stop:
            g_stop = min(starts[b + 1], stop)
            indices_input_b = indices_input[b]
            indices_output_b = indices_output[b]
            offset = offsets_rows[b]
            for w in range(g - starts[b], g_stop - starts[b]):
                o = indices_output_b[w]
                if indices_input_b[w] < 0 or o < 0:
                    continue
                counts[c, offset + lookup_rows[o]] += 1
            g = g_stop
            b += 1
    return counts


@numba.njit(cache=True, parallel=True)
def _assemble_scatter(
    indices_input: TypedList,
    indices_output: TypedList,
    values: TypedList,
    offsets_rows: np.ndarray,
    lookup_rows: np.ndarray,
    offsets_columns: np.ndarray,
    lookup_columns: np.ndarray,
    starts: np.ndarray,
    cursors: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    """
    Place the entries of the weights in the rows of the matrix, using the
    cursors :func:`_cursors` made from the counts of :func:`_assemble_count`,
    so that each row holds its entries in the order of the elements of the
    weights, however many chunks there are.
    """
    num_chunks = cursors.shape[0]
    total = starts[~0]
    for c in numba.prange(num_chunks):
        start = c * total // num_chunks
        stop = (c + 1) * total // num_chunks
        b = np.searchsorted(starts, start, side="right") - 1
        g = start
        while g < stop:
            g_stop = min(starts[b + 1], stop)
            indices_input_b = indices_input[b]
            indices_output_b = indices_output[b]
            values_b = values[b]
            offset_rows = offsets_rows[b]
            offset_columns = offsets_columns[b]
            for w in range(g - starts[b], g_stop - starts[b]):
                i = indices_input_b[w]
                o = indices_output_b[w]
                if i < 0 or o < 0:
                    continue
                r = offset_rows + lookup_rows[o]
                p = cursors[c, r]
                cursors[c, r] = p + 1
                indices[p] = offset_columns + lookup_columns[i]
                data[p] = values_b[w]
            g = g_stop
            b += 1


@numba.njit(cache=True, parallel=True)
def _cursors(cursors: np.ndarray) -> np.ndarray:
    """
    Turn the counts of each chunk in each row into where each chunk starts
    writing that row, in place, and return where each row starts.

    The chunks of a row are laid out in order, so that the entries of a row
    keep the order they had before they were sorted.
    """
    num_chunks, num_rows = cursors.shape
    counts = np.zeros(num_rows + 1, dtype=np.int64)
    for r in numba.prange(num_rows):
        total = 0
        for c in range(num_chunks):
            n = cursors[c, r]
            cursors[c, r] = total
            total += n
        counts[r + 1] = total
    indptr = np.cumsum(counts)
    for r in numba.prange(num_rows):
        for c in range(num_chunks):
            cursors[c, r] += indptr[r]
    return indptr


@numba.njit(cache=True, parallel=True)
def _transpose_count(
    indptr: np.ndarray,
    indices: np.ndarray,
    num_columns: int,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Count the entries of a matrix in CSR form in each of its columns,
    separately for each chunk of rows between consecutive `bounds`.
    """
    num_chunks = bounds.shape[0] - 1
    counts = np.zeros((num_chunks, num_columns), dtype=np.int64)
    for c in numba.prange(num_chunks):
        for k in range(indptr[bounds[c]], indptr[bounds[c + 1]]):
            counts[c, indices[k]] += 1
    return counts


@numba.njit(cache=True, parallel=True)
def _transpose_scatter(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    bounds: np.ndarray,
    cursors: np.ndarray,
    indices_transposed: np.ndarray,
    data_transposed: np.ndarray,
) -> None:
    """
    Transpose a matrix in CSR form, using the cursors :func:`_cursors` made
    from the counts of :func:`_transpose_count`, so that each row of the
    transpose holds its entries in the order of the rows they came from.
    """
    num_chunks = bounds.shape[0] - 1
    for c in numba.prange(num_chunks):
        for r in range(bounds[c], bounds[c + 1]):
            for k in range(indptr[r], indptr[r + 1]):
                j = indices[k]
                p = cursors[c, j]
                cursors[c, j] = p + 1
                indices_transposed[p] = r
                data_transposed[p] = data[k]


_rows_per_block = 64
"""
How many consecutive rows of the matrix one thread computes at a time.

Enough that threads do not share the cache lines they write to, while the
blocks are still dealt out to the threads in turn, so that rows with many
entries are spread over all of them.
"""


@numba.njit(cache=True, parallel=True)
def _matmul(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    num_threads: int,
) -> None:
    """
    Multiply a matrix in CSR form by a dense matrix, adding the product to
    `y`.

    Each row is summed in the order its entries are stored, by one thread,
    so the result does not depend on how many threads there are.  The number
    of threads is passed in rather than asked for here, which would keep
    :mod:`numba` from caching the compiled function.
    """
    num_rows, num_batch = y.shape
    num_blocks = (num_rows + _rows_per_block - 1) // _rows_per_block
    num_chunks = min(num_threads, num_blocks)
    for c in numba.prange(num_chunks):
        for b in range(c, num_blocks, num_chunks):
            stop = min((b + 1) * _rows_per_block, num_rows)
            for r in range(b * _rows_per_block, stop):
                for k in range(indptr[r], indptr[r + 1]):
                    j = indices[k]
                    v = data[k]
                    for t in range(num_batch):
                        y[r, t] += v * x[j, t]


@numba.njit(cache=True, parallel=True)
def _scale(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    table_rows: np.ndarray,
    table_columns: np.ndarray,
    factor: np.ndarray,
    volume: np.ndarray,
    result: np.ndarray,
) -> None:
    """
    Multiply each entry of a transposed matrix by the factor of its input
    cell and divide it by the volume of its output cell.

    Each row of a table describes one axis of the rows or the columns: its
    stride and length there, and its stride in `factor` and in `volume`.
    """
    num_rows = indptr.shape[0] - 1
    for r in numba.prange(num_rows):
        offset_factor = 0
        offset_volume = 0
        for a in range(table_rows.shape[0]):
            i = (r // table_rows[a, 0]) % table_rows[a, 1]
            offset_factor += i * table_rows[a, 2]
            offset_volume += i * table_rows[a, 3]
        for k in range(indptr[r], indptr[r + 1]):
            c = indices[k]
            f = offset_factor
            v = offset_volume
            for a in range(table_columns.shape[0]):
                i = (c // table_columns[a, 0]) % table_columns[a, 1]
                f += i * table_columns[a, 2]
                v += i * table_columns[a, 3]
            result[k] = data[k] * factor[f] / volume[v]

from typing import Any, Sequence
import dataclasses
import numpy as np
import numba
from numba.typed.typedlist import List as TypedList
from regridding import _util
from regridding._weights._weights_transposed._weights_transposed import (
    _check_grids,
    _factors,
)
from ._regridder import (
    Regridder,
    _axes,
    _check_orthogonal,
    _flat,
    _index_dtype,
    _lists,
    _lookup,
    _offset,
    _orthogonal,
)

__all__ = [
    "RegridderBuilder",
]

_Weights = tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]
"""Weights, or a piece of them, in the format :func:`regridding.weights` returns."""

_unknown: Any = object()
"""Stands for the unit of the weights until a piece has said what it is."""


class RegridderBuilder:
    r"""
    Assemble a :class:`~regridding.Regridder`, and its conservative transpose
    if asked for, from weights too large to hold at once, given a piece at a
    time.

    :meth:`Regridder.from_weights` takes all the weights at once, and they
    can take as much memory again as the matrix it assembles.  A builder
    takes them a piece at a time instead, twice.  In the first pass,
    :meth:`count` counts the entries each piece puts in each row of the
    matrix, which fixes the size of the matrix.  In the second,
    :meth:`place` writes the entries of each piece where they belong.
    Nothing of a piece is kept once it has been counted or placed, so a
    caller who computes each piece again for the second pass never holds
    more than one.

    A piece is a rectangular block of the elements of the weights along
    their orthogonal axes, such as every channel of one spectral line, in the
    format :func:`regridding.weights` returns for those elements, and it
    says where it lies with the index of its first element.  An element no
    piece covers has no entries.  The second pass has to give the same
    pieces in the same order, with the same entries, which is checked: the
    entries of each piece as it is placed, and those of each row by
    :meth:`build`.

    Exact zeros are dropped along with the empty slots: they add nothing to
    a product, and they can be much of the weights, such as those of the
    cells a field stop blocks.  The matrix is then the one
    :meth:`Regridder.from_weights` assembles from the same weights with
    their zeros dropped, bit for bit, as long as each row gets its entries in
    the same order.  A row keeps its entries in the order they were placed,
    which is the order of the elements as long as the pieces which put
    entries in it are placed in that order.  Pieces share rows only where
    the operator sums the outputs of their elements, so pieces which hold
    whole sums can be placed in any order.

    With ``conservative=True``, the second pass also takes the grids of each
    piece, and :meth:`build` also returns the conservative transpose of the
    operator, the one :meth:`Regridder.transpose_conservative` computes from
    the grids of all the weights, bit for bit.  Each entry is scaled while
    the grids of its piece are at hand, so the grids are never needed all at
    once either.  The scaled values are kept in the order of the entries of
    the operator until :meth:`build` transposes them, which takes memory for
    a third set of values until then.

    .. code-block:: python

        builder = regridding.RegridderBuilder(
            shape_input=(4, 7, 24, 1139, 1139),
            shape_output=(4, 7, 1, 2048, 1040),
            axis_input=(-2, -1),
            axis_output=(-2, -1),
            shape_weights=(4, 7, 24),
            conservative=True,
            dtype=np.float32,
        )

        # every channel and wavelength of one spectral line at a time
        for line in range(7):
            builder.count(weights_of(line), start=(0, line, 0))
        for line in range(7):
            grid_input, grid_output = grids_of(line)
            builder.place(
                weights_of(line),
                start=(0, line, 0),
                coordinates_input=grid_input,
                coordinates_output=grid_output,
            )

        regridder, transposed = builder.build()

    :func:`regridding.weights` perturbs a curvilinear output grid with random
    numbers drawn for the whole grid it is given, so weights computed a
    piece at a time are the same as those computed at once only where the
    output grid is a uniform lattice, which it does not perturb.

    Parameters
    ----------
    shape_input
        The shape of the arrays the operator accepts, as
        :meth:`Regridder.from_weights` takes it.
    shape_output
        The shape of the arrays the operator returns, as
        :meth:`Regridder.from_weights` takes it.
    axis_input
        The resampled axes of `shape_input`.
    axis_output
        The resampled axes of `shape_output`.
    shape_weights
        The length of the weights along each orthogonal axis, which is the
        shape :func:`regridding.weights` would give them all at once.
    conservative
        Whether to build the conservative transpose of the operator too,
        from the grids :meth:`place` is given with each piece.
    dtype
        The type of the entries of the matrices, as
        :meth:`Regridder.from_weights` takes it.

    Raises
    ------
    ValueError
        If the shapes do not fit each other, as for
        :meth:`Regridder.from_weights`.
    NotImplementedError
        With ``conservative=True``, if the operator sums an orthogonal axis
        it shares the input along.
    """

    def __init__(
        self,
        shape_input: tuple[int, ...],
        shape_output: tuple[int, ...],
        axis_input: None | int | Sequence[int],
        axis_output: None | int | Sequence[int],
        shape_weights: tuple[int, ...],
        conservative: bool = False,
        dtype: type = np.float64,
    ) -> None:
        shape_input = tuple(int(n) for n in shape_input)
        shape_output = tuple(int(n) for n in shape_output)
        shape_weights = tuple(int(n) for n in shape_weights)
        axis_input = _axes(axis_input, len(shape_input))
        axis_output = _axes(axis_output, len(shape_output))

        num_orthogonal = len(shape_weights)
        for shape, axis, name in (
            (shape_input, axis_input, "input"),
            (shape_output, axis_output, "output"),
        ):
            if len(shape) != num_orthogonal + len(axis):
                raise ValueError(
                    f"the {name} should have {num_orthogonal + len(axis)} axes, "
                    f"one for each orthogonal axis of the weights and each "
                    f"resampled axis, but has a shape of {shape}"
                )

        orthogonal_input = _orthogonal(len(shape_input), axis_input)
        orthogonal_output = _orthogonal(len(shape_output), axis_output)
        _check_orthogonal(
            shape_input=shape_input,
            shape_output=shape_output,
            orthogonal_input=orthogonal_input,
            orthogonal_output=orthogonal_output,
            shape_weights=shape_weights,
        )

        if conservative:
            for k, num in enumerate(shape_weights):
                shared = shape_input[orthogonal_input[k]] == 1
                summed = shape_output[orthogonal_output[k]] == 1
                if num > 1 and shared and summed:
                    raise NotImplementedError(
                        f"orthogonal axis {k} is summed after the input is "
                        f"shared along it, which a conservative build does not "
                        f"support"
                    )

        self.shape_input = shape_input
        """The shape of the arrays the operator accepts."""

        self.shape_output = shape_output
        """The shape of the arrays the operator returns."""

        self.axis_input = axis_input
        """The resampled axes of the input, ascending and non-negative."""

        self.axis_output = axis_output
        """The resampled axes of the output, ascending and non-negative."""

        self.shape_weights = shape_weights
        """The length of the weights along each orthogonal axis, in order."""

        self.conservative = bool(conservative)
        """Whether the conservative transpose is built too."""

        self.dtype = dtype
        """The type of the entries of the matrices."""

        self._regridder = Regridder(
            shape_input=shape_input,
            shape_output=shape_output,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_weights=shape_weights,
            indptr=np.empty(0, dtype=np.int64),
            indices=np.empty(0, dtype=np.int64),
            data=np.empty(0),
        )
        """The operator being built, before it has any entries."""

        self._lookup_rows = _lookup(
            shape_output, axis_output, self._regridder._strides_output
        )
        """Where each output index lands in the rows, after the offset."""

        self._lookup_columns = _lookup(
            shape_input, axis_input, self._regridder._strides_input
        )
        """Where each input index lands in the columns, after the offset."""

        num_rows, _ = self._regridder.shape_matrix
        self._counts: None | np.ndarray = np.zeros(num_rows + 1, dtype=np.int64)
        """
        The number of entries counted in each row, after a leading zero, until
        the first pass ends and they are summed in place into where each row
        starts.
        """

        self._pieces: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        """The start, the shape and the number of entries of each piece counted."""

        self._covered = np.zeros(shape_weights, dtype=bool)
        """Which elements of the weights the pieces counted so far cover."""

        self._num_placed = 0
        """The number of pieces placed so far."""

        self._matrix: None | _Matrix = None
        """The arrays of the matrix, once the second pass has begun."""

        self._unit: Any = _unknown
        """The unit of the values of the weights, once a piece has said."""

        self._failure: None | str = None
        """Why the builder cannot go on, if it cannot."""

        self._built = False
        """Whether :meth:`build` has returned the operators."""

    def count(
        self,
        weights: _Weights,
        start: Sequence[int],
    ) -> None:
        """
        Count the entries of a piece of the weights, in the first pass.

        Parameters
        ----------
        weights
            A piece of the weights, as :func:`regridding.weights` computes
            them for the elements it holds.
        start
            The index of the first element of the piece along each orthogonal
            axis of the weights.

        Raises
        ------
        ValueError
            If a piece has been placed already, which ends the first pass, or
            if the piece does not fit the weights, covers an element a piece
            counted before it does, or has values in another unit.
        """
        self._check_usable()
        counts = self._counts
        if counts is None:
            raise ValueError(
                "a piece has been placed, which ends the first pass, so no more "
                "can be counted"
            )

        piece = self._piece(weights, start)

        covered = self._covered[piece.index]
        if covered.any():
            raise ValueError(
                f"the piece starting at {piece.start}, with a shape of "
                f"{piece.shape}, covers elements of the weights which a piece "
                f"counted before it does"
            )

        num_entries = 0
        if piece.lists is not None:
            num_entries = _count(
                *piece.lists,
                piece.order,
                piece.groups,
                piece.offsets_rows,
                self._lookup_rows,
                counts,
            )

        covered[...] = True
        self._pieces.append((piece.start, piece.shape, int(num_entries)))
        self._settle_unit(piece.unit)

    def place(
        self,
        weights: _Weights,
        start: Sequence[int],
        coordinates_input: None | np.ndarray | tuple[np.ndarray, ...] = None,
        coordinates_output: None | np.ndarray | tuple[np.ndarray, ...] = None,
        weights_input: None | np.ndarray = None,
    ) -> None:
        """
        Place the entries of a piece of the weights in the matrix, in the
        second pass.

        The first piece placed ends the first pass.

        Parameters
        ----------
        weights
            The piece of the weights which was counted in the same place of
            the first pass.
        start
            The index of the first element of the piece along each orthogonal
            axis of the weights.
        coordinates_input
            With ``conservative=True``, the vertices of the input grid of the
            elements of this piece, as given to :func:`regridding.weights`.
        coordinates_output
            With ``conservative=True``, the vertices of the output grid of the
            elements of this piece, as given to :func:`regridding.weights`.
        weights_input
            With ``conservative=True``, the weights applied to the input values
            of the elements of this piece, if any, as
            :meth:`Regridder.transpose_conservative` takes them.

        Raises
        ------
        ValueError
            If the piece is not the one counted in the same place of the first
            pass, or the grids are missing from a conservative build or given
            to another.  Also if the piece has entries other than those
            counted for it, after which the builder cannot go on.
        """
        self._check_usable()

        k = self._num_placed
        if k >= len(self._pieces):
            raise ValueError(
                f"the {len(self._pieces)} pieces counted in the first pass have "
                f"all been placed"
            )

        piece = self._piece(weights, start)

        start_counted, shape_counted, num_counted = self._pieces[k]
        if (piece.start, piece.shape) != (start_counted, shape_counted):
            raise ValueError(
                f"piece {k} of the first pass starts at {start_counted} with a "
                f"shape of {shape_counted}, but this one starts at {piece.start} "
                f"with a shape of {piece.shape}; the second pass has to give the "
                f"same pieces in the same order"
            )

        grids = (coordinates_input, coordinates_output, weights_input)
        if not self.conservative and any(a is not None for a in grids):
            raise ValueError(
                "the grids are needed only for the conservative transpose, which "
                "this builder was not asked for"
            )

        scaling = _Scaling.none()
        if self.conservative and piece.lists is not None:
            if coordinates_input is None or coordinates_output is None:
                raise ValueError(
                    "a conservative build needs the grids of each piece, as "
                    "`coordinates_input` and `coordinates_output`"
                )
            scaling = self._scaling(
                piece=piece,
                coordinates_input=coordinates_input,
                coordinates_output=coordinates_output,
                weights_input=weights_input,
            )

        matrix = self._matrix
        if matrix is None:
            matrix = self._start_placing()

        num_entries = 0
        overflow = False
        if piece.lists is not None:
            num_entries, overflow = _place(
                *piece.lists,
                piece.order,
                piece.groups,
                piece.offsets_rows,
                self._lookup_rows,
                piece.offsets_columns,
                self._lookup_columns,
                matrix.cursors,
                matrix.indices,
                matrix.data,
                self.conservative,
                matrix.data_conservative,
                scaling.scale,
                scaling.factor,
                scaling.offsets_factor,
                scaling.lookup_factor,
                scaling.volume,
                scaling.offsets_volume,
                scaling.lookup_volume,
            )

        self._num_placed += 1

        if overflow or num_entries != num_counted:
            self._failure = (
                f"piece {k}, which starts at {piece.start}, does not have the "
                f"{num_counted} entries counted for it in the first pass; both "
                f"passes have to give the same weights"
            )
            raise ValueError(self._failure)

    def build(self) -> tuple[Regridder, None | Regridder]:
        """
        Finish the operator, and its conservative transpose if asked for, once
        every piece counted has been placed.

        The builder lets go of the matrices, which belong to the operators
        from then on.

        Returns
        -------
        The operator, and its conservative transpose, or :obj:`None` for the
        transpose without ``conservative=True``.

        Raises
        ------
        ValueError
            If a piece counted in the first pass has not been placed, or a
            row was not given the entries counted for it.
        """
        self._check_usable()

        num_counted = len(self._pieces)
        if self._num_placed != num_counted:
            raise ValueError(
                f"{self._num_placed} of the {num_counted} pieces counted in the "
                f"first pass have been placed"
            )

        matrix = self._matrix
        if matrix is None:
            matrix = self._start_placing()

        wrong = np.flatnonzero(matrix.cursors != matrix.indptr[1:])
        if wrong.size:
            self._failure = (
                f"{wrong.size} rows of the matrix, the first of them row "
                f"{wrong[0]}, were not given the entries counted for them in "
                f"the first pass; both passes have to give the same weights"
            )
            raise ValueError(self._failure)

        self._matrix = None
        self._built = True

        unit = None if self._unit is _unknown else self._unit
        regridder = dataclasses.replace(
            self._regridder,
            indptr=matrix.indptr,
            indices=matrix.indices,
            data=matrix.data,
            unit=unit,
        )

        transposed = None
        if self.conservative:
            values = dataclasses.replace(regridder, data=matrix.data_conservative)
            # the copy starts without the link to the operator it was
            # transposed from, whose values it does not share, as
            # `Regridder.transpose_conservative` returns it
            transposed = dataclasses.replace(values._transpose(), unit=None)

        return regridder, transposed

    def _check_usable(self) -> None:
        """Raise if the builder has failed, or has built its operators."""
        if self._failure is not None:
            raise ValueError(f"this builder cannot go on: {self._failure}")
        if self._built:
            raise ValueError("this builder has built its operators already")

    def _settle_unit(self, unit: Any) -> None:
        """Take `unit` for that of the weights, if no piece has said yet."""
        if self._unit is _unknown:
            self._unit = unit

    def _start_placing(self) -> "_Matrix":
        """
        End the first pass: sum the counts of the rows, in place, into where
        each row starts, and make room for the entries.
        """
        counts = self._counts
        if counts is None:  # pragma: nocover
            raise RuntimeError("the second pass has begun already")
        np.cumsum(counts, out=counts)
        _, num_columns = self._regridder.shape_matrix
        num_entries = int(counts[~0])
        num_conservative = num_entries if self.conservative else 0
        matrix = _Matrix(
            indptr=counts,
            cursors=counts[:~0].copy(),
            indices=np.empty(num_entries, dtype=_index_dtype(num_columns)),
            data=np.empty(num_entries, dtype=self.dtype),
            data_conservative=np.empty(num_conservative, dtype=self.dtype),
        )
        self._counts = None
        self._matrix = matrix
        return matrix

    def _piece(
        self,
        weights: _Weights,
        start: Sequence[int],
    ) -> "_Piece":
        """
        Check a piece of the weights against the operator being built, and
        lay it out for the kernels.

        Parameters
        ----------
        weights
            The piece of the weights.
        start
            The index of its first element along each orthogonal axis.
        """
        array, shape_grid_input, shape_grid_output = weights
        array = np.asarray(array)
        shape = array.shape
        start = tuple(int(s) for s in start)

        num_orthogonal = len(self.shape_weights)
        if array.ndim != num_orthogonal or len(start) != num_orthogonal:
            raise ValueError(
                f"a piece of the weights should have {num_orthogonal} orthogonal "
                f"axes, as the weights do, and start somewhere along each, but "
                f"this one has a shape of {shape} and starts at {start}"
            )
        for s, n, num in zip(start, shape, self.shape_weights):
            if s < 0 or s + n > num:
                raise ValueError(
                    f"the piece starting at {start}, with a shape of {shape}, "
                    f"does not fit in the weights, which have a shape of "
                    f"{self.shape_weights}"
                )

        for shape_grid, shape_operator, axis, name in (
            (shape_grid_input, self.shape_input, self.axis_input, "input"),
            (shape_grid_output, self.shape_output, self.axis_output, "output"),
        ):
            # the resampled axes of the grids are counted from the end, as
            # `Regridder.from_weights()` counts them
            ndim = len(shape_operator)
            for ax in axis:
                num_grid = None
                if ndim - ax <= len(shape_grid):
                    num_grid = shape_grid[ax - ndim]
                if num_grid != shape_operator[ax]:
                    raise ValueError(
                        f"the {name} has {shape_operator[ax]} elements along the "
                        f"resampled axis {ax}, but the grid this piece was built "
                        f"for has {num_grid}"
                    )

        regridder = self._regridder
        strides_rows = regridder._strides_output
        strides_columns = regridder._strides_input
        orthogonal_input = regridder._orthogonal_input
        orthogonal_output = regridder._orthogonal_output

        blocks = []
        offsets_rows = []
        offsets_columns = []
        unit = _unknown
        for local in np.ndindex(*shape):
            indices_input, indices_output, values = array[local]
            unit_values = getattr(values, "unit", None)
            if unit_values is not None:
                values = values.value
            if unit is _unknown:
                unit = unit_values
            elif unit_values != unit:
                raise ValueError(
                    f"the values of the elements of a piece should have one "
                    f"unit, but this one has values in {unit} and in "
                    f"{unit_values}"
                )
            blocks.append((indices_input, indices_output, values))

            index = tuple(s + i for s, i in zip(start, local))
            offsets_rows.append(
                _offset(
                    index,
                    orthogonal_output,
                    self.shape_output,
                    strides_rows,
                    self.shape_weights,
                )
            )
            offsets_columns.append(
                _offset(
                    index,
                    orthogonal_input,
                    self.shape_input,
                    strides_columns,
                    self.shape_weights,
                )
            )

        known = unit is not _unknown and self._unit is not _unknown
        if known and unit != self._unit:
            raise ValueError(
                f"the values of this piece are in {unit}, but those of the "
                f"pieces before it are in {self._unit}"
            )

        offsets_rows = np.array(offsets_rows, dtype=np.int64)
        offsets_columns = np.array(offsets_columns, dtype=np.int64)

        # elements with the same offset put their entries in the same rows,
        # and elements with different offsets in rows of their own, so each
        # group of them is placed by one thread, in the order of the elements
        order = np.argsort(offsets_rows, kind="stable").astype(np.int64)
        changes = np.flatnonzero(np.diff(offsets_rows[order])) + 1
        groups = np.concatenate([[0], changes, [order.size]]).astype(np.int64)

        return _Piece(
            start=start,
            shape=shape,
            unit=unit,
            lists=_lists(blocks) if blocks else None,
            order=order,
            groups=groups,
            offsets_rows=offsets_rows,
            offsets_columns=offsets_columns,
        )

    def _scaling(
        self,
        piece: "_Piece",
        coordinates_input: np.ndarray | tuple[np.ndarray, ...],
        coordinates_output: np.ndarray | tuple[np.ndarray, ...],
        weights_input: None | np.ndarray,
    ) -> "_Scaling":
        """
        What each entry of a piece is multiplied and divided by in the
        conservative transpose, as :meth:`Regridder.transpose_conservative`
        computes it from the cells of the element the entry came from.

        Parameters
        ----------
        piece
            The piece of the weights.
        coordinates_input
            The vertices of the input grid of its elements.
        coordinates_output
            The vertices of the output grid of its elements.
        weights_input
            The weights applied to the input values of its elements, if any.
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

        orthogonal_input = self._regridder._orthogonal_input
        orthogonal_output = self._regridder._orthogonal_output
        shape_cells_input = _cells(self.shape_input, orthogonal_input, piece.shape)
        shape_cells_output = _cells(self.shape_output, orthogonal_output, piece.shape)

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

        lookup_factor = _lookup(
            shape_cells_input,
            self.axis_input,
            {ax: strides_factor[ax] for ax in self.axis_input},
        )
        lookup_volume = _lookup(
            shape_cells_output,
            self.axis_output,
            {ax: strides_volume[ax] for ax in self.axis_output},
        )

        # the cells of each element of the piece, which knows its element, so
        # it need not find it from the row and the column of an entry
        offsets_factor = []
        offsets_volume = []
        for local in np.ndindex(*piece.shape):
            offsets_factor.append(
                sum(i * strides_factor[ax] for i, ax in zip(local, orthogonal_input))
            )
            offsets_volume.append(
                sum(i * strides_volume[ax] for i, ax in zip(local, orthogonal_output))
            )

        # a dimensionless unit, like a percentage, scales the weights by the
        # number it stands for, and any other is dropped, as
        # `Regridder.transpose_conservative` does
        scale = 1.0
        if piece.unit is not _unknown and piece.unit is not None:
            scale = float(_util._dimensionless(1.0 << piece.unit, strict=False))

        return _Scaling(
            scale=scale,
            factor=factor,
            offsets_factor=np.array(offsets_factor, dtype=np.int64),
            lookup_factor=lookup_factor,
            volume=volume,
            offsets_volume=np.array(offsets_volume, dtype=np.int64),
            lookup_volume=lookup_volume,
        )


@dataclasses.dataclass(frozen=True)
class _Matrix:
    """The arrays of the matrix being built, in the second pass."""

    indptr: np.ndarray
    """Where each row starts in :attr:`indices` and :attr:`data`."""

    cursors: np.ndarray
    """Where the next entry of each row goes."""

    indices: np.ndarray
    """The column of each entry."""

    data: np.ndarray
    """The value of each entry."""

    data_conservative: np.ndarray
    """
    The value of each entry in the conservative transpose, in the order of the
    entries of the operator, or nothing without ``conservative=True``.
    """


@dataclasses.dataclass(frozen=True)
class _Piece:
    """A piece of the weights, laid out for the kernels of the builder."""

    start: tuple[int, ...]
    """The index of its first element along each orthogonal axis."""

    shape: tuple[int, ...]
    """Its number of elements along each orthogonal axis."""

    unit: Any
    """The unit of its values, :obj:`None` if they have none, or unknown if it
    has no elements."""

    lists: None | tuple[TypedList, TypedList, TypedList]
    """
    The input indices, output indices and values of each of its elements, in
    order, or :obj:`None` if it has none.
    """

    order: np.ndarray
    """The elements, in the order of their groups."""

    groups: np.ndarray
    """Where each group of elements which share rows starts in :attr:`order`,
    with the number of elements last."""

    offsets_rows: np.ndarray
    """Where the rows of each element start in the matrix."""

    offsets_columns: np.ndarray
    """Where the columns of each element start in the matrix."""

    @property
    def index(self) -> tuple[Any, ...]:
        """
        Where the piece lies in the elements of the weights, as an index which
        gives a view, even of weights with no orthogonal axes.
        """
        return tuple(slice(s, s + n) for s, n in zip(self.start, self.shape)) + (
            Ellipsis,
        )


@dataclasses.dataclass(frozen=True)
class _Scaling:
    """
    What the entries of a piece are scaled by in the conservative transpose,
    see :meth:`RegridderBuilder._scaling`.
    """

    scale: float
    """What a dimensionless unit of the weights stands for."""

    factor: np.ndarray
    """
    The volume of each input cell over the square of its weight, laid out flat
    as the cells of the piece are.
    """

    offsets_factor: np.ndarray
    """Where the cells of each element start in :attr:`factor`."""

    lookup_factor: np.ndarray
    """Where each input index lands in :attr:`factor`, after the offset."""

    volume: np.ndarray
    """The volume of each output cell, laid out flat as the cells of the piece are."""

    offsets_volume: np.ndarray
    """Where the cells of each element start in :attr:`volume`."""

    lookup_volume: np.ndarray
    """Where each output index lands in :attr:`volume`, after the offset."""

    @classmethod
    def none(cls) -> "_Scaling":
        """No scaling, for a build without the conservative transpose."""
        empty = np.empty(0, dtype=np.int64)
        return cls(
            scale=1.0,
            factor=np.empty(0),
            offsets_factor=empty,
            lookup_factor=empty,
            volume=np.empty(0),
            offsets_volume=empty,
            lookup_volume=empty,
        )


def _cells(
    shape: tuple[int, ...],
    orthogonal: tuple[int, ...],
    shape_piece: tuple[int, ...],
) -> tuple[int, ...]:
    """
    The shape of the cells of a grid of a piece of the weights: `shape` along
    the resampled axes, and the length of the piece along the orthogonal ones.
    """
    result = list(shape)
    for ax, n in zip(orthogonal, shape_piece):
        result[ax] = n
    return tuple(result)


@numba.njit(cache=True, parallel=True)
def _count(
    indices_input: TypedList,
    indices_output: TypedList,
    values: TypedList,
    order: np.ndarray,
    groups: np.ndarray,
    offsets_rows: np.ndarray,
    lookup_rows: np.ndarray,
    counts: np.ndarray,
) -> int:
    """
    Count the entries of a piece of the weights in each row of the matrix,
    skipping empty slots and exact zeros, and return how many there are.

    Each group of elements puts its entries in rows of its own, so the
    groups are counted in parallel.

    Parameters
    ----------
    indices_input
        The input index of each entry, for each element of the piece.
    indices_output
        The output index of each entry, for each element of the piece.
    values
        The value of each entry, for each element of the piece.
    order
        The elements, in the order of their groups.
    groups
        Where each group starts in `order`, with the number of elements last.
    offsets_rows
        Where the rows of each element start in the matrix.
    lookup_rows
        Where each output index lands in the rows, after the offset.
    counts
        The number of entries in each row so far, after a leading zero, which
        the entries of the piece are added to.
    """
    num_groups = groups.shape[0] - 1
    totals = np.zeros(num_groups, dtype=np.int64)
    for g in numba.prange(num_groups):
        total = 0
        for q in range(groups[g], groups[g + 1]):
            b = order[q]
            indices_input_b = indices_input[b]
            indices_output_b = indices_output[b]
            values_b = values[b]
            offset = offsets_rows[b]
            for w in range(values_b.shape[0]):
                o = indices_output_b[w]
                if indices_input_b[w] < 0 or o < 0 or values_b[w] == 0:
                    continue
                counts[offset + lookup_rows[o] + 1] += 1
                total += 1
        totals[g] = total
    return int(totals.sum())


@numba.njit(cache=True, parallel=True)
def _place(
    indices_input: TypedList,
    indices_output: TypedList,
    values: TypedList,
    order: np.ndarray,
    groups: np.ndarray,
    offsets_rows: np.ndarray,
    lookup_rows: np.ndarray,
    offsets_columns: np.ndarray,
    lookup_columns: np.ndarray,
    cursors: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    conservative: bool,
    data_conservative: np.ndarray,
    scale: float,
    factor: np.ndarray,
    offsets_factor: np.ndarray,
    lookup_factor: np.ndarray,
    volume: np.ndarray,
    offsets_volume: np.ndarray,
    lookup_volume: np.ndarray,
) -> tuple[int, bool]:
    """
    Place the entries of a piece of the weights in the rows of the matrix,
    each at the cursor of its row, skipping empty slots and exact zeros, and
    return how many there are and whether any found its row full.

    The value of each entry in the conservative transpose is its value in
    the operator, multiplied by `scale` and the factor of its input cell and
    divided by the volume of its output cell, in the order
    :func:`regridding._regridder._regridder._scale` multiplies them, so that
    it gives exactly the same numbers.

    Parameters
    ----------
    indices_input
        The input index of each entry, for each element of the piece.
    indices_output
        The output index of each entry, for each element of the piece.
    values
        The value of each entry, for each element of the piece.
    order
        The elements, in the order of their groups.
    groups
        Where each group starts in `order`, with the number of elements last.
    offsets_rows
        Where the rows of each element start in the matrix.
    lookup_rows
        Where each output index lands in the rows, after the offset.
    offsets_columns
        Where the columns of each element start in the matrix.
    lookup_columns
        Where each input index lands in the columns, after the offset.
    cursors
        Where the next entry of each row goes, which is moved on past it.
    indices
        The column of each entry of the matrix.
    data
        The value of each entry of the matrix.
    conservative
        Whether to compute the values of the conservative transpose too.
    data_conservative
        The value of each entry of the matrix in the conservative transpose.
    scale
        What a dimensionless unit of the weights stands for.
    factor
        The volume of each input cell of the piece over the square of its
        weight.
    offsets_factor
        Where the cells of each element start in `factor`.
    lookup_factor
        Where each input index lands in `factor`, after the offset.
    volume
        The volume of each output cell of the piece.
    offsets_volume
        Where the cells of each element start in `volume`.
    lookup_volume
        Where each output index lands in `volume`, after the offset.
    """
    num_entries = data.shape[0]
    num_groups = groups.shape[0] - 1
    totals = np.zeros(num_groups, dtype=np.int64)
    overflows = np.zeros(num_groups, dtype=np.bool_)
    for g in numba.prange(num_groups):
        total = 0
        for q in range(groups[g], groups[g + 1]):
            b = order[q]
            indices_input_b = indices_input[b]
            indices_output_b = indices_output[b]
            values_b = values[b]
            offset_rows = offsets_rows[b]
            offset_columns = offsets_columns[b]
            offset_factor = 0
            offset_volume = 0
            if conservative:
                offset_factor = offsets_factor[b]
                offset_volume = offsets_volume[b]
            for w in range(values_b.shape[0]):
                i = indices_input_b[w]
                o = indices_output_b[w]
                v = values_b[w]
                if i < 0 or o < 0 or v == 0:
                    continue
                r = offset_rows + lookup_rows[o]
                p = cursors[r]
                if p >= num_entries:
                    # a row given more entries than were counted for it
                    # overruns the next, which `build()` finds, but not the
                    # end of the matrix
                    overflows[g] = True
                    continue
                cursors[r] = p + 1
                indices[p] = offset_columns + lookup_columns[i]
                data[p] = v
                if conservative:
                    f = offset_factor + lookup_factor[i]
                    u = offset_volume + lookup_volume[o]
                    data_conservative[p] = data[p] * scale * factor[f] / volume[u]
                total += 1
        totals[g] = total
    return int(totals.sum()), bool(overflows.any())

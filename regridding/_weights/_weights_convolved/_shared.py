"""
The convolution of a set of weights with a kernel, in a form both the host
and a CUDA device can compile.

:mod:`._host` runs this on the CPU through :func:`numba.njit`, and
:mod:`._cuda` runs the same source on a GPU through :func:`numba.cuda.jit`,
which is the arrangement the clipping kernel uses as well: a fix to the
convolution applies to both, and the device cannot drift away from the host
it is tested against.

The work is divided into *runs*: maximal stretches of consecutive weights
which share an input cell.  A run is convolved on its own, in a scratch box
which covers its footprint, the bounding box of the run's output cells grown
by the kernel.  Each weight of the run scatters itself through the kernel
into the box, and the box is then read out in order, so a run costs one
multiply-add for each weight and element of the kernel, plus one visit to
each cell of the box, however many output cells the input cell covers.
Reading the box in order means a run comes out ordered by output cell and
with no cell twice, so weights ordered by input cell, as
:func:`regridding.weights` returns them, are convolved into the same ordered,
merged form they arrived in.

The scratch box is not allocated here, since the host and the device
allocate it differently: the host keeps one per thread and grows it as
needed, and the device, which cannot allocate inside a kernel, measures every
run's box first and gives each its own slice of one allocation.

Weights that are not ordered by input cell are still convolved correctly,
since every weight belongs to some run.  They just split into more, shorter
runs, and a pair can then appear once for each run that reaches it, which
:func:`regridding.regrid_from_weights` sums.

Array arguments are annotated as :class:`numpy.ndarray`, which is what they
are on the host.  On a device they are
:class:`numba.cuda.cudadrv.devicearray.DeviceNDArray`, which is not a
subclass but is indexed the same way.
"""

from typing import Any, Callable
import numpy as np

__all__ = [
    "num_axis",
    "build",
]

num_axis = 8
"""
The most axes a kernel can act along.

Each run keeps a few small arrays with one element per axis, such as the
corner and size of its scratch box, and a device has to know the size of an
array like that when the kernel is compiled.  Weights resample one or two
axes in practice, so this is generous.
"""


def _valid(
    indices_input: np.ndarray,
    indices_output: np.ndarray,
    w: int,
    skip_negative: bool,
) -> bool:
    """
    Test whether a weight is a real one rather than an empty slot.

    Parameters
    ----------
    indices_input
        The flattened index of the input cell of each weight.
    indices_output
        The flattened index of the output cell of each weight.
    w
        The index of the weight.
    skip_negative
        Whether a negative index marks an empty slot, as it does in weights
        built on a device.  On the host a negative index counts from the end
        of the grid instead, as :mod:`numpy` counts it, and every weight is
        a real one.
    """
    if not skip_negative:
        return True
    return indices_input[w] >= 0 and indices_output[w] >= 0


def build(jit: Callable[[Callable], Any]) -> tuple[Callable, Callable, Callable]:
    """
    Compile the shared kernel bodies for one target.

    Parameters
    ----------
    jit
        A callable which compiles a plain Python function for the target,
        such as :func:`numba.njit` for the host or :func:`numba.cuda.jit`
        with ``device=True`` for a CUDA device.

    Returns
    -------
    The compiled ``(starts_run, box_run, convolve_run)``.
    """

    valid = jit(_valid)

    def starts_run(
        indices_input: np.ndarray,
        indices_output: np.ndarray,
        w: int,
        skip_negative: bool,
    ) -> bool:
        """
        Test whether a weight is the first of a run.

        A weight starts a run if it is a real one and the weight before it
        is not part of the same run, either because it is an empty slot or
        because its input cell is a different one.

        Parameters
        ----------
        indices_input
            The flattened index of the input cell of each weight.
        indices_output
            The flattened index of the output cell of each weight.
        w
            The index of the weight.
        skip_negative
            Whether a negative index marks an empty slot.
        """
        if not valid(indices_input, indices_output, w, skip_negative):
            return False
        if w == 0:
            return True
        if not valid(indices_input, indices_output, w - 1, skip_negative):
            return True
        return indices_input[w - 1] != indices_input[w]

    starts_run = jit(starts_run)

    def box_run(
        indices_input: np.ndarray,
        indices_output: np.ndarray,
        shape_output: np.ndarray,
        shape_kernel: np.ndarray,
        skip_negative: bool,
        start: int,
        lower: np.ndarray,
        extent: np.ndarray,
    ) -> tuple[int, int]:
        """
        Find where the run beginning at `start` ends, and the scratch box its
        footprint needs.

        The box is the bounding box of the run's output cells, grown by the
        kernel and clipped to the grid: light the kernel spreads beyond the
        grid is lost, as it would be off the edge of a sensor.  Every output
        cell lies inside the grid, so the box always holds at least the run's
        own cells.

        Parameters
        ----------
        indices_input
            The flattened index of the input cell of each weight.
        indices_output
            The flattened index of the output cell of each weight, which
            has to lie inside the grid.
        shape_output
            The number of output cells along each resampled axis.
        shape_kernel
            The length of the kernel along each resampled axis.
        skip_negative
            Whether a negative index marks an empty slot.
        start
            The index of the first weight of the run.
        lower
            An output array with one element per axis, for the lower corner
            of the box.
        extent
            An output array with one element per axis, for the size of the
            box.

        Returns
        -------
        The index one past the last weight of the run, and the number of
        cells in the box.
        """

        num = indices_input.shape[0]
        ndim = shape_output.shape[0]

        index_input = indices_input[start]

        stop = start + 1
        while stop < num:
            if not valid(indices_input, indices_output, stop, skip_negative):
                break
            if indices_input[stop] != index_input:
                break
            stop += 1

        # the bounding box of the output cells of the run, which `extent`
        # holds the upper corner of until it is grown by the kernel
        for a in range(ndim):
            lower[a] = shape_output[a]
            extent[a] = -1
        for q in range(start, stop):
            r = indices_output[q]
            for a in range(ndim - 1, -1, -1):
                c = r % shape_output[a]
                r = r // shape_output[a]
                if c < lower[a]:
                    lower[a] = c
                if c > extent[a]:
                    extent[a] = c

        size = 1
        for a in range(ndim):
            center = shape_kernel[a] // 2
            low = lower[a] - center
            high = extent[a] + shape_kernel[a] - 1 - center
            if low < 0:
                low = 0
            if high > shape_output[a] - 1:
                high = shape_output[a] - 1
            lower[a] = low
            extent[a] = high - low + 1
            size = size * extent[a]

        return stop, size

    box_run = jit(box_run)

    def convolve_run(
        indices_input: np.ndarray,
        indices_output: np.ndarray,
        values: np.ndarray,
        kernel: np.ndarray,
        shape_grid: np.ndarray,
        shape_output: np.ndarray,
        offsets: np.ndarray,
        start: int,
        stop: int,
        lower: np.ndarray,
        extent: np.ndarray,
        coordinates: np.ndarray,
        strides: np.ndarray,
        scratch_values: np.ndarray,
        scratch_reached: np.ndarray,
        index_scratch: int,
        write: bool,
        index_write: int,
        result_input: np.ndarray,
        result_output: np.ndarray,
        result_values: np.ndarray,
    ) -> int:
        """
        Convolve the run of weights from `start` to `stop` in its scratch
        box, and return how many weights it becomes.

        The light that a weight sends to output cell :math:`i` is spread
        over the cells around it by the kernel, which is centered on
        :math:`i`.  Each weight scatters into the box once for each element
        of the kernel, and the box is then read out in order.  A cell no
        weight reaches with a nonzero element of the kernel is skipped, so
        the count depends only on where the kernel is nonzero and is the
        same whether or not anything is written.

        Parameters
        ----------
        indices_input
            The flattened index of the input cell of each weight.
        indices_output
            The flattened index of the output cell of each weight, which
            has to lie inside the grid.
        values
            The value of each weight.
        kernel
            The kernel, with one row for each cell of `shape_grid` and one
            column for each element of the kernel.
        shape_grid
            The number of rows of the kernel along each resampled axis,
            either one, if the kernel does not vary along that axis, or the
            number of output cells along it.  The row a weight uses is the
            one for the cell its light lands in.
        shape_output
            The number of output cells along each resampled axis.
        offsets
            The offset of each element of the kernel from its center, with
            one row for each element and one column for each resampled axis.
        start
            The index of the first weight of the run.
        stop
            The index one past the last weight of the run, from `box_run`.
        lower
            The lower corner of the box, from `box_run`.
        extent
            The size of the box, from `box_run`.
        coordinates
            Scratch space with one element per axis, for the cell a weight
            lands in.
        strides
            Scratch space with one element per axis, for the strides of the
            box.
        scratch_values
            Scratch space for what each cell of the box receives.
        scratch_reached
            Scratch space for whether each cell of the box is reached.
        index_scratch
            Where in the scratch arrays the run's box begins.
        write
            Whether to write the result, or only to count it.
        index_write
            Where in the result arrays the run begins.
        result_input
            An output array for the flattened index of the input cell.
        result_output
            An output array for the flattened index of the output cell.
        result_values
            An output array for the convolved weights.
        """

        ndim = shape_output.shape[0]
        size_kernel = offsets.shape[0]

        size = 1
        for a in range(ndim - 1, -1, -1):
            strides[a] = size
            size = size * extent[a]

        for t in range(size):
            scratch_values[index_scratch + t] = 0
            scratch_reached[index_scratch + t] = False

        for q in range(start, stop):

            # the cell the weight lands in, and the row of the kernel for it
            r = indices_output[q]
            row = 0
            stride = 1
            for a in range(ndim - 1, -1, -1):
                c = r % shape_output[a]
                r = r // shape_output[a]
                coordinates[a] = c
                if shape_grid[a] > 1:
                    row += c * stride
                stride = stride * shape_grid[a]

            value = values[q]

            for k in range(size_kernel):

                element = kernel[row, k]
                if element == 0:
                    continue

                index_box = 0
                inside = True
                for a in range(ndim):
                    position = coordinates[a] + offsets[k, a] - lower[a]
                    if position < 0 or position >= extent[a]:
                        inside = False
                        break
                    index_box += position * strides[a]

                if not inside:
                    continue

                scratch_values[index_scratch + index_box] += value * element
                scratch_reached[index_scratch + index_box] = True

        index_input = indices_input[start]

        count = 0

        for t in range(size):

            if not scratch_reached[index_scratch + t]:
                continue

            if write:
                # the cell of the box, visited in the order of its flattened
                # index in the grid
                index_cell = 0
                stride = 1
                r = t
                for a in range(ndim - 1, -1, -1):
                    c = lower[a] + r % extent[a]
                    r = r // extent[a]
                    index_cell += c * stride
                    stride = stride * shape_output[a]

                result_input[index_write + count] = index_input
                result_output[index_write + count] = index_cell
                result_values[index_write + count] = scratch_values[index_scratch + t]

            count += 1

        return count

    convolve_run = jit(convolve_run)

    return starts_run, box_run, convolve_run

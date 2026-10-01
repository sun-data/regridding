"""
The convolution of a set of weights with a kernel, in a form both the host
and a CUDA device can compile.

:mod:`._host` runs this on the CPU through :func:`numba.njit`, and
:mod:`._cuda` runs the same source on a GPU through :func:`numba.cuda.jit`,
which is the arrangement the clipping kernel uses as well: a fix to the
convolution applies to both, and the device cannot drift away from the host
it is tested against.

The work is divided into *runs*: maximal stretches of consecutive weights
which share an input cell.  A run is convolved on its own, by gathering
rather than scattering: every output cell its convolved footprint can reach
sums the contributions of the run's weights to it.  That needs no scratch
space to merge into and no atomics, so each run is one independent thread,
and since it visits the cells of its footprint in order, a run comes out
ordered by output cell and with no cell twice.  Weights ordered by input
cell, as :func:`regridding.weights` returns them, are therefore convolved
into the same ordered, merged form they arrived in.

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

__all__ = [
    "num_axis",
    "build",
]

num_axis = 8
"""
The most axes a kernel can act along.

Each run keeps the bounding box of its footprint in two arrays with one
element per axis, and a device has to know the size of an array like that
when the kernel is compiled.  Weights resample one or two axes in practice,
so this is generous.
"""


def _valid(indices_input, indices_output, w, skip_negative):
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


def build(jit: Callable[[Callable], Any]) -> tuple[Callable, Callable]:
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
    The compiled ``(starts_run, convolve_run)``.
    """

    valid = jit(_valid)

    def starts_run(indices_input, indices_output, w, skip_negative):
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

    def convolve_run(
        indices_input,
        indices_output,
        values,
        kernel,
        varying,
        shape_output,
        shape_kernel,
        skip_negative,
        start,
        lower,
        extent,
        write,
        index_write,
        result_input,
        result_output,
        result_values,
    ):
        """
        Convolve the run of weights beginning at `start`, and return how
        many weights it becomes.

        The light that a weight sends to output cell :math:`i` is spread
        over the cells around it by the kernel, which is centered on
        :math:`i`.  The footprint of the run is the bounding box of its
        output cells grown by the kernel, clipped to the grid: light the
        kernel spreads beyond the grid is lost, as it would be off the edge
        of a sensor.  Each cell of the footprint is visited in order and
        gathers what each weight of the run sends it.  A cell no weight
        reaches with a nonzero element of the kernel is skipped, so the
        count depends only on where the kernel is nonzero and is the same
        whether or not anything is written.

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
            The kernel, flattened to one row for every output cell if it
            varies between them, or to a single row if it does not.
        varying
            Whether `kernel` has a row for every output cell.
        shape_output
            The number of output cells along each resampled axis.
        shape_kernel
            The length of the kernel along each resampled axis.
        skip_negative
            Whether a negative index marks an empty slot.
        start
            The index of the first weight of the run.
        lower
            Scratch space with one element per axis, for the lower corner
            of the footprint.
        extent
            Scratch space with one element per axis, for the size of the
            footprint.
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

        # the output cells of the run lie inside the grid, and the grown box
        # contains them, so clipping it to the grid never leaves it empty
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

        count = 0

        for t in range(size):

            # the cell of the footprint, visited in the order of its
            # flattened index in the grid
            index_cell = 0
            stride = 1
            r = t
            for a in range(ndim - 1, -1, -1):
                c = lower[a] + r % extent[a]
                r = r // extent[a]
                index_cell += c * stride
                stride = stride * shape_output[a]

            total = 0.0
            reached = False

            for q in range(start, stop):

                index_kernel = 0
                stride = 1
                inside = True
                r_cell = index_cell
                r_weight = indices_output[q]
                for a in range(ndim - 1, -1, -1):
                    c_cell = r_cell % shape_output[a]
                    c_weight = r_weight % shape_output[a]
                    r_cell = r_cell // shape_output[a]
                    r_weight = r_weight // shape_output[a]
                    k = c_cell - c_weight + shape_kernel[a] // 2
                    if k < 0 or k >= shape_kernel[a]:
                        inside = False
                        break
                    index_kernel += k * stride
                    stride = stride * shape_kernel[a]

                if not inside:
                    continue

                row = 0
                if varying:
                    row = indices_output[q]

                element = kernel[row, index_kernel]
                if element == 0:
                    continue

                reached = True
                total += values[q] * element

            if reached:
                if write:
                    result_input[index_write + count] = index_input
                    result_output[index_write + count] = index_cell
                    result_values[index_write + count] = total
                count += 1

        return count

    convolve_run = jit(convolve_run)

    return starts_run, convolve_run

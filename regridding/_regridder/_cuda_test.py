from typing import Any
import pytest
import numpy as np
import astropy.units as u
from numba import cuda
import regridding
from regridding._regridder import _regridder_test

requires_cuda = pytest.mark.cuda
"""
Mark a test as needing a CUDA device.

The mark is what the `tests-cuda` workflow selects on and what `conftest`
skips on, so a test says once that it needs a device.
"""


def _regridder(role: str) -> regridding.Regridder:
    """The operator on the host for one of the roles of the host tests."""
    regridder, _, _ = _regridder_test.TestRoles()._regridder("interleaved", role)
    return regridder


@requires_cuda
@pytest.mark.parametrize(argnames="role", argvalues=list(_regridder_test.roles))
def test_matches_host(role: str) -> None:
    """
    The operator and its transpose give what they do on the host, to within
    the rounding of a sum taken in a different order, and exactly the same
    numbers every time.
    """
    import torch

    regridder = _regridder(role)
    rng = np.random.default_rng(9)

    for operator in (regridder, regridder.T):
        values = rng.normal(size=(3,) + operator.shape_input)
        expected = operator(values)

        on_device = operator.to("cuda")
        results = [
            on_device(torch.as_tensor(values, device="cuda")).cpu().numpy()
            for _ in range(3)
        ]

        assert np.allclose(results[0], expected, rtol=1e-13, atol=1e-15)
        for result in results[1:]:
            assert np.array_equal(result, results[0])


@requires_cuda
def test_device_array() -> None:
    """
    An array of :mod:`numba` gives one back, and a tensor of :mod:`torch`
    gives a tensor, with the same numbers.
    """
    import torch

    regridder = _regridder("ctis").to("cuda")
    values = np.random.default_rng(10).normal(size=regridder.shape_input)

    result_numba = regridder(cuda.to_device(values))
    result_torch = regridder(torch.as_tensor(values, device="cuda"))

    assert cuda.is_cuda_array(result_numba)
    assert isinstance(result_torch, torch.Tensor)
    assert np.array_equal(result_numba.copy_to_host(), result_torch.cpu().numpy())


@requires_cuda
@pytest.mark.parametrize(
    argnames="dtype_values,dtype_matrix,dtype_result",
    argvalues=_regridder_test.dtypes,
)
def test_call_dtype(
    dtype_values: type,
    dtype_matrix: type,
    dtype_result: type,
) -> None:
    """
    The product has the type it has on the host, and is the product of the
    same numbers in double precision on the device, rounded once, since the
    rows are summed in the same order whatever the type.
    """
    import torch

    regridder = _regridder_test._ctis(dtype_matrix)
    rng = np.random.default_rng(13)

    for operator in (regridder, regridder.T):
        values = 10 * rng.normal(size=(3,) + operator.shape_input)
        values = values.astype(dtype_values)

        on_device = operator.to("cuda")
        double = _regridder_test._double(operator).to("cuda")

        result = on_device(torch.as_tensor(values, device="cuda")).cpu().numpy()
        expected = double(torch.as_tensor(values, device="cuda", dtype=torch.float64))
        expected = expected.cpu().numpy()

        assert result.dtype == dtype_result
        assert result.dtype == operator(values).dtype
        assert np.array_equal(result, expected.astype(dtype_result))


@requires_cuda
def test_call_invalid() -> None:
    """Values which do not broadcast to the input raise, as on the host."""
    import torch

    regridder = _regridder("ctis").to("cuda")
    values = torch.ones((2, 3, 8, 8), device="cuda")
    with pytest.raises(ValueError, match="cannot be broadcast"):
        regridder(values)


@requires_cuda
@pytest.mark.parametrize(
    argnames="per_row,width",
    argvalues=[(1, 1), (3, 2), (6, 4), (12, 8), (24, 16), (100, 32)],
)
def test_matmul(per_row: int, width: int) -> None:
    """
    Rows summed by groups of each size give the product of the dense matrix,
    to within the rounding of a sum taken in a different order, and exactly
    the same numbers every time and for each column alone.
    """
    from regridding._regridder import _cuda as _cuda_regridder

    rng = np.random.default_rng(12)
    num_rows = 1001
    num_columns = 500
    lengths = rng.integers(0, 2 * per_row + 1, size=num_rows)
    indptr = np.zeros(num_rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = rng.integers(0, num_columns, size=indptr[~0]).astype(np.int32)
    data = rng.normal(size=indptr[~0])
    x = rng.normal(size=(num_columns, 3))

    assert _cuda_regridder._width(data.size, num_rows) == width

    dense = np.zeros((num_rows, num_columns))
    np.add.at(dense, (np.repeat(np.arange(num_rows), lengths), indices), data)
    expected = dense @ x

    matrix = tuple(cuda.to_device(a) for a in (indptr, indices, data))

    def product(x: np.ndarray) -> np.ndarray:
        """The product of the matrix with `x` on the device."""
        y = cuda.device_array((num_rows, x.shape[1]))
        _cuda_regridder.matmul(*matrix, cuda.to_device(x), y)
        return y.copy_to_host()

    result = product(x)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.array_equal(product(x), result)
    for t in range(x.shape[1]):
        column = np.ascontiguousarray(x[:, t : t + 1])
        assert np.array_equal(product(column), result[:, t : t + 1])


@requires_cuda
def test_from_device_weights() -> None:
    """
    Weights built on the device are brought back to the host with their
    empty slots dropped, and give what resampling with them does.
    """
    from regridding._weights._weights_transposed import _cuda_test

    kwargs: dict[str, Any] = dict(method="conservative", **_cuda_test.grids)
    weights = regridding.weights(device="cuda", **kwargs)

    regridder = regridding.Regridder.from_weights(
        weights,
        axis_input=_cuda_test.axes["axis_input"],
        axis_output=_cuda_test.axes["axis_output"],
    )

    host = _cuda_test._to_host(weights)
    assert regridder.num_entries == sum(e[2].size for e in host[0].reshape(-1))

    values = np.random.default_rng(11).normal(size=regridder.shape_input)
    expected = regridding.regrid_from_weights(
        *host,
        values_input=values,
        **_cuda_test.axes,
    )

    assert np.array_equal(regridder(values), expected)


def _assert_same(device: regridding.Regridder, host: regridding.Regridder) -> None:
    """
    Check that an operator on the device has exactly the matrix of one on
    the host, and the same shapes.
    """
    assert device.device == "cuda"
    assert device.shape_input == host.shape_input
    assert device.shape_output == host.shape_output
    for a, e in (
        (device.indptr, host.indptr),
        (device.indices, host.indices),
        (device.data, host.data),
    ):
        a = a.cpu().numpy()
        assert a.dtype == e.dtype
        assert np.array_equal(a, e)


@requires_cuda
@pytest.mark.parametrize(argnames="role", argvalues=list(_regridder_test.roles))
def test_assemble(role: str) -> None:
    """
    An operator assembled on the device, its transpose and its conservative
    transpose are exactly those assembled on the host, entry for entry.
    """
    host, weights, grids = _regridder_test.TestRoles()._regridder("interleaved", role)
    grid_input, grid_output, axis_input, axis_output = grids

    device = regridding.Regridder.from_weights(
        weights,
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=host.shape_input,
        shape_output=host.shape_output,
        device="cuda",
    )

    _assert_same(device, host)
    _assert_same(device.T, host.T)
    assert device.T.T is device

    if role not in ("both", "all"):
        _assert_same(
            device.transpose_conservative(grid_input, grid_output),
            host.transpose_conservative(grid_input, grid_output),
        )


@requires_cuda
def test_assemble_device_weights() -> None:
    """
    Weights built on the device are assembled where they are, into exactly
    the matrix the host assembles from the same weights.
    """
    from regridding._weights._weights_transposed import _cuda_test

    kwargs: dict[str, Any] = dict(method="conservative", **_cuda_test.grids)
    weights = regridding.weights(device="cuda", **kwargs)

    host = regridding.Regridder.from_weights(weights, **_cuda_test.axes)
    device = regridding.Regridder.from_weights(
        weights, **_cuda_test.axes, device="cuda"
    )

    _assert_same(device, host)
    _assert_same(device.T, host.T)


@requires_cuda
def test_assemble_single() -> None:
    """
    An operator in single precision is the double precision one rounded,
    on the device as on the host, and resamples to within that rounding.
    """
    import torch

    host, weights, (_, _, axis_input, axis_output) = (
        _regridder_test.TestRoles()._regridder("trailing", "ctis")
    )
    kwargs: dict[str, Any] = dict(
        axis_input=axis_input,
        axis_output=axis_output,
        shape_input=host.shape_input,
        shape_output=host.shape_output,
        dtype=np.float32,
    )
    single_host = regridding.Regridder.from_weights(weights, **kwargs)
    single_device = regridding.Regridder.from_weights(weights, **kwargs, device="cuda")

    assert np.array_equal(single_host.data, host.data.astype(np.float32))
    _assert_same(single_device, single_host)

    values = np.random.default_rng(12).normal(size=host.shape_input)
    expected = host(values)
    result = single_device(torch.as_tensor(values, device="cuda")).cpu().numpy()
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


@requires_cuda
def test_to() -> None:
    """Moving an operator to the device and back gives the same matrix."""
    host = _regridder("ctis")
    device = host.to("cuda")

    _assert_same(device, host)
    assert device.to("cuda") is device

    back = device.to(None)
    assert back.device is None
    for a, e in ((back.indptr, host.indptr), (back.indices, host.indices)):
        assert np.array_equal(a, e)
    assert np.array_equal(back.data, host.data)


@requires_cuda
@pytest.mark.parametrize(
    argnames="unit",
    argvalues=[u.percent, u.s],
    ids=["percent", "seconds"],
)
def test_transpose_conservative_units(unit: Any) -> None:
    """
    The conservative transpose of weights with a unit is scaled on the
    device exactly as on the host.
    """
    weights, grids = _regridder_test._weights_trailing()
    grid_input, grid_output, axis_input, axis_output = grids
    weights = _regridder_test._with_unit(weights, unit)

    host = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    device = regridding.Regridder.from_weights(
        weights, axis_input, axis_output, device="cuda"
    )

    _assert_same(
        device.transpose_conservative(grid_input, grid_output),
        host.transpose_conservative(grid_input, grid_output),
    )


@requires_cuda
def test_assemble_empty_element() -> None:
    """
    Elements of the weights with no entries, from input grids which miss the
    output grid entirely, are assembled on the device as on the host.
    """
    grid_input, grid_output, axis_input, axis_output = _regridder_test._grids(
        "trailing"
    )
    x, y = grid_input
    x = x.copy()
    x[0] += 100
    weights = regridding.weights(
        (x, y),
        grid_output,
        axis_input=axis_input,
        axis_output=axis_output,
        method="conservative",
    )
    assert min(e[2].size for e in weights[0].reshape(-1)) == 0

    host = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    device = regridding.Regridder.from_weights(
        weights, axis_input, axis_output, device="cuda"
    )

    _assert_same(device, host)
    _assert_same(device.T, host.T)


@requires_cuda
def test_empty() -> None:
    """
    Weights with no elements give an operator with no entries on the device,
    as on the host, which can still be applied.
    """
    import torch

    weights = _regridder_test._weights_empty()
    host = regridding.Regridder.from_weights(weights, -1, -1)
    device = regridding.Regridder.from_weights(weights, -1, -1, device="cuda")

    _assert_same(device, host)
    values = torch.ones(device.shape_input, device="cuda", dtype=torch.float64)
    assert tuple(device(values).shape) == host.shape_output


def _operator_slow() -> regridding.Regridder:
    """An operator on the device with enough entries to be worth racing."""
    x_input = np.linspace(0, 1, 1_000_001)
    x_output = np.linspace(-0.01, 1.01, 300_001)
    weights = regridding.weights((x_input,), (x_output,), method="conservative")
    return regridding.Regridder.from_weights(weights, device="cuda")


def _delay() -> None:
    """Queue some tens of milliseconds of work on the current stream."""
    import torch

    a = torch.rand(4096, 4096, device="cuda")
    for _ in range(4):
        a = a @ a
        a = a / a.norm()


@requires_cuda
def test_stream() -> None:
    """
    On a stream of :mod:`torch` other than the default, a product waits for
    the values :mod:`torch` is still computing there, and is done before
    :mod:`torch` reads it there.
    """
    import torch

    regridder = _operator_slow()
    values = torch.rand(regridder.shape_input, device="cuda", dtype=torch.float64)
    expected = regridder(values).cpu()

    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(5):
            _delay()
            late = values * 1
            assert torch.equal(regridder(late).cpu(), expected)


# this runs on the device, where `coverage` cannot follow it
@cuda.jit
def _copy_late(source: Any, target: Any) -> None:  # pragma: nocover
    """Copy `source` to `target`, a few tens of milliseconds late."""
    i = cuda.grid(1)  # type: ignore[call-arg]
    if i < source.size:
        for _ in range(20):
            cuda.nanosleep(1_000_000)
        target[i] = source[i]


@requires_cuda
def test_stream_numba() -> None:
    """
    On a stream of :mod:`torch` other than the default, an array of
    :mod:`numba` is read once :mod:`numba` has written it, and the result is
    done before :mod:`numba` reads it, both on its default stream.
    """
    import torch

    regridder = _operator_slow()
    values = np.random.default_rng(16).normal(size=regridder.shape_input)
    expected = regridder(torch.as_tensor(values, device="cuda")).cpu().numpy()

    source = cuda.to_device(values)
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(3):
            # values which `numba` is still writing, with the result read
            # where `torch` computed it
            target = cuda.device_array_like(source)
            blocks = (source.size + 255) // 256
            _copy_late[blocks, 256](source, target)  # type: ignore[index]
            result = regridder(target)
            result = torch.as_tensor(result, device="cuda").cpu().numpy()
            assert np.array_equal(result, expected)

            # a result which `torch` is still computing, read by `numba`
            _delay()
            result = regridder(source)
            assert np.array_equal(result.copy_to_host(), expected)


@requires_cuda
def test_call_host() -> None:
    """
    Values on the host give a result on the host, read as :mod:`numpy` reads
    them, so that a list is not rounded to single precision on its way and
    the result has the type it has on the host.
    """
    regridder = _regridder("ctis")
    device = regridder.to("cuda")
    values = np.random.default_rng(17).normal(size=regridder.shape_input) + 1e-9

    for v in (values, values.tolist()):
        result = device(v)
        expected = regridder(v)
        assert isinstance(result, np.ndarray)
        assert result.dtype == expected.dtype == np.float64
        assert np.allclose(result, expected, rtol=1e-13, atol=1e-15)

    single = _regridder_test._ctis(np.float32)
    assert single.to("cuda")(1.0).dtype == single(1.0).dtype == np.float64


@requires_cuda
def test_call_units() -> None:
    """
    Values on the host get the units they get on the host, while values on
    the device raise rather than lose the unit of the operator.
    """
    import torch

    weights, (_, _, axis_input, axis_output) = _regridder_test._weights_trailing()
    weights = _regridder_test._with_unit(weights, u.percent)
    host = regridding.Regridder.from_weights(weights, axis_input, axis_output)
    device = host.to("cuda")

    values = np.random.default_rng(18).normal(size=host.shape_input) << u.ph
    result = device(values)
    expected = host(values)
    assert result.unit == expected.unit
    assert np.allclose(result.value, expected.value, rtol=1e-13, atol=1e-15)

    for on_device in (
        torch.as_tensor(values.value, device="cuda"),
        cuda.to_device(values.value),
    ):
        with pytest.raises(TypeError, match="unit"):
            device(on_device)


@requires_cuda
def test_call_reassigned() -> None:
    """The arrays assigned to an operator on the device are the ones applied."""
    import torch

    device = _regridder("ctis").to("cuda")
    values = np.random.default_rng(19).normal(size=device.shape_input)
    values = torch.as_tensor(values, device="cuda")

    before = device(values)
    device.data = device.data * 2
    assert torch.equal(device(values), 2 * before)

from typing import Any
import pytest
import numpy as np
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
def test_call_invalid() -> None:
    """Values which do not broadcast to the input raise, as on the host."""
    import torch

    regridder = _regridder("ctis").to("cuda")
    values = torch.ones((2, 3, 8, 8), device="cuda")
    with pytest.raises(ValueError, match="cannot be broadcast"):
        regridder(values)


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

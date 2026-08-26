"""
Test configuration shared by the whole package.

The device tests are marked rather than named, so that what selects them on
a runner with a GPU is the same thing that skips them on one without.

This lives inside the package rather than beside it because the tests do:
they are shipped, so anything which decides whether to run them has to be
shipped with them or an installed copy runs them on a machine with no
device and fails instead of skipping.
"""

import pytest
from regridding import _cuda


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Skip the tests which need a CUDA device when there is not one.

    Parameters
    ----------
    config
        The active configuration.
    items
        The collected tests.
    """
    if _cuda.available():
        return

    skip = pytest.mark.skip(reason="a CUDA device is needed to run this")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)

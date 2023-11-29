import numba

__all__ = [
    "sign",
]


@numba.njit(inline="always", error_model="numpy")
def sign(x: float):
    """
    Numba-compiled version of the `sign function <https://en.wikipedia.org/wiki/Sign_function>`_

    Parameters
    ----------
    x
        the value to find the sign of
    """
    return bool(x > 0) - bool(x < 0)

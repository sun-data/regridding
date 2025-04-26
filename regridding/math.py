import numba

__all__ = [
    "sign",
    "negate_3d",
    "sum_3d",
    "difference_3d",
    "multiply_3d",
    "dot_3d",
    "cross_3d",
]


@numba.njit(cache=True, inline="always", error_model="numpy")
def sign(x: float):
    """
    Numba-compiled version of the `sign function <https://en.wikipedia.org/wiki/Sign_function>`_

    Parameters
    ----------
    x
        the value to find the sign of
    """
    return bool(x > 0) - bool(x < 0)


@numba.njit(cache=True, inline="always", error_model="numpy")
def negate_3d(
    a: tuple[float, float, float],
) -> tuple[float, float, float]:
    r"""
    Compute :math:`-a` where :math:`a` is a 3D vector.

    Parameters
    ----------
    a
        A 3D vector.
    """
    x, y, z = a

    return -x, -y, -z


@numba.njit(cache=True, inline="always", error_model="numpy")
def sum_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    r"""
    Compute the sum :math:`a + b` between two 3D vectors :math:`a`
    and :math:`b`.

    Parameters
    ----------
    a
        A 3D vector.
    b
        Another 3D vector.
    """
    x_a, y_a, z_a = a
    x_b, y_b, z_b = b

    x = x_a + x_b
    y = y_a + y_b
    z = z_a + z_b

    return x, y, z


@numba.njit(cache=True, inline="always", error_model="numpy")
def difference_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    r"""
    Compute the difference :math:`a - b` between two 3D vectors :math:`a`
    and :math:`b`.

    Parameters
    ----------
    a
        A 3D vector.
    b
        Another 3D vector.
    """
    x_a, y_a, z_a = a
    x_b, y_b, z_b = b

    x = x_a - x_b
    y = y_a - y_b
    z = z_a - z_b

    return x, y, z


@numba.njit(cache=True, inline="always", error_model="numpy")
def multiply_3d(
    r: float,
    a: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Multiply a 3D vector :math:`a` by a scalar :math:`r`.

    Parameters
    ----------
    r
        Scalar operand.
    a
        Vector operand.
    """
    x, y, z = a

    return r * x, r * y, r * z


@numba.njit(cache=True, inline="always", error_model="numpy")
def dot_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    r"""
    Compute the scalar product :math:`a \cdot b` between two 3D vectors :math:`a`
    and :math:`b`.

    Parameters
    ----------
    a
        A 3D vector.
    b
        Another 3D vector.
    """
    x_a, y_a, z_a = a
    x_b, y_b, z_b = b

    return x_a * x_b + y_a * y_b + z_a * z_b


@numba.njit(cache=True, inline="always", error_model="numpy")
def cross_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    r"""
    Compute the vector product :math:`a \times b` between two 3D vectors :math:`a`
    and :math:`b`.

    Parameters
    ----------
    a
        A 3D vector.
    b
        Another 3D vector.
    """

    x_a, y_a, z_a = a
    x_b, y_b, z_b = b

    x = +(y_a * z_b - z_a * y_b)
    y = -(x_a * z_b - z_a * x_b)
    z = +(x_a * y_b - y_a * x_b)

    return x, y, z

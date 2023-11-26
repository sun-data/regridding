import numpy as np
import regridding


def _values_input(num_x: int, num_y: int) -> np.ndarray:
    return np.random.uniform(0, 1, size=(num_x - 1, num_y - 1))


def _vertices_input(num_x: int, num_y: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    x, y = np.meshgrid(x, y, indexing="ij")

    angle = 0.4
    x_result = x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x
    y_result = x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y

    return x_result, y_result


def _vertices_output(num_x: int, num_y: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2, 2, num=num_x)
    y = np.linspace(-2, 2, num=num_y)
    x, y = np.meshgrid(x, y, indexing="ij")
    return x, y


def time_regrid(
    vertices_input: tuple[np.ndarray, ...],
    vertices_output: tuple[np.ndarray, ...],
    values_input: np.ndarray,
    method: str,
    order: int,
):
    regridding.regrid(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        values_input=values_input,
        method=method,
        order=order
    )


time_regrid.params = (
    [_vertices_input(64, 64)],
    [_vertices_output(64, 64)],
    [_values_input(64, 64)],
    ["conservative"],
    [1],
)
time_regrid.param_names = (
    "vertices_input",
    "vertices_output",
    "values_input",
    "method",
    "order",
)

import numpy as np
import regridding

coordinates_input = None
coordinates_output = None
values_input = None


def setup_regrid_2d(
    num: int,
    method: str,
):
    num_x = num_y = num

    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    x, y = np.meshgrid(x, y, indexing="ij")

    angle = 0.4
    x_input = x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x
    y_input = x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y

    x_output = np.linspace(x_input.min(), x_input.max(), num=num_x)
    y_output = np.linspace(y_input.min(), y_input.max(), num=num_y)
    x_output, y_output = np.meshgrid(x_output, y_output, indexing="ij")

    global coordinates_input
    global coordinates_output
    global values_input

    coordinates_input = (x_input, y_input)
    coordinates_output = (x_output, y_output)
    values_input = np.random.uniform(0, 1, size=(num_x - 1, num_y - 1))

    regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        method=method,
    )


def time_regrid_2d(
    num: int,
    method: str,
):
    regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        method=method,
    )


time_regrid_2d.setup = setup_regrid_2d
time_regrid_2d.params = (
    [100, 200, 300, 400, 500],
    ["conservative"],
)
time_regrid_2d.param_names = (
    "number of edges per axis",
    "method",
)
time_regrid_2d.timeout = 480

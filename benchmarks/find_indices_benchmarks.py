import numpy as np
import regridding

vertices_input = None
vertices_output = None


def setup_find_indices_1d(num: int, method: str):
    global vertices_input
    global vertices_output
    vertices_input = np.linspace(-1, 1, num=num)
    vertices_output = np.linspace(-1.1, 1.1, num=num)


def time_find_indices_1d(num: int, method: str):
    regridding.find_indices(
        vertices_input=vertices_input,
        vertices_output=vertices_output,
        method=method,
    )


time_find_indices_1d.setup = setup_find_indices_1d
time_find_indices_1d.params = ([10, 100, 1000, 10000], ["brute"])
time_find_indices_1d.param_names = ["num", "method"]
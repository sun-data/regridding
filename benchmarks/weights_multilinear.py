import numpy as np
import regridding

coordinates_input: tuple[np.ndarray, ...] = ()
coordinates_output: tuple[np.ndarray, ...] = ()


def setup_weights_multilinear_2d(
    num_point: int,
    grid: str,
) -> None:
    """
    A grid of about a million vertices, and random points over it.

    The ``rectilinear`` grid is searched along each axis on its own, and the
    ``curvilinear`` one, which is the same grid rotated and distorted,
    through its hierarchy of bounding boxes.
    """
    global coordinates_input
    global coordinates_output

    x, y = np.meshgrid(
        np.linspace(-1, 1, num=1001),
        np.linspace(-1, 1, num=1000),
        indexing="ij",
    )

    if grid == "curvilinear":
        angle = 0.4
        x, y = (
            x * np.cos(angle) - y * np.sin(angle) + 0.05 * x * x,
            x * np.sin(angle) + y * np.cos(angle) + 0.05 * y * y,
        )

    rng = np.random.default_rng(0)
    coordinates_input = (x, y)
    coordinates_output = (
        rng.uniform(x.min(), x.max(), size=num_point),
        rng.uniform(y.min(), y.max(), size=num_point),
    )

    time_weights_multilinear_2d(num_point, grid)


def time_weights_multilinear_2d(
    num_point: int,
    grid: str,
) -> None:
    regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        method="multilinear",
        coalesce=False,
    )


time_weights_multilinear_2d.setup = setup_weights_multilinear_2d
time_weights_multilinear_2d.params = (
    [1, 10_000, 1_000_000],
    ["rectilinear", "curvilinear"],
)
time_weights_multilinear_2d.param_names = (
    "number of points",
    "grid",
)
time_weights_multilinear_2d.timeout = 240

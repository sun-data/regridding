import sys
import math
import numpy as np
import numba
# import matplotlib.pyplot as plt

__all__ = []


@numba.njit(
    error_model="numpy",
    # parallel=True,
    # inline="always",
    # boundscheck=True,
)
def _conservative_ramshaw(
        # values_input: np.ndarray,
        # values_output: np.ndarray,
        grid_input: tuple[np.ndarray, np.ndarray],
        grid_output: tuple[np.ndarray, np.ndarray],
        epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    weights = numba.typed.List()
    for x in range(0):
        weights.append((0., 0., 0.))

    input_x, input_y = grid_input
    # output_x, output_y = grid_output

    shape_input = input_x.shape
    # shape_output = np.broadcast_shapes(output_x.shape, output_y.shape)

    grids_sweep = grid_input, grid_output
    grids_static = grid_output, grid_input
    grids_input = "sweep", "static"
    axes = 0, 1

    # k = slice(None, -1)
    # # k = slice(1, None)
    # grids_sweep = grids_sweep[k]
    # grids_static = grids_static[k]
    # grids_input = grids_input[k]
    # axes = 1,

    area_input = np.zeros((shape_input[0] - 1, shape_input[1] - 1))
    for axis in axes:
        _grid_area_sweep(
            out=area_input,
            grid_x=grid_input[0],
            grid_y=grid_input[1],
            axis=axis,
        )

    # values_input = values_input / area_input

    for grid_sweep, grid_static, grid_input in zip(grids_sweep, grids_static, grids_input):
        grid_static_x, grid_static_y = grid_static
        grid_sweep_x, grid_sweep_y = grid_sweep
        for axis in axes:
            _sweep_axis(
                # values_input=values_input,
                # values_output=values_output,
                area_input=area_input,
                grid_sweep_x=grid_sweep_x,
                grid_sweep_y=grid_sweep_y,
                grid_static_x=grid_static_x,
                grid_static_y=grid_static_y,
                axis=axis,
                grid_input=grid_input,
                epsilon=epsilon,
                weights=weights,
            )

    # return values_output
    return weights


@numba.njit(error_model="numpy", parallel=True)
def _sweep_axis(
        # values_input: np.ndarray,
        # values_output: np.ndarray,
        area_input: np.ndarray,
        grid_static_x: np.ndarray,
        grid_static_y: np.ndarray,
        grid_sweep_x: np.ndarray,
        grid_sweep_y: np.ndarray,
        axis: int,
        grid_input: str,
        epsilon: float,
        weights: numba.typed.List,
) -> None:

    if grid_input == "static":
        input_is_sweep = False
    elif grid_input == "sweep":
        input_is_sweep = True
    else:
        print(f"The `grid_input` argument must have the value 'static' or 'sweep', got '{grid_input}'")

    # print(input_is_sweep)

    if axis == 0:
        grid_sweep_y, grid_sweep_x = grid_sweep_x.T, grid_sweep_y.T
        grid_static_y, grid_static_x = grid_static_x, grid_static_y
        # if input_is_sweep:
        #     values_input = values_input.T
        # else:
        #     values_output = values_output.T
    elif axis == 1:
        pass
    else:
        print(f"The `axis` argument must be 0 or 1, got {axis}")

    shape_static_x, shape_static_y = list(grid_static_x.shape)
    shape_sweep_x, shape_sweep_y = list(grid_sweep_x.shape)

    edges_bottom = slice(None, ~0), 0
    edges_right = ~0, slice(None, ~0)
    edges_top = slice(None, 0, -1), ~0
    edges_left = 0, slice(None, 0, -1)

    edges_border_static_x = np.concatenate((
        grid_static_x[edges_bottom],
        grid_static_x[edges_right],
        grid_static_x[edges_top],
        grid_static_x[edges_left],
    ))
    edges_border_static_y = np.concatenate((
        grid_static_y[edges_bottom],
        grid_static_y[edges_right],
        grid_static_y[edges_top],
        grid_static_y[edges_left],
    ))

    cells_bottom = slice(None), 0
    cells_right = ~0, slice(None)
    cells_top = slice(None, None, -1), ~0
    cells_left = 0, slice(None, None, -1)

    grid_static_m = np.arange(shape_static_x)[:, np.newaxis]
    grid_static_n = np.arange(shape_static_y)[np.newaxis, :]
    grid_static_m, grid_static_n = np.broadcast_arrays(grid_static_m, grid_static_n)
    edges_border_static_m = np.concatenate((
        grid_static_m[edges_bottom],
        grid_static_m[edges_right],
        grid_static_m[edges_top],
        grid_static_m[edges_left],
    ))
    edges_border_static_n = np.concatenate((
        grid_static_n[edges_bottom],
        grid_static_n[edges_right],
        grid_static_n[edges_top],
        grid_static_n[edges_left],
    ))

    cells_static_m = np.arange(shape_static_x - 1)[:, np.newaxis]
    cells_static_n = np.arange(shape_static_y - 1)[np.newaxis, :]
    cells_static_m, cells_static_n = np.broadcast_arrays(cells_static_m, cells_static_n)
    cells_border_static_m = np.concatenate((
        cells_static_m[cells_bottom],
        cells_static_m[cells_right],
        cells_static_m[cells_top],
        cells_static_m[cells_left],
    ))
    cells_border_static_n = np.concatenate((
        cells_static_n[cells_bottom],
        cells_static_n[cells_right],
        cells_static_n[cells_top],
        cells_static_n[cells_left],
    ))

    weight = numba.typed.List()

    for i in range(shape_sweep_x):
        w = numba.typed.List()
        for _ in range(0):
            w.append((0., 0., 0.))
        weight.append(w)

    for i in numba.prange(shape_sweep_x):

        sweep_is_inside_static = False
        j = 0

        point_sweep_1x = grid_sweep_x[i, j]
        point_sweep_1y = grid_sweep_y[i, j]

        if _point_is_inside_polygon(
                vertices_x=edges_border_static_x,
                vertices_y=edges_border_static_y,
                point_x=point_sweep_1x,
                point_y=point_sweep_1y,
        ):
            # print("initial point inside...")
            m, n = _indices_of_point_brute(
                grid_x=grid_static_x,
                grid_y=grid_static_y,
                point_x=point_sweep_1x,
                point_y=point_sweep_1y,
            )
            sweep_is_inside_static = True
        else:
            m = sys.maxsize
            n = sys.maxsize

        m_old = sys.maxsize
        n_old = sys.maxsize

        while True:

            point_sweep_2x = grid_sweep_x[i, j + 1]
            point_sweep_2y = grid_sweep_y[i, j + 1]

            # print("i", i)
            # print("j", j)
            # print("m", m)
            # print("n", n)
            # print("point_sweep_1", (point_sweep_1x, point_sweep_1y))
            # print("point_sweep_2", (point_sweep_2x, point_sweep_2y))
            #
            # plt.figure()
            # plt.plot(grid_static_x, grid_static_y, color="black")
            # plt.plot(grid_static_x.T, grid_static_y.T, color="black")
            # plt.plot(grid_sweep_x, grid_sweep_y, color="red")
            # plt.plot(grid_sweep_x.T, grid_sweep_y.T, color="red")
            # plt.plot([point_sweep_1x, point_sweep_2x], [point_sweep_1y, point_sweep_2y], zorder=10)

            if not sweep_is_inside_static:
                point_sweep_2x, point_sweep_2y, j_new, m_new, n_new, sweep_is_inside_static = _step_outside_static(
                    # area_output=area_output,
                    grid_static_x=grid_static_x,
                    grid_static_y=grid_static_y,
                    edges_border_static_m=edges_border_static_m,
                    edges_border_static_n=edges_border_static_n,
                    cells_border_static_m=cells_border_static_m,
                    cells_border_static_n=cells_border_static_n,
                    # shape_sweep_x=shape_sweep_x,
                    # shape_sweep_y=shape_sweep_y,
                    point_sweep_1x=point_sweep_1x,
                    point_sweep_1y=point_sweep_1y,
                    point_sweep_2x=point_sweep_2x,
                    point_sweep_2y=point_sweep_2y,
                    # input_is_sweep=input_is_sweep,
                    epsilon=epsilon,
                    # i=i,
                    j=j,
                    m=m,
                    n=n,
                )

            else:
                point_sweep_2x, point_sweep_2y, j_new, m_new, n_new = _step_inside_static(
                    # values_input=values_input,
                    # values_output=values_output,
                    # area_output=area_output,
                    area_input=area_input,
                    grid_static_x=grid_static_x,
                    grid_static_y=grid_static_y,
                    shape_sweep_x=shape_sweep_x,
                    shape_sweep_y=shape_sweep_y,
                    point_sweep_1x=point_sweep_1x,
                    point_sweep_1y=point_sweep_1y,
                    point_sweep_2x=point_sweep_2x,
                    point_sweep_2y=point_sweep_2y,
                    axis=axis,
                    input_is_sweep=input_is_sweep,
                    epsilon=epsilon,
                    m_old=m_old,
                    n_old=n_old,
                    i=i,
                    j=j,
                    m=m,
                    n=n,
                    weight=weight[i],
                )

            # print("j_new", j_new)
            # print("m_new", m_new)
            # print("n_new", n_new)

            # if input_is_sweep:
            #     plt.pcolormesh(grid_static_x, grid_static_y, values_output, zorder=-100)
            # else:
            #     plt.pcolormesh(grid_sweep_x, grid_sweep_y, values_output, zorder=-100)
            # plt.colorbar()

            if j_new >= (shape_sweep_y - 1):
                break
            if sweep_is_inside_static:
                if not 0 <= m_new < (shape_static_x - 1):
                    break
                if not 0 <= n_new < (shape_static_y - 1):
                    break

            j = j_new
            m_old = m
            n_old = n
            m = m_new
            n = n_new
            point_sweep_1x = point_sweep_2x
            point_sweep_1y = point_sweep_2y

            # print()

        # print("---------------------------")

    for i in range(shape_sweep_x):
        weight_i = weight[i]
        for w in range(len(weight_i)):
            weights.append(weight_i[w])

        # indices_input += index_input[i]
        # indices_output += index_output[i]
        # weights += weight[i]


@numba.njit(inline="always")
def _step_outside_static(
        # area_output: np.ndarray,
        grid_static_x: np.ndarray,
        grid_static_y: np.ndarray,
        edges_border_static_m: np.ndarray,
        edges_border_static_n: np.ndarray,
        cells_border_static_m: np.ndarray,
        cells_border_static_n: np.ndarray,
        # shape_sweep_x: int,
        # shape_sweep_y: int,
        point_sweep_1x: float,
        point_sweep_1y: float,
        point_sweep_2x: float,
        point_sweep_2y: float,
        # input_is_sweep: bool,
        epsilon: float,
        # i: int,
        j: int,
        m: int,
        n: int,

) -> tuple[float, float, int, int, int, bool]:

    # print("step outside")

    direction_sweep_x = point_sweep_2x - point_sweep_1x
    direction_sweep_y = point_sweep_2y - point_sweep_1y

    sweep_is_inside_static = False
    j_new = j + 1
    m_new = m
    n_new = n

    u_min = np.inf

    # print("m and n not set")
    for v in range(len(edges_border_static_m)):

        index_vertex_1 = v - 1
        index_vertex_2 = v

        vertex_static_1m = edges_border_static_m[index_vertex_1]
        vertex_static_1n = edges_border_static_n[index_vertex_1]
        vertex_static_2m = edges_border_static_m[index_vertex_2]
        vertex_static_2n = edges_border_static_n[index_vertex_2]

        direction_m = vertex_static_2m - vertex_static_1m
        direction_n = vertex_static_2n - vertex_static_1n

        vertex_static_1x = grid_static_x[vertex_static_1m, vertex_static_1n]
        vertex_static_1y = grid_static_y[vertex_static_1m, vertex_static_1n]

        vertex_static_2x = grid_static_x[vertex_static_2m, vertex_static_2n]
        vertex_static_2y = grid_static_y[vertex_static_2m, vertex_static_2n]

        direction_static_x = vertex_static_2x - vertex_static_1x
        direction_static_y = vertex_static_2y - vertex_static_1y

        # plt.plot(
        #     [vertex_static_1x, vertex_static_2x],
        #     [vertex_static_1y, vertex_static_2y],
        #     color="cyan"
        # )

        t, u = _two_line_segment_intersection_parameters(
            x1=vertex_static_1x, y1=vertex_static_1y,
            x2=vertex_static_2x, y2=vertex_static_2y,
            x3=point_sweep_1x, y3=point_sweep_1y,
            x4=point_sweep_2x, y4=point_sweep_2y,
        )

        if u > u_min:
            continue

        if (0 - epsilon) < u < (1 + epsilon):
            if (0 - epsilon) < t < (1 + epsilon):

                sweep_is_inside_static = True

                m_new = cells_border_static_m[index_vertex_1]
                n_new = cells_border_static_n[index_vertex_1]

                if -epsilon < t < epsilon:
                    if direction_sweep_x * direction_static_x + direction_sweep_y * direction_static_y < 0:
                        m_new = m_new - direction_m
                        n_new = n_new - direction_n

                elif (1 - epsilon) < t < (1 + epsilon):
                    if direction_sweep_x * direction_static_x + direction_sweep_y * direction_static_y > 0:
                        m_new = m_new + direction_m
                        n_new = n_new + direction_n

                if (1 - epsilon) < u < (1 + epsilon):
                    pass
                else:
                    j_new = j
                    u_min = u

                # plt.scatter(point_sweep_2x, point_sweep_2y)

                # break

        elif math.isnan(u) or math.isnan(t):

            if point_sweep_1x != point_sweep_2x:
                u1 = (vertex_static_1x - point_sweep_1x) / (point_sweep_2x - point_sweep_1x)
                u2 = (vertex_static_2x - point_sweep_1x) / (point_sweep_2x - point_sweep_1x)
            else:
                u1 = (vertex_static_1y - point_sweep_1y) / (point_sweep_2y - point_sweep_1y)
                u2 = (vertex_static_2y - point_sweep_1y) / (point_sweep_2y - point_sweep_1y)

            if u1 > u_min:
                continue

            # print("u1", u1)
            # print("u2", u2)

            if u1 > u2:
                u1, u2 = u2, u1

            if (0 - epsilon) < u1 < (1 + epsilon):

                sweep_is_inside_static = True

                m_new = cells_border_static_m[index_vertex_1]
                n_new = cells_border_static_n[index_vertex_1]

                if (1 - epsilon) < u < (1 + epsilon):
                    pass
                else:
                    j_new = j
                    u_min = u1

                # break

    # if not input_is_sweep:
    #     i_left = i - 1
    #     i_right = i
    #     area_sweep = (point_sweep_1x * point_sweep_2y - point_sweep_2x * point_sweep_1y) / 2
    #
    #     if 0 <= i_left < (shape_sweep_x - 1):
    #         area_output[i_left, j] += area_sweep
    #     if 0 <= i_right < (shape_sweep_x - 1):
    #         area_output[i_right, j] -= area_sweep

    if math.isfinite(u_min):
        point_sweep_2x = point_sweep_1x + u_min * (point_sweep_2x - point_sweep_1x)
        point_sweep_2y = point_sweep_1y + u_min * (point_sweep_2y - point_sweep_1y)

    # print("point_sweep_1", (point_sweep_1x, point_sweep_1y))
    # print("point_sweep_2", (point_sweep_2x, point_sweep_2y))

    # plt.scatter(point_sweep_1x, point_sweep_1y)
    # plt.scatter(point_sweep_2x, point_sweep_2y)

    return point_sweep_2x, point_sweep_2y, j_new, m_new, n_new, sweep_is_inside_static


@numba.njit(inline="always")
def _step_inside_static(
        # values_input: np.ndarray,
        # values_output: np.ndarray,
        area_input: np.ndarray,
        grid_static_x: np.ndarray,
        grid_static_y: np.ndarray,
        shape_sweep_x: int,
        shape_sweep_y: int,
        point_sweep_1x: float,
        point_sweep_1y: float,
        point_sweep_2x: float,
        point_sweep_2y: float,
        axis: int,
        input_is_sweep: bool,
        epsilon: float,
        m_old: int,
        n_old: int,
        i: int,
        j: int,
        m: int,
        n: int,
        weight: numba.typed.List,
) -> tuple[float, float, int, int, int]:

    # print("step inside")

    # intersection_found = False

    shape_static_x, shape_static_y = grid_static_x.shape

    vertices_static_m = (0, 1, 1, 0)
    vertices_static_n = (0, 0, 1, 1)

    direction_sweep_x = point_sweep_2x - point_sweep_1x
    direction_sweep_y = point_sweep_2y - point_sweep_1y

    # if axis == 0:
    #     normal_sweep_x = -direction_sweep_y
    #     normal_sweep_y = direction_sweep_x
    # else:
    normal_sweep_x = direction_sweep_y
    normal_sweep_y = -direction_sweep_x

    j_new = j + 1
    m_new = m
    n_new = n

    m_left = m_right = m
    n_left = n_right = n

    u_min = np.inf

    for v in range(len(vertices_static_m)):

        index_vertex_static_1 = v - 1
        index_vertex_static_2 = v

        vertex_static_1m = m + vertices_static_m[index_vertex_static_1]
        vertex_static_1n = n + vertices_static_n[index_vertex_static_1]

        vertex_static_2m = m + vertices_static_m[index_vertex_static_2]
        vertex_static_2n = n + vertices_static_n[index_vertex_static_2]

        direction_m = vertex_static_2m - vertex_static_1m
        direction_n = vertex_static_2n - vertex_static_1n

        normal_m = direction_n
        normal_n = -direction_m

        vertex_static_1x = grid_static_x[vertex_static_1m, vertex_static_1n]
        vertex_static_1y = grid_static_y[vertex_static_1m, vertex_static_1n]

        vertex_static_2x = grid_static_x[vertex_static_2m, vertex_static_2n]
        vertex_static_2y = grid_static_y[vertex_static_2m, vertex_static_2n]

        direction_static_x = vertex_static_2x - vertex_static_1x
        direction_static_y = vertex_static_2y - vertex_static_1y

        if axis == 0:
            normal_static_x = -direction_static_y
            normal_static_y = direction_static_x
        else:
            normal_static_x = direction_static_y
            normal_static_y = -direction_static_x

        # plt.plot(
        #     [vertex_static_1x, vertex_static_2x],
        #     [vertex_static_1y, vertex_static_2y],
        #     color="lime"
        # )

        t, u = _two_line_segment_intersection_parameters(
            x1=vertex_static_1x, y1=vertex_static_1y,
            x2=vertex_static_2x, y2=vertex_static_2y,
            x3=point_sweep_1x, y3=point_sweep_1y,
            x4=point_sweep_2x, y4=point_sweep_2y,
        )

        # print(t, u)

        if u > u_min:
            # print("u greater than u_min, continuing")
            continue

        if (0 - epsilon) < u < (1 + epsilon):
            if (0 - epsilon) < t < (1 + epsilon):

                # print("intersection found")

                # if intersection_found:
                #     continue

                m_test = m + normal_m
                n_test = n + normal_n

                if -epsilon < t < epsilon:
                    m_test = m_test - direction_m
                    n_test = n_test - direction_n
                elif (1 - epsilon) < t < (1 + epsilon):
                    m_test = m_test + direction_m
                    n_test = n_test + direction_n

                if (m_test == m_old) and (n_test == n_old):
                    continue

                if -epsilon < u < epsilon:
                    projection_x = (point_sweep_2x - point_sweep_1x) * normal_static_x
                    projection_y = (point_sweep_2y - point_sweep_1y) * normal_static_y
                    projection = projection_x + projection_y
                    # print("projection", projection)
                    if projection < 0:
                        continue
                    else:
                        j_new = j
                elif (1 - epsilon) < u < (1 + epsilon):
                    pass
                else:
                    j_new = j

                u_min = u

                m_new = m_test
                n_new = n_test

                # intersection_found = True

                # plt.scatter(point_sweep_2x, point_sweep_2y)

        elif math.isnan(u) or math.isnan(t):

            # print("parallel lines found")

            if point_sweep_1x != point_sweep_2x:
                u1 = (vertex_static_1x - point_sweep_1x) / (point_sweep_2x - point_sweep_1x)
                u2 = (vertex_static_2x - point_sweep_1x) / (point_sweep_2x - point_sweep_1x)
            else:
                u1 = (vertex_static_1y - point_sweep_1y) / (point_sweep_2y - point_sweep_1y)
                u2 = (vertex_static_2y - point_sweep_1y) / (point_sweep_2y - point_sweep_1y)

            # print("u1", u1)
            # print("u2", u2)

            if u1 > u_min:
                continue

            if u1 > u2:
                u1, u2 = u2, u1
                direction_sweep_m = -direction_m
                direction_sweep_n = -direction_n
            else:
                direction_sweep_m = direction_m
                direction_sweep_n = direction_n

            if -epsilon < u1 < epsilon:
                # print("boop")
                if (0 - epsilon) < u2 < (1 + epsilon):
                    m_new = m + direction_sweep_m
                    n_new = n + direction_sweep_n
                    if (1 - epsilon) < u2 < (1 + epsilon):
                        pass
                    else:
                        j_new = j
                        u_min = u2
                        # plt.scatter(point_sweep_2x, point_sweep_2y, zorder=10)

                m_left = m
                m_right = m + normal_m

                n_left = n
                n_right = n + normal_n

                # plt.plot(
                #     [point_sweep_1x, point_sweep_1x + normal_sweep_x],
                #     [point_sweep_1y, point_sweep_1y + normal_sweep_y],
                #     color="purple"
                # )
                # plt.plot(
                #     [vertex_static_1x, vertex_static_1x + normal_static_x],
                #     [vertex_static_1x, vertex_static_1x + normal_static_y],
                #     color="orange"
                # )

                if normal_sweep_x * normal_static_x + normal_sweep_y * normal_static_y < 0:
                    # print("switch")
                    m_left, m_right = m_right, m_left
                    n_left, n_right = n_right, n_left

                # break

            elif epsilon <= u1 <= (1 + epsilon):

                m_new = m + direction_sweep_m
                n_new = n + direction_sweep_n

                if (m_new == m_old) and (n_new == n_old):
                    continue

                if (1 - epsilon) < u1 < (1 + epsilon):
                    pass
                else:
                    j_new = j
                    u_min = u1

                # break

    if math.isfinite(u_min):
        point_sweep_2x = point_sweep_1x + u_min * (point_sweep_2x - point_sweep_1x)
        point_sweep_2y = point_sweep_1y + u_min * (point_sweep_2y - point_sweep_1y)

    i_left, i_right = i - 1, i

    # print("i_left", i_left)
    # print("i_right", i_right)
    # print("m_left", m_left)
    # print("m_right", m_right)
    # print("n_left", n_left)
    # print("n_right", n_right)

    if input_is_sweep:
        shape_input_x = shape_sweep_x
        shape_input_y = shape_sweep_y
        shape_output_x = shape_static_x
        shape_output_y = shape_static_y

        i_input_left = i_left
        i_input_right = i_right
        j_input_left = j
        j_input_right = j

        i_output_left = m_left
        i_output_right = m_right
        j_output_left = n_left
        j_output_right = n_right

    else:
        shape_input_x = shape_static_x
        shape_input_y = shape_static_y
        shape_output_x = shape_sweep_x
        shape_output_y = shape_sweep_y

        i_input_left = m_left
        i_input_right = m_right
        j_input_left = n_left
        j_input_right = n_right

        i_output_left = i_left
        i_output_right = i_right
        j_output_left = j
        j_output_right = j

    i_input_right = int(i_input_right)
    i_output_right = int(i_output_right)

    if axis == 0:
        if input_is_sweep:
            shape_input_x, shape_input_y = shape_input_y, shape_input_x
            i_input_left, j_input_left = j_input_left, i_input_left
            i_input_right, j_input_right = j_input_right, i_input_right
        else:
            shape_output_x, shape_output_y = shape_output_y, shape_output_x
            i_output_left, j_output_left = j_output_left, i_output_left
            i_output_right, j_output_right = j_output_right, i_output_right

    area_sweep = (point_sweep_1x * point_sweep_2y - point_sweep_2x * point_sweep_1y) / 2

    if 0 <= i_input_left < (shape_input_x - 1) and 0 <= j_input_left < (shape_input_y - 1):
        if 0 <= i_output_left < (shape_output_x - 1) and 0 <= j_output_left < (shape_output_y - 1):
            weight.append((
                (shape_input_y - 1) * i_input_left + j_input_left,
                (shape_output_y - 1) * i_output_left + j_output_left,
                area_sweep / area_input[i_input_left, j_input_left],
            ))

    if 0 <= i_input_right < (shape_input_x - 1) and 0 <= j_input_right < (shape_input_y - 1):
        if 0 <= i_output_right < (shape_output_x - 1) and 0 <= j_output_right < (shape_output_y - 1):
            weight.append((
                (shape_input_y - 1) * i_input_right + j_input_right,
                (shape_output_y - 1) * i_output_right + j_output_right,
                -area_sweep / area_input[i_input_right, j_input_right],
            ))

    # if 0 <= i_input_left < (shape_input_x - 1) and 0 <= j_input_left < (shape_input_y - 1):
    #     value_input_left = values_input[i_input_left, j_input_left]
    # else:
    #     value_input_left = 0
    #
    # if 0 <= i_input_right < (shape_input_x - 1) and 0 <= j_input_right < (shape_input_y - 1):
    #     value_input_right = values_input[int(i_input_right), j_input_right]
    # else:
    #     value_input_right = 0
    #
    # if 0 <= i_output_left < (shape_output_x - 1) and 0 <= j_output_left < (shape_output_y - 1):
    #     values_output[i_output_left, j_output_left] += value_input_left * area_sweep / 2
    #
    # if 0 <= i_output_right < (shape_output_x - 1) and 0 <= j_output_right < (shape_output_y - 1):
    #     values_output[int(i_output_right), j_output_right] -= value_input_right * area_sweep / 2

    # if not input_is_sweep:
    #     area_output[i_output_left, j_output_left] += area_sweep
    #     area_output[int(i_output_right), j_output_right] -= area_sweep

    # print("point_sweep_1", (point_sweep_1x, point_sweep_1y))
    # print("point_sweep_2", (point_sweep_2x, point_sweep_2y))
    # plt.scatter(point_sweep_1x, point_sweep_1y, zorder=10)
    # plt.scatter(point_sweep_2x, point_sweep_2y, zorder=10)

    return point_sweep_2x, point_sweep_2y, j_new, m_new, n_new


@numba.njit(inline="always")
def _grid_area_sweep(
        out: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        axis: int,
):

    if axis == 0:
        grid_y, grid_x = grid_x.T, grid_y.T
        out = out.T
    elif axis == 1:
        pass
    else:
        print("axis should be zero or one")
    #     raise ValueError(f"The `axis` argument must be 0 or 1, got {axis}")

    shape_output_x, shape_output_y = grid_x.shape

    for i in numba.prange(shape_output_x):

        for j in range(shape_output_y - 1):

            index_vertex_1j = j
            index_vertex_2j = j + 1

            vertex_1x = grid_x[i, index_vertex_1j]
            vertex_1y = grid_y[i, index_vertex_1j]

            vertex_2x = grid_x[i, index_vertex_2j]
            vertex_2y = grid_y[i, index_vertex_2j]

            area_sweep = (vertex_1x * vertex_2y - vertex_2x * vertex_1y) / 2
            # if axis == 0:
            #     area_sweep = -area_sweep

            i_left = i - 1
            i_right = i

            if i_left >= 0:
                out[i_left, j] += area_sweep
            if i_right < (shape_output_x - 1):
                out[i_right, j] -= area_sweep


@numba.njit(inline="always", error_model="numpy")
def _indices_of_point_brute(
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        point_x: float,
        point_y: float,
) -> tuple[int, int]:

    shape_x, shape_y = grid_x.shape

    vertices_x = np.empty(4)
    vertices_y = np.empty(4)

    for i in range(shape_x - 1):
        for j in range(shape_y - 1):

            v1 = i + 0, j + 0
            v2 = i + 1, j + 0
            v3 = i + 1, j + 1
            v4 = i + 0, j + 1

            vertices_x[0] = grid_x[v1]
            vertices_x[1] = grid_x[v2]
            vertices_x[2] = grid_x[v3]
            vertices_x[3] = grid_x[v4]

            vertices_y[0] = grid_y[v1]
            vertices_y[1] = grid_y[v2]
            vertices_y[2] = grid_y[v3]
            vertices_y[3] = grid_y[v4]

            if _point_is_inside_polygon(
                vertices_x=vertices_x,
                vertices_y=vertices_y,
                point_x=point_x,
                point_y=point_y,
            ):
                return i, j

    return 0, 0


@numba.njit(inline="always", error_model="numpy")
def _point_is_inside_polygon(
        vertices_x: np.ndarray,
        vertices_y: np.ndarray,
        point_x: float,
        point_y: float,
) -> bool:

    x3 = point_x
    y3 = point_y
    x4 = x3 + 1
    y4 = y3

    result = False
    for v in range(len(vertices_x)):
        x1 = vertices_x[v - 1]
        y1 = vertices_y[v - 1]
        x2 = vertices_x[v]
        y2 = vertices_y[v]

        t, u = _two_line_segment_intersection_parameters(
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
        )
        # print("t", t)
        # print("u", u)

        if 0 <= t < 1:
            if u == 0:
                return True
            elif u > 0:
                result = not result

        # if u == 0:
        #     print("point on edge")
        #     return True
        # elif u > 0:
        #     if 0 <= t <= 1:
        #         print("here")
        #         result = not result


        # slope = (y0 - y1) / (x0 - x1)
        # condition_1 = (y1 >= point_y) != (y0 >= point_y)
        # condition_2 = point_x < ((point_y - y1) / slope + x1)
        # result = result ^ (condition_1 & condition_2)

    return result


@numba.njit(inline="always", error_model="numpy")
def _two_line_segment_intersection(
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
        x4: float, y4: float,
) -> tuple[float, float]:

    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t_denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    u_denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    t = t_numerator / t_denominator
    u = u_numerator / u_denominator

    if (0 < t <= 1) and (0 < u <= 1):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return x, y

    else:
        return np.nan, np.nan


@numba.njit(inline="always", error_model="numpy")
def _two_line_segment_intersection_parameters(
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
        x4: float, y4: float,
) -> tuple[float, float]:

    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t_denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    u_denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # print("t_numerator", t_numerator)
    # print("t_denominator", t_denominator)
    # print("u_numerator", u_numerator)
    # print("u_denominator", u_denominator)

    t = t_numerator / t_denominator
    u = u_numerator / u_denominator

    return t, u

from comp_math import AutoDiffNum

import numpy as np
import matplotlib.pyplot as plt


def progonka(mat: np.matrix, vals: np.array):
    n = len(vals)

    a = [-mat[0, 1] / mat[0, 0]]
    b = [vals[0] / mat[0, 0]]

    for i in range(1, n):
        y = mat[i, i] + mat[i, i - 1] * a[-1]

        if i < n - 1:
            a.append(- mat[i, i + 1] / y)
        b.append((vals[i] - mat[i, i - 1] * b[-1]) / y)

    res = [b[-1]]
    for i in range(n - 2, -1, -1):
        res.append(a[i] * res[-1] + b[i])

    return np.array(res)[::-1]


def get_coeff_splain_3(x_nodes, y_nodes):
    n = len(x_nodes) - 1

    a = y_nodes

    h = [x_nodes[i + 1] - x_nodes[i] for i in range(n)]

    mat = np.matrix(
        [([1.] + [0.] * n)] +

        [[0.] * (i - 1) +
         [h[i - 1]] + [2 * (h[i] + h[i - 1])] + [h[i]] +
         [0.] * (n - i - 1) for i in range(1, n)] +

        [([0.] * n + [1.])]
    )

    vals = np.array([0.] +
                    [3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]) for i in
                     range(1, n)] +
                    [0.])

    c = progonka(mat, vals)

    b = []
    for i in range(n):
        b.append(1 / h[i] * (a[i + 1] - a[i]) - h[i] / 3 * (c[i + 1] + 2 * c[i]))

    d = []
    for i in range(n):
        d.append((c[i + 1] - c[i]) / (3 * h[i]))

    coeff_mat = np.matrix([[a[i], b[i], c[i], d[i]] for i in range(n)])
    return coeff_mat


def get_interpolant_points(x_points, x_nodes, coeff_mat):
    n = len(x_nodes) - 1

    a = np.array(coeff_mat[:, 0])
    b = np.array(coeff_mat[:, 1])
    c = np.array(coeff_mat[:, 2])
    d = np.array(coeff_mat[:, 3])

    y_points = []
    for x in x_points:
        for i in range(n):
            if x_nodes[i] <= x <= x_nodes[i + 1]:
                y_points.append(
                    (a[i] + b[i] * (x - x_nodes[i]) + c[i] * (x - x_nodes[i]) ** 2 + d[i] * (x - x_nodes[i]) ** 3)[0])
                break

    return y_points


def get_points_from_file(filename: str):
    f = open(filename, 'r')
    points = list(map(lambda x: tuple(map(float, x.split(' '))), f.readlines()))
    f.close()

    return points


def put_coeffs_to_file(filename: str, flag: str, coeff_mat_x, coeff_mat_y):
    f = open(filename, flag)

    for coeffs in zip(coeff_mat_x.tolist(), coeff_mat_y.tolist()):
        coeffs_1_2 = list(coeffs[0]) + list(coeffs[1])
        f.write(str(coeffs_1_2).strip("[]") + '\n')

    f.close()


def lab1_base(filename_in: str, factor: int, filename_out: str):
    ## 2
    points = np.loadtxt(filename_in)

    x_points = points[:, 0]
    y_points = points[:, 1]

    plt.subplots(1, 1, figsize=(12, 6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_points, y_points, 'b-', linewidth=2)
    plt.grid()
    plt.show()

    ## 3
    x_nodes = x_points[::factor]
    y_nodes = y_points[::factor]

    n = len(x_nodes)

    plt.subplots(1, 1, figsize=(12, 6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_nodes, y_nodes, 'ro', markersize=5)
    plt.grid()
    plt.show()

    ##4
    t = np.arange(0, n, 1)
    coeff_mat_x = get_coeff_splain_3(t, x_nodes)
    coeff_mat_y = get_coeff_splain_3(t, y_nodes)

    ##6
    t_for_plotting = np.arange(0, n - 1 + 1 / factor, 1 / factor)
    x_ip_points = get_interpolant_points(t_for_plotting, t, coeff_mat_x)
    y_ip_points = get_interpolant_points(t_for_plotting, t, coeff_mat_y)

    plt.subplots(1, 1, figsize=(12, 6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_points, y_points, 'b-', linewidth=4, label="Исходный контур P")
    plt.plot(x_ip_points, y_ip_points, 'g-', linewidth=2, label="Сплайн $\\tilde{P}$")
    plt.plot(x_nodes, y_nodes, 'ro', markersize=4, label="Интерполяционные узлы $\hat{P}$")
    plt.legend()
    plt.grid()
    plt.show()

    ##5
    error_rate = [np.sqrt((x_ip_points[i] - x_points[i]) ** 2 + (y_ip_points[i] - y_points[i]) ** 2) for i in
                  range(len(x_ip_points))]

    print(np.mean(error_rate))
    print(np.std(error_rate))

    ##7
    put_coeffs_to_file(filename_out, "w+", coeff_mat_x, coeff_mat_y)

    ##8-11
    plt.subplots(1, 1, figsize=(12, 6))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_ip_points, y_ip_points, 'g-', linewidth=2, zorder=1)

    x_ampl = max(x_ip_points) - min(x_ip_points)
    y_ampl = max(y_ip_points) - min(y_ip_points)
    derivative_vector_length = min(x_ampl, y_ampl) / 10

    t_for_derivative = np.linspace(0, n - 1, 10)
    x_d_points = AutoDiffNum.G(t_for_derivative, t, coeff_mat_x)
    y_d_points = AutoDiffNum.G(t_for_derivative, t, coeff_mat_y)

    d_points = []
    d_vectors = []
    n_vectors = []
    for i in range(len(t_for_derivative)):
        x_point = get_interpolant_points([t_for_derivative[i]], t, coeff_mat_x)
        y_point = get_interpolant_points([t_for_derivative[i]], t, coeff_mat_y)

        start_point = [x_point[0], y_point[0]]
        d_points.append(start_point)

        vector_length = np.sqrt(x_d_points[i] ** 2 + y_d_points[i] ** 2)
        d_vector = [x_d_points[i], y_d_points[i]] / vector_length * derivative_vector_length
        d_vectors.append(d_vector)

        n_vector = AutoDiffNum.R(d_vector)
        n_vectors.append(n_vector)

    d_points = np.array(d_points)
    d_vectors = np.array(d_vectors)
    n_vectors = np.array(n_vectors)

    plt.quiver(d_points[:, 0], d_points[:, 1], d_vectors[:, 0], d_vectors[:, 1], color='r', width=0.005,
               label="Направления производных G(t)", zorder=2)
    plt.quiver(d_points[:, 0], d_points[:, 1], n_vectors[:, 0], n_vectors[:, 1], color='b', width=0.005,
               label="Направления нормалей R(t)", zorder=2)
    plt.plot(d_points[:, 0], d_points[:, 1], 'bo', markersize=10, zorder=3)

    plt.legend()
    plt.grid()
    plt.show()


lab1_base('contour.txt', 10, 'coeffs.txt')

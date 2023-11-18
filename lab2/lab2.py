import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Константы
start_t = np.float64(0)
end_t = np.float64(1.75418438)
C = np.float64(1.03439984)
g = np.float64(9.8)


def composite_simpson(a, b, n, func):
    if n % 2 == 0:
        n += 1

    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)

    return h / 3 * (func(a) +
                    2 * np.sum([func(x_i) for x_i in x[2:-1:2]]) +
                    4 * np.sum([func(x_i) for x_i in x[1:-1:2]]) +
                    func(x[-1]))


def composite_trapezoid(a, b, n, func):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)

    return h / 2 * (func(x[0]) +
                    2 * np.sum([func(x_i) for x_i in x[1:-1]]) +
                    func(x[-1]))


# Аналитическое решение интеграла
def analytics_integral():
    return np.sqrt(2 * C / g) * end_t


# Расчет и вывод абсолютной погрешности и порядка точности
def absolute_err(a, b, max_n, func):
    n = [i for i in range(3, max_n)]
    h = (b - a) / n

    err_simpson = np.abs(np.array([composite_simpson(a, b, n_i, func) for n_i in n]) - analytics_integral())
    err_trapezoid = np.abs(np.array([composite_trapezoid(a, b, n_i, func) for n_i in n]) - analytics_integral())

    plt.subplots(1, 1, figsize=(10, 6))

    plt.loglog(h, err_simpson, 'ro', markersize=2)
    plt.loglog(h, err_trapezoid, 'bo', markersize=2)
    plt.loglog(h, h, 'k--', linewidth=2)
    plt.loglog(h, h ** 2 / 100, 'k--', linewidth=2)
    plt.loglog(h, h ** 4 / 100, 'k--', linewidth=2)

    plt.grid()
    plt.show()


# Common
f_x = lambda t1: C * t1 - C * 1 / 2 * np.sin(2 * t1)
f_y = lambda t3: C * 1 / 2 - C * 1 / 2 * np.cos(2 * t3)

f_dx_dt = lambda t2: C * (1 - np.cos(2 * t2))
f_dy_dt = lambda t4: C * np.sin(2 * t4)

f_dy_dx = lambda t5: f_dy_dt(t5) / f_dx_dt(t5)

f = lambda t6: (0 if t6 == 0 else
                np.sqrt((1. + f_dy_dx(t6) ** 2) / (2. * g * f_y(t6))) * f_dx_dt(t6))

# 1
# print(composite_simpson(start_t, end_t, 9999, f))
# print(composite_trapezoid(start_t, end_t, 9999, f))
# print(analytics_integral())
#
# absolute_err(start_t, end_t, 999, f)


# 2

start_x = np.float64(0)
end_x = np.float64(2)

t_nodes = np.linspace(start_t, end_t, 19)

print([f_x(t_i) for t_i in t_nodes])
plt.plot([f_x(t_i) for t_i in t_nodes], [f_y(t_i) for t_i in t_nodes], 'o-')
plt.show()

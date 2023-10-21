import numpy as np


class AutoDiffNum:
    def __init__(self, a=0.0, b=1.0):
        self.a = np.float64(a)
        self.b = np.float64(b)

    def __add__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a + other.a, self.b + other.b)
        else:
            return AutoDiffNum(self.a + other, self.b)

    def __radd__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(other.a + self.a, other.b + self.b)
        else:
            return AutoDiffNum(other + self.a, self.b)

    def __sub__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a - other.a, self.b - other.b)
        else:
            return AutoDiffNum(self.a - other, self.b)

    def __rsub__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(other.a - self.a, other.b - self.b)
        else:
            return AutoDiffNum(other - self.a, self.b)

    def __mul__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a * other.a, self.b * other.a + self.a * other.b)
        else:
            return AutoDiffNum(self.a * other, self.b * other)

    def __rmul__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.a * other.a, self.b * other.a + self.a * other.b)
        else:
            return AutoDiffNum(self.a * other, self.b * other)

    def __pow__(self, other):
        return AutoDiffNum(self.a ** 2, other * self.b * (self.a ** (other - 1)))

    def __str__(self):
        return f"{self.a} + {self.b}e"

    @staticmethod
    def G(x_points, x_nodes, coeff_mat):
        n = len(x_nodes) - 1

        a = coeff_mat[:, 0]
        b = coeff_mat[:, 1]
        c = coeff_mat[:, 2]
        d = coeff_mat[:, 3]

        y_points = []
        for x in x_points:
            for i in range(n):
                if x_nodes[i] <= x <= x_nodes[i + 1]:
                    dual_number = a[i] + b[i] * (AutoDiffNum(x) - x_nodes[i]) + c[i] * (
                            AutoDiffNum(x) - x_nodes[i]) ** 2 + d[i] * (AutoDiffNum(x) - x_nodes[i]) ** 3
                    y_points.append(dual_number[0, 0].b)
                    break

        return y_points

    @staticmethod
    def R(vector):
        return [-vector[1], vector[0]]

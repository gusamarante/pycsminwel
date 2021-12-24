import numpy as np
from pycsminwel import csminwel


# Test function
def rosenbrock(x):
    """
    Rosenbrock Function
    :param x: vector of legth 2
    :return: float
    """
    return (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 4


def drosenbrock(x):
    """
    Gradient of the Rosenbrock Function
    :param x: vector of legth 2
    :return: gradient vector
    """
    dr = np.zeros(2)
    dr[0] = 2 * (x[0] - 1) - 8 * 105 * x[0] * (x[1] - x[0] ** 2) ** 3
    return dr


# Run the optimization
x0 = np.array([10, -9])
h0 = np.eye(2) * 0.5
res, bad_grad = csminwel(rosenbrock, x0, h0)
print(res)

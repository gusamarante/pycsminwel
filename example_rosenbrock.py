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
    dr[1] = 4 * 105 * (x[1] - x[0] ** 2) ** 3
    return dr


print(' ===== Optimization with NUMERCIAL derivatives =====')
x0 = np.array([10, -9])
h0 = np.eye(2) * 0.5
fh, xh, _, _, itct, _, _ = csminwel(rosenbrock, x0, h0)
print(f'Convergence in {itct} steps')
print(f'Minimal value is {round(fh, 4)}')
print('Optimal point is', xh)
print('\n')

print(' ===== Optimization with ANALYTICAL derivatives =====')
x0 = np.array([10, -9])
h0 = np.eye(2) * 0.5
fh, xh, _, _, itct, _, _ = csminwel(rosenbrock, x0, h0, drosenbrock)
print(f'Convergence in {itct} steps')
print(f'Minimal value is {round(fh, 4)}')
print('Optimal point is', xh)
print('\n')

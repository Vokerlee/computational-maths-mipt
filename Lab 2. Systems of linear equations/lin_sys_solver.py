import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics

####################################################################################

# GENERAL FUNCTIONS

class LinearEqSystem:
    def __init__(self, matrix, f):
        assert np.shape(matrix)[0] == np.shape(matrix)[1]
        assert np.shape(matrix)[0] == np.shape(f)[0]

        self.matrix    = matrix
        self.f         = f

        self.dimension = np.shape(f)[0]
        self.solution  = np.zeros(self.dimension)
        self.is_solved = False

    def solve_eq(self, solution_method):
        self.solution = solution_method(self)
        self.is_solved = True

def minors_degenerated(matrix):
    assert np.shape(matrix)[0] == np.shape(matrix)[1]
    n = np.shape(matrix)[0]

    for i in range(1, n + 1):
        if np.linalg.det(matrix[np.ix_([j for j in range(i)], [j for j in range(i)])]) == 0:
            return True

    return False

def get_eigenvalue_max(matrix):
    epsilon = 1e-2
    lambda_matrix_old = 1
    lambda_matrix_new = -1

    y = np.ones(np.shape(matrix)[0])

    while np.absolute(lambda_matrix_new - lambda_matrix_old) > epsilon:
        y_new = np.dot(matrix, y)

        lambda_matrix_old = lambda_matrix_new
        lambda_matrix_new = np.linalg.norm(y_new, ord=2) / np.linalg.norm(y, ord=2)

        y = y_new

    return lambda_matrix_new

def get_eigenvalue_min(matrix):
    return get_eigenvalue_max(np.linalg.inv(matrix))

def get_LU_decomposition(matrix):
    assert np.shape(matrix)[0] == np.shape(matrix)[1]
    n = np.shape(matrix)[0]

    L = np.full((n, n), 0.)
    U = np.full((n, n), 0.)

    for i in range(n):
        L[i][i] = 1

        if i != 0:
            for j in range(i, n):
                U[i][j] = matrix[i][j]
                for k in range(0, i):
                    U[i][j] -= L[i][k] * U[k][j]

            for j in range(i + 1, n):
                L[j][i] = matrix[j][i]
                for k in range(0, i):
                    L[j][i] -= L[j][k] * U[k][i]
                L[j][i] /= U[i][i]
        else:
            for j in range(0, n):
                U[0, j] = matrix[0, j]
                L[j, 0] = matrix[j, 0] / matrix[0, 0]

    return L, U

def method_LU_decomposition(linear_system):
    L, U = get_LU_decomposition(linear_system.matrix)
    n = linear_system.dimension

    y = np.zeros(n)
    for i in range(n):
        y[i] = linear_system.f[i]
        for j in range(i):
            y[i] -= y[j] * L[i, j]

    x = np.zeros(n)
    for i in range(n):
        x[n - 1 - i] = y[n - 1 - i]
        for j in range(1, i + 1):
            x[n - 1 - i] -= x[n - 1 - i + j] * U[n - 1 - i, n - 1 - i + j]

        x[n - 1 - i] /= U[n - 1 - i, n - 1 - i]

    return x

def method_upper_relaxation(linear_system):
    w = 1.2
    epsilon = 1e-15

    L = linear_system.matrix - np.triu(linear_system.matrix)
    U = linear_system.matrix - np.tril(linear_system.matrix)
    D = linear_system.matrix - L - U

    x = np.zeros(linear_system.dimension)
    n_iters = 0

    iter_matrix = - np.dot(np.linalg.inv(D + w * L), (w - 1) * D + w * U)
    iter_const  = w * np.dot(np.linalg.inv(D + w * L), linear_system.f)

    while np.linalg.norm(np.dot(linear_system.matrix, x) - linear_system.f, ord=2) > epsilon:
        n_iters += 1
        x = np.dot(iter_matrix, x) + iter_const

    print("\nAmount of iterations:", n_iters, "\n")

    return x

####################################################################################

# LAB SPECIFIC FUNCTIONS

def get_system_params1(n):
    n = n - 1

    matrix = np.full((n + 1, n + 1), np.float64(0))
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                matrix[i, j] = 11 + i
            if i + 1 == j:
                matrix[i, j] = 1
            if i - 1 == j:
                matrix[i, j] = 1
    
    matrix[n, 0] = 1
    matrix[n, n] = 1

    for i in range(1, n):
        matrix[n, i] = 2

    f = np.array([np.float64(i) / n for i in range(1, n + 2)])

    return matrix, f

####################################################################################

matrix, f = get_system_params1(100)
print("The number of conditionality:", np.linalg.norm(matrix, ord=2) * np.linalg.norm(np.linalg.inv(matrix), ord=2))

print("\nLambda maximum:", get_eigenvalue_max(matrix))
print("Lambda minimum:", get_eigenvalue_min(matrix))
print("Stop criteria for power method: epsilon = 1e-2")

linear_system1 = LinearEqSystem(matrix, f)
linear_system2 = LinearEqSystem(matrix, f)

if minors_degenerated(linear_system1.matrix) == False:
    print("\nMatrix of linear system can be LU decomposed!")
else:
    print("\nMatrix of linear system cannot(!) be LU decomposed!")

print("\n1) Solution through LU decomposition method:")
linear_system1.solve_eq(method_LU_decomposition)
print(linear_system1.solution)

print("\n2) Solution through upper relaxation method:")
linear_system2.solve_eq(method_upper_relaxation)
print(linear_system2.solution)

print("\nStop criteria for upper relaxation method: epsilon = 1e-15")
print("Printing of residuals:", np.linalg.norm(np.dot(linear_system2.matrix, linear_system2.solution) - linear_system2.f, ord=2))

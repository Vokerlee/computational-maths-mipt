import statistics
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

####################################################################################

# GENERAL FUNCTIONS

def solve_diff_eq(h_step, t_start, t_end, x0_vec, diff_function, method):
    assert(t_start <= t_end)
    assert(h_step > 0)

    n_steps = int((t_end - t_start) / h_step) + 1
    t = np.linspace(t_start, t_end, n_steps)

    x_vec = np.zeros((n_steps, np.shape(x0_vec)[0]))
    for i in range(np.shape(x0_vec)[0]):
        x_vec[0][i] = x0_vec[i]

    return method(h_step, t, x_vec, diff_function)

def method_Euler_1st_order(h_step, t, x_vec, diff_function):
    n_steps = np.shape(x_vec)[0]

    for j in range(1, np.shape(t)[0]):
        k1_vec = diff_function(t[j - 1], x_vec[j - 1])
        x_vec[j] = x_vec[j - 1] + h_step * k1_vec

    return t, x_vec

def method_classic_RK_4th_order(h_step, t, x_vec, diff_function):
    n_steps = np.shape(x_vec)[0]

    for j in range(1, np.shape(t)[0]):
        k1_vec = diff_function(t[j - 1], x_vec[j - 1])
        k2_vec = diff_function(t[j - 1] + 1 / 2 * h_step, x_vec[j - 1] + 1 / 2 * h_step * k1_vec)
        k3_vec = diff_function(t[j - 1] + 1 / 2 * h_step, x_vec[j - 1] + 1/2 * h_step * k2_vec)
        k4_vec = diff_function(t[j - 1] + 1 * h_step, x_vec[j - 1] + 1 * h_step * k3_vec)

        x_vec[j] = x_vec[j - 1] + \
            h_step * (1 / 6 * k1_vec + 2 / 6 * k2_vec + 2 / 6 * k3_vec + 1 / 6 * k4_vec)

    return t, x_vec

def method_Adams_3d_order(h_step, t, x_vec, diff_function):
    n_steps = np.shape(x_vec)[0]

    f_vec = np.zeros((n_steps, np.shape(x_vec[0])[0]))

    f_vec[0] = diff_function(t[0], x_vec[0])
    x_vec[1] = x_vec[0] + h_step * f_vec[0]

    f_vec[1] = diff_function(t[1], x_vec[1])
    x_vec[2] = x_vec[1] + h_step * f_vec[1]

    f_vec[2] = diff_function(t[2], x_vec[2])

    for j in range(3, np.shape(t)[0]):
        x_vec[j] = x_vec[j - 1] + \
            h_step * (23 / 12 * f_vec[j - 1] - 16 / 12 * f_vec[j - 2] + 5 / 12 * f_vec[j - 2])
        f_vec[j] = diff_function(t[j], x_vec[j])

    return t, x_vec

####################################################################################

# LAB SPECIFIC STUFF

def get_function_lab1(param):
    return lambda t, x_vec: \
        np.array([x_vec[1], param * (1 - x_vec[0] ** 2) * x_vec[1] - x_vec[0]])

####################################################################################

# SCRIPT START

func_lab = get_function_lab1(1)

step = 0.01

t_step, solution_vec1 = \
    solve_diff_eq(step, 0, 100, np.array([2, 0]), func_lab, method_Euler_1st_order)
t_step, solution_vec2 = \
    solve_diff_eq(step, 0, 100, np.array([2, 0]), func_lab, method_classic_RK_4th_order)
t_step, solution_vec3 = \
    solve_diff_eq(step, 0, 100, np.array([2, 0]), func_lab, method_Adams_3d_order)

x1_version1 = np.array([solution_vec1[i][0] for i in range(np.shape(solution_vec1)[0])])
x2_version1 = np.array([solution_vec1[i][1] for i in range(np.shape(solution_vec1)[0])])

x1_version2 = np.array([solution_vec2[i][0] for i in range(np.shape(solution_vec2)[0])])
x2_version2 = np.array([solution_vec2[i][1] for i in range(np.shape(solution_vec2)[0])])

x1_version3 = np.array([solution_vec3[i][0] for i in range(np.shape(solution_vec3)[0])])
x2_version3 = np.array([solution_vec3[i][1] for i in range(np.shape(solution_vec3)[0])])

axes = plt.figure().add_subplot(projection='3d')

axes.plot(x1_version1, x2_version1, t_step, c='g', label='Euler method (1st order)')
axes.plot(x1_version2, x2_version2, t_step, c='r', label='Classic RK method (4th order)')
axes.plot(x1_version3, x2_version3, t_step, c='b', label='Adams method (3d order)')

axes.legend()
axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_zlabel('t')

axes.set_title('Step ' + str(step))

axes.view_init(elev=20., azim=-35, roll=0)

plt.show()

# SCRIPT END

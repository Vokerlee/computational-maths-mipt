import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker   import MaxNLocator

####################################################################################

matplotlib.rcParams['axes.formatter.limits'] = (-5, 4)

MAX_LOCATOR_NUMBER = 10
FIGURE_XSIZE = 10
FIGURE_YSIZE = 6

BACKGROUND_RGB = '#F5F5F5'
MAJOR_GRID_RGB = '#919191'

LEGEND_FRAME_ALPHA = 0.95

def set_axis_properties(axes):
    axes.xaxis.set_major_locator(MaxNLocator(MAX_LOCATOR_NUMBER))
    axes.minorticks_on()
    axes.grid(which='major', linewidth=2, color=MAJOR_GRID_RGB)
    axes.grid(which='minor', linestyle=':')

####################################################################################

# LAB SPECIFIC STUFF

h = 1e-5
x_0 = 1 / np.sqrt(2) # break poin
x_start = 0
x_end = 1

def k(x):
    return x ** 2 + 0.5

def q(x):
    if x < x_0:
        return 1
    else:
        return np.exp(-x ** 2)

def f(x):
    if x < x_0:
        return 1
    else:
        return np.cos(x)
    
def get_system_coeffs(h_step, N, N_break_left):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    N_break_right = N_break_left + 1

    for n in range(1, N_break_left):
        a[n] = k((n + 0.5) * h_step)
        b[n] = -(k((n + 0.5) * h_step) + k((n - 0.5) * h_step) + q(n * h_step) * h_step ** 2)
        c[n] = k((n - 0.5) * h_step)
        d[n] = -f(n * h_step) * h_step * h_step

    for n in range(N_break_right + 1, N):
        a[n] = k((n+ 0.5) * h_step)
        b[n] = -(k((n + 0.5) * h_step) + k((n - 0.5) * h_step) + q(n * h_step) * h_step ** 2)
        c[n] = k((n - 0.5) * h_step)
        d[n] = -f(n * h_step) * h_step ** 2

    return a, b, c, d

def run_through(left_u, right_u, h_step, break_point):
    N = int((x_end - x_start) / h_step)
    x = np.array([x_start + n * h_step for n in range(N + 1)])
    
    N_break_left = int(np.floor((break_point - x_start) / h_step))
    N_break_right = N_break_left + 1

    u = np.zeros(N + 1)
    u[0] = left_u
    u[N] = right_u

    a, b, c, d = get_system_coeffs(h_step, N, N_break_left)

    alpha = np.zeros(N + 1)
    beta  = np.zeros(N + 1)

    alpha[1] = -a[1] / b[1]
    beta[1]  = (d[1] - c[1] * left_u) / b[1]

    alpha[N - 1] = -c[N - 1] / b[N - 1]
    beta[N - 1]  = (d[N - 1] - c[N - 1] * right_u) / b[N - 1]

    for n in range(2, N_break_left):
        alpha[n] = -a[n] / (b[n] + c[n] * alpha[n - 1])
        beta[n]  = (d[n] - c[n] * beta[n - 1]) / (b[n] + c[n] * alpha[n - 1])

    for n in range(N - 2, N_break_left, -1):
        alpha[n] = -c[n] / (b[n] + a[n] * alpha[n + 1])
        beta[n]  = (d[n] - a[n] * beta[n + 1]) / (b[n] + a[n] * alpha[n + 1])

    u[N_break_left] = (k(N_break_left * h_step) * beta[N_break_left - 1] + \
                       k(N_break_right * h_step) * beta[N_break_right + 1]) / \
                      (k(N_break_left * h_step) * (1 - alpha[N_break_left - 1]) + \
                       k(N_break_right * h_step) * (1 - alpha[N_break_right + 1]))
    
    u[N_break_right] = u[N_break_left].copy()

    u[N_break_left - 1] = alpha[N_break_left - 1] * u[N_break_left] + beta[N_break_left - 1]
    u[N_break_right + 1] = alpha[N_break_right + 1] * u[N_break_right] + beta[N_break_right + 1]

    for n in range(N_break_left - 1, 0, -1):
        u[n] = alpha[n] * u[n + 1] + beta[n]

    for n in range(N_break_right + 1, N):
        u[n] = alpha[n] * u[n - 1] + beta[n]

    return x, u

####################################################################################

# SCRIPT START

x, u = run_through(1, 0, h, x_0)

figure = plt.figure(figsize=(FIGURE_XSIZE, FIGURE_YSIZE), facecolor=BACKGROUND_RGB)
gs = GridSpec(ncols=1, nrows=1, figure=figure)
axes = figure.add_subplot(gs[0, 0])
set_axis_properties(axes)

axes.plot(x, u, color='blue', linewidth=2, label="u(x): solution")

axes.set_title('$u(x)$')
axes.set_xlabel(r'$x$', fontsize=18)
axes.set_ylabel(r'$u$', fontsize=18)

axes.plot([x_0, x_0], [0, 1], linewidth=1, color='green', label="break point")

plt.legend()
plt.show()

# SCRIPT END
import statistics
import matplotlib
import math
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

# SCRIPT START

EPSILON = 1e-8
step    = 1e-5

y_start = 1.0 # for even solution
y_deriv_start = 0.5 # for odd solution

y_end   = 0.0

x_start = 0.0
x_end   = 10

found_energies = []

n_steps = int((x_end - x_start) / step) + 1

def U(x):
    return -2 / (np.cosh(x) ** 2)

def shoot_energy(h_step, y_0, y_0_deriv, energy):
    vec_sol = [np.array([y_0_deriv, y_0])]
    vec_prev = vec_sol[0]

    for i in range(n_steps - 1):
        vec = vec_prev + h_step * (np.array([(U(x_start + h_step * i) - energy) * vec_prev[1], vec_prev[0]]))
        vec_sol.append(vec)
        vec_prev = vec

    return np.array(vec_sol)

def newton(h_step, y_0, y_0_deriv, energy):
    vec_sol = shoot_energy(h_step, y_0, y_0_deriv, energy)
    Fprev = y_end - vec_sol[n_steps - 1][1]
    energy_prev = energy

    energy += 10 * EPSILON if Fprev > 0 else -EPSILON * 10
    diff = 2 * EPSILON

    while diff >= EPSILON:
        vec_sol = shoot_energy(h_step, y_0, y_0_deriv, energy)
        F = y_end - vec_sol[n_steps - 1][1]
        dF = (F - Fprev) / (energy - energy_prev)
        energy_prev = energy
        Fprev = F

        diff = F / dF
        energy = energy - diff
        diff = abs(diff)

    return vec_sol, energy

def calc_int_trapeze(step, func_vals):
    assert step >= 0
    return (sum(func_vals) - (func_vals[0] + func_vals[-1]) / 2) * step

def make_plot_even(h_step, y_0, y_0_deriv, energy, label_plot):
    vec_sol, energy_final = newton(h_step, y_0, y_0_deriv, energy)
    if (energy_final > 0):
        return
    if energy_final in found_energies:
        return
    
    found_energies.append(energy_final)

    x = np.linspace(x_start - x_end, x_start + x_end, 2 * n_steps - 1)
    y = np.array([vec_sol[abs(i)][1] for i in range(-n_steps + 1, n_steps, 1)])
    y = y / calc_int_trapeze(h_step, y ** 2)

    label_plot = label_plot + " for energy " + str(energy_final)

    plt.plot(x, y, label=label_plot)

def make_plot_odd(h_step, y_0, y_0_deriv, energy, label_plot):
    vec_sol, energy_final = newton(h_step, y_0, y_0_deriv, energy)
    if (energy_final > 0):
        return
    if energy_final in found_energies:
        return
    
    found_energies.append(energy_final)

    x = np.linspace(x_start - x_end, x_start + x_end, 2 * n_steps - 1)
    y = np.array([vec_sol[i][1] for i in range(n_steps)])
    y = np.array([vec_sol[abs(i)][1] for i in range(-n_steps + 1, n_steps, 1)])
    for i in range(len(y)):
        if i < n_steps:
            y[i] = -y[i]
    y = y / calc_int_trapeze(h_step, y ** 2)

    label_plot = label_plot + " for energy " + str(energy_final)

    plt.plot(x, y, label=label_plot)



figure = plt.figure(figsize=(FIGURE_XSIZE, FIGURE_YSIZE), facecolor=BACKGROUND_RGB)
gs = GridSpec(ncols=1, nrows=1, figure=figure)
axes = figure.add_subplot(gs[0, 0])
set_axis_properties(axes)

energy_vals = np.array([-1.9]) # np.linspace(-2.0, 0.0, 10)

for energy in energy_vals:
    make_plot_even(step, y_start, 0.0, energy, "$y(x)$: solution")
    # make_plot_odd(step, 0.0, y_deriv_start, energy, "$y(x)$: solution")

axes.set_title('Solutions $y(x)$')
axes.set_xlabel(r'$x$', fontsize=18)
axes.set_ylabel(r'$y$', fontsize=18)

plt.legend()
plt.tight_layout()
plt.show()

# SCRIPT END

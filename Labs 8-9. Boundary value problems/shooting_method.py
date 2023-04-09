import statistics
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

####################################################################################

# SCRIPT START

EPSILON = 1e-8
step    = 1e-5
x_start = 0.0
x_end   = 1.0
y_end   = 0.0

n_steps = int((x_end - x_start) / step) + 1

def shoot(h_step, y_0, alpha):
    vec_sol = [np.array([alpha, y_0])]
    vec_prev = vec_sol[0]

    for i in range(n_steps - 1):
        vec = vec_prev + h_step * (np.array([0.5 * vec_prev[0] ** 2 / (1 - 0.5 * vec_prev[1]), vec_prev[0]])) 
        vec_sol.append(vec)
        vec_prev = vec

    return np.array(vec_sol)

def newton(h_step, y_0, alpha):
    vec_sol = shoot(h_step, y_0, alpha)
    Fprev = y_end - vec_sol[n_steps - 1][1]
    aprev = alpha

    alpha += 1 if Fprev > 0 else -1
    diff = 2 * EPSILON

    while diff >= EPSILON:
        vec_sol = shoot(h_step, y_0, alpha)
        F = y_end - vec_sol[n_steps - 1][1]
        dF = (F - Fprev) / (alpha - aprev)
        aprev = alpha
        Fprev = F

        diff = F / dF
        alpha = alpha - diff
        diff = abs(diff)

    return vec_sol, alpha

def make_plot(h_step, y_0, alpha, label_plot):
    vec_sol, alpha_fin = newton(h_step, y_0, alpha)
    x = np.arange(x_start, x_end, h_step)
    y = np.array([vec_sol[i][1] for i in range(n_steps)])

    plt.plot(x, y, label=label_plot)

y_0_values = np.array([0.25, 0.5, 1.0, 1.5, 1.8, 1.9, 1.95])
y_0_labels = np.array(['$y({x_start}) = y_0$ = ' + str(y_0_values[i]) for i in range(len(y_0_values))])
plt.figure(figsize=[16, 9])
plt.title(f"Solutions $y(x)$")
plt.xlabel("$x$", fontsize=15)
plt.ylabel("$y$", fontsize=15)
plt.grid()

for i in range(len(y_0_values)):
    make_plot(step, y_0_values[i], 0.0, y_0_labels[i])

plt.legend()
plt.tight_layout()
plt.show()

# SCRIPT END

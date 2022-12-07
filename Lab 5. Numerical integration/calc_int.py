import statistics
import matplotlib
import math
import numpy  as np

####################################################################################

# GENERAL FUNCTIONS

def calc_int_Simpson(step, func_vals):
    assert step >= 0
    odd_sum  = 0
    even_sum = 0

    for i in range(1, len(func_vals) - 1):
        if i % 2 == 0:
            even_sum += func_vals[i]
        else:
            odd_sum += func_vals[i]

    return (func_vals[0] + 4 * odd_sum + 2 * even_sum + func_vals[-1]) * step / 3

def calc_int_trapeze(step, func_vals):
    assert step >= 0
    return (sum(func_vals) - (func_vals[0] + func_vals[-1]) / 2) * step

def calc_int_Runge(step, func_vals):
    assert step >= 0
    integral_trapeze = calc_int_trapeze(step, func_vals)

    sparse_func_vals = \
        np.array([func_vals[2 * i] for i in range(len(func_vals) // 2 + len(func_vals) % 2)])
    integral_trapeze_sparse = calc_int_trapeze(2 * step, sparse_func_vals)

    return integral_trapeze + (integral_trapeze - integral_trapeze_sparse) / (2 ** 2 - 1)

####################################################################################

# LAB SPECIFIC STUFF

step = 0.125
func_vals = np.array([0, 0.124670, 0.247234, 0.364902, 0.473112, 
                      0.563209, 0.616193, 0.579699, 0])

####################################################################################

# SCRIPT START

print('Simpson formula: ', calc_int_Simpson(step, func_vals))
print('Trapeze formula: ', calc_int_trapeze(step, func_vals))
print('Runge formula (rule): ', calc_int_Runge(step, func_vals))

# SCRIPT END

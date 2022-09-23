import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os
import shutil
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec  import GridSpec
from matplotlib.ticker    import MaxNLocator, ScalarFormatter, FixedFormatter

####################################################################################

matplotlib.rcParams['axes.formatter.limits'] = (-5, 4)

MAX_LOCATOR_NUMBER = 10
FIGURE_XSIZE = 10
FIGURE_YSIZE = 8

BACKGROUND_RGB = '#F5F5F5'
MAJOR_GRID_RGB = '#919191'

LEGEND_FRAME_ALPHA = 0.95

####################################################################################

def set_axis_properties(axes):
    axes.xaxis.set_major_locator(MaxNLocator(MAX_LOCATOR_NUMBER))
    axes.minorticks_on()
    axes.grid(which='major', linewidth=2, color=MAJOR_GRID_RGB)
    axes.grid(which='minor', linestyle=':')

####################################################################################

FUNCTIONS_ARRAY = np.array([
    lambda x: np.sin(x ** 2),
    lambda x: np.cos(np.sin(x)),
    lambda x: np.exp(np.sin(np.cos(x))),
    lambda x: np.log(x + 3.0),
    lambda x: np.sqrt(x + 3.0)])

FUNCTIONS_DERIVATIVES_ARRAY = np.array([
    lambda x: 2 * x * np.cos(x ** 2),
    lambda x: -np.sin(np.sin(x)) * np.cos(x),
    lambda x: np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * (-np.sin(x)),
    lambda x: 1.0 / (x + 3.0),
    lambda x: 0.5 / np.sqrt(x + 3.0)])

DERIVATIVES_ARRAY = np.array([
    lambda func, x, h: (func(x + h) - func(x)) / h,
    lambda func, x, h: (func(x) - func(x - h)) / h,
    lambda func, x, h: (func(x + h) - func(x - h)) / 2 / h,
    lambda func, x, h: (4.0 / 3.0 * (func(x + h) - func(x - h)) / 2 / h - 1.0 / 3.0 * (func(x + 2 * h) - func(x - 2 * h)) / 4 / h),
    lambda func, x, h: (3.0 / 2.0 * (func(x + h) - func(x - h)) / 2 / h - 3.0 / 5.0 * (func(x + 2 * h) - func(x - 2 * h)) / 4 / h + 1.0 / 10.0 * (func(x + 3 * h) - func(x - 3 * h)) / 6 / h)])

FUNCTION_NAMES = np.array(['$sin(x^2)$', '$cos(sin(x))$', '$e^{sin(cos(x))}$', '$ln(x + 3)$', '$(x + 3)^{1/2}$'])
DERIVATIVE_NAMES = np.array(['1', '2', '3', '4', '5'])

def get_function(number):
    return FUNCTIONS_ARRAY[number]

def get_function_deriv(number):
    return FUNCTIONS_DERIVATIVES_ARRAY[number]

def get_deriv(number):
    return DERIVATIVES_ARRAY[number]

def call_function(number, x):
    function = get_function(number)
    return function(x)

def calc_function_deriv(number, x):
    function = get_function_deriv(number)
    return function(x)

def calc_approx_function_deriv(func_number, deriv_number, x, h):
    deriv_function = get_deriv(deriv_number)
    analytical_function = get_function(func_number)

    return deriv_function(analytical_function, x, h)

def get_step(n):
    return 1 / 2 ** n

def get_step_array(n):
    return np.array([get_step(i) for i in range(n)])

####################################################################################

# SCRIPT START

if os.path.exists('plots_folder'):
    shutil.rmtree('plots_folder')

os.mkdir('plots_folder')
os.chdir('./plots_folder')

h_step_array = get_step_array(21)
ln_h_step_array = np.log(h_step_array)

x_point = 10

for func in range(len(FUNCTIONS_ARRAY)):
    figure = plt.figure(figsize=(FIGURE_XSIZE, FIGURE_YSIZE), facecolor=BACKGROUND_RGB)
    gs = GridSpec(ncols=1, nrows=1, figure=figure)
    axes = figure.add_subplot(gs[0, 0])
    set_axis_properties(axes)

    axes.set_xlabel('$ln(h_n), h_n = 2 / 2^n, n = \overline{1, 21}$')
    axes.set_ylabel('$ln(error)$')
    axes.set_title('Error for ' + FUNCTION_NAMES[func] + ' for different derivatives, in point $x_0 = ' + str(x_point) + '$')

    for deriv in range(len(DERIVATIVES_ARRAY)):
        error_array = np.array([])
        for h_step in h_step_array:
            error_array = np.append(error_array, np.abs(calc_function_deriv(func, x_point) - 
                                    calc_approx_function_deriv(func, deriv, x_point, h_step)))
        ln_error_array = np.log(error_array)
        
        label_name = 'Derivative method ' + DERIVATIVE_NAMES[deriv]
        axes.plot(ln_h_step_array, ln_error_array, label=label_name, marker='o', markersize=6, linestyle='-')

    axes.legend(framealpha=LEGEND_FRAME_ALPHA)
    figure.savefig('error for function ' + str(func + 1) + '.pdf')

os.chdir('./..')

# SCRIPT END

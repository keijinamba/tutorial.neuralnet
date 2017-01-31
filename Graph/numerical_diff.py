# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from Common.base_functions import *

#
def quadratic_function(x):
	return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y = quadratic_function(x)
y2 = numerical_diff(quadratic_function, x)
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.plot(x, y2)
plt.show()
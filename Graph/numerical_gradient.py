# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Common.base_functions import *

def sum_of_squares(x):
	return np.sum(x ** 2)


x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(sum_of_squares, np.array([X, Y]))

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()
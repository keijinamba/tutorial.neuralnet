# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from Common.base_functions import *

def sum_of_squares(x):
	return np.sum(x ** 2)

init_x = np.array([-3.0, 4.0])
answer = gradient_discent(sum_of_squares, init_x = init_x, lr = 0.1, step_num = 100)

print('result of gradient discent function')
print(answer)
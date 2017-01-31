import numpy as np

# 活性化関数
# Step function
def step_function(x):
	return np.array(x > 0, dtype = np.int)

# Identity function
def identity_function(x):
	return x

# Sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Softmax function
def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T 

	x = x - np.max(x) # オーバーフロー対策
	return np.exp(x) / np.sum(np.exp(x))

# ReLU function
def relu(x):
	return np.maximum(0, x)


# 損失関数
# Mean-squared-error function
def mean_squared_error(y, t):
	return 0.5 * np.sum((y - t) ** 2)

# Cross-entropy-error function
def cross_entropy_error(y, t):
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))


# その他
# Numerical-differentiation function
def numerical_diff(f, x):
	h = 1e-4
	return (f(x + h) - f(x - h)) / (2 * h)

# Neumerical-gradient function
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

# Gradient-descent function
def gradient_discent(f, init_x, lr = 0.01, step_num = 100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad

	return x
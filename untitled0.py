# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:17:48 2019

@author: palan
"""

## Add a columns of 1s as intercept to X. This becomes the 2nd column

def train(X, y, W, B, alpha, max_iters):
    '''
    Performs GD on all training examples,    X: Training data set,
    y: Labels for training data,
    W: Weights vector,
    B: Bias variable,
    alpha: The learning rate,
    max_iters: Maximum GD iterations.
    '''
	dW = 0 # Weights gradient accumulator
	dB = 0 # Bias gradient accumulator
	m = X.shape[0] # No. of training examples    for i in range(max_iters):
	dW = 0 # Reseting the accumulators
	dB = 0
	for j in range(m):
		# 1. Iterate over all examples,
		# 2. Compute gradients of the weights and biases in w_grad and b_grad,
		# 3. Update dW by adding w_grad and dB by adding b_grad,
		W = W - alpha * (dW / m) # Update the weights
		B = B - alpha * (dB / m) # Update the bias
	return W, B # Return the updated weights and bias.
#
## Add a columns of 1s as intercept to X. This becomes the 2nd column
X_df['intercept'] = 1

## Transform to Numpy arrays for easier matrix math
## and start beta at 0, 0
X = np.array(X_df)
y = np.array(y_df).flatten()
beta = np.array([0, 0])

def cost_calc(X, y, beta):
## number of training records
	m = len(y)
## Calculate the cost
	J = np.sum((X.dot(beta)-y)**2)/2/m
	return J

cost_calc(X, y, beta)


## Calculate covariance between x and y
#def covariance(inp_data_train_x1,inp_data_train_x2):
#	mean_x1 = mean_c(inp_data_train_x1)
#	mean_x2 = mean_c(inp_data_train_x1)
#	covar = 0.0
#	for i in range(len(inp_data_train_x1)):
#		covar = covar + (inp_data_train_x1[i] - mean_x1) * (inp_data_train_x2[i] - mean_x2)
#	return (covar/len(inp_data_train_x1))
#
## Calculate the variance of a list of numbers
#def variance(inp_data_train_x1):
#	mean_x1 = mean_c(inp_data_train_x1)
#	return sum([(x-mean_x1)**2 for x in inp_data_train_x1])

## Calculate coefficients
#def coefficients(inp_data_train_x1):
#	x1 = [row[0] for row in inp_data_train_x1]
#	x2 = [row[1] for row in inp_data_train_x1]
#	x1_mean, x2_mean = mean_c(x), mean_c(y)
#	b1 = covariance(x1, x1_mean, x2, x2_mean) / variance(x1, x1_mean)
#	b0 = x2_mean - b1 * x1_mean
#	b2 =
#	b3 =
#	b4 =
#	b5 =
#	return b0, b1, b2, b3, b4, b5


#mse = 0.0
#for i in range(len(inp_data_train_x1)):
#			mse = mse + (hx[i]-pred_train[i])**2
#mse = mse/len(inp_data_train_x1)
#rmse = sqrt(mse)
#
#print('rmse',rmse)

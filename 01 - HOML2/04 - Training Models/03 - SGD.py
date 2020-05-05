# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 4 - Training Models
        Section: Stochastic Gradient Descent (SGD)

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np
import matplotlib.pyplot as plt

#
# Stochastic Gradient Descent
#
print("---------------------------")
print("Stochastic Gradient Descent")
print("---------------------------")

# From 01 - Linear Regression...
X = 2 * np.random.rand(100, 1)
y = 4 + ( 3 * X ) + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]
X_new = np.array([[0], [0.5], [1], [1.5], [2]])
X_new_b = np.c_[np.ones((len(X_new),1)), X_new]
plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])

# From 02 - Gradient Descent
m = 100
theta = np.random.randn(2,1) # random initialization

# New stuff for SGD
n_epochs = 50
t0, t1 = 5, 30 # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule((epoch * m) + i)
        theta = theta - (eta * gradients)
        # Again stuff from 02 - Gradient Descent...
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new, y_predict, 'r-') # draw line for each step so that
                                         # we see how it approaches the end

print("Final theta =", theta)

plt.show()

print("---------------------")
print("Now with Scikit-Learn")
print("---------------------")

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print("Intercept =", sgd_reg.intercept_)
print("Coeficients =", sgd_reg.coef_)

# Note: instances are picked randomly, meaning that some could be
#       used more than one time, and others are never used.
#       Prevent this by previously shuffling X_b and y, then iterate
#       through pairs of those shuffled elements of (X_b, y) instead
#       of using random_index.

# Note: the need to specify a pair of t0 and t1 values used in the
#       learning_schedule() function that decreases the learning rate
#       (the value of eta) with time may seem odd, but there is some
#       merit on it. If you use just one value, you have less freedom
#       to adjust the speed of the learning rate decrease. Think in the
#       example values 5 and 30. In the beginning (t = 0), they are the
#       same than 1/6 (= 5/30). So why not use 1/6? Because t is summed
#       to t0, with time and using 5 and 30 we have 5/30, 5/31, 5/32...
#       and this is a slower decrease rate than 1/6, 1/7, 1/8..., if
#       we used the equivalent 1 and 6 initial values. So, by doubling
#       the initial values, we can start at the same learning rate, but
#       set the decrease to be a lot slower. By changing the proportion
#       between t0 and t1, we set the initial learning rate; by changing
#       t1 leaving t0 proportionally unchanged, we set the learning rate
#       speed. If we used only a t value for parametrization we would
#       have trouble setting values for those 2 freedom angles.

print("-------------------------")
print("Ridge Regression with SGD")
print("-------------------------")

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_predict = sgd_reg.predict([[1.5]])

print("Predict =", sgd_predict)
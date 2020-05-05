# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 4 - Training Models
        Section: Gradient Descent

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np
import matplotlib.pyplot as plt

#
# Batch Gradient Descent
#
print("----------------------")
print("Batch Gradient Descent")
print("----------------------")

# From 01 - Linear Regression...
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]
X_new = np.array([[0], [0.5], [1], [1.5], [2]])
X_new_b = np.c_[np.ones((len(X_new),1)), X_new]
plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])

# New stuff for Gradient Descent
eta = 0.1 # learning rate (0.02 is slower, 0.5 is divergent)
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    # Again stuff from 01 - Linear Regression...
    y_predict = X_new_b.dot(theta)
    plt.plot(X_new, y_predict, 'r-') # draw line for each step so that
                                     # we see how it approaches the end

print("Final theta = ", theta)

plt.show()

# Note: we can use here Grid Search (see HOML2, Chapter 2) to
#       find the best learning rate (the eta variable value above).

# Note: usually it is best to set a high number of iterations (if
#       it is too low, algorithm will stop before reaching the
#       optimal solution) but then use some tolerance E such that
#       if the difference between steps is less than E, then stop
#       the cycle. So it will stop either by reaching E, or reaching
#       the maximum number of iterations. Reaching the maximum number
#       of iterations, in this situation, is maybe a good indicator
#       that either E is too high or learning rate is too low.

# Note: when using tolerance E, think that the lower the E value, the
#       more precise will be the result. But think that dividing E
#       by 10 will increase the computing time by 10 times.

print("----------------------------------------")
print("Lasso Regression with subgradient vector")
print("----------------------------------------")

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_predict = lasso_reg.predict([[1.5]])

print("Predict =", lasso_predict)

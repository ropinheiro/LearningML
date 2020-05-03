# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 4 - Training Models
        Section: Linear Regression

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np
import matplotlib.pyplot as plt

#
# Sub-Section: Linear Regression
#
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)

plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])

#
# Sub-Section: The Normal Equation
#
X_b = np.c_[np.ones((100,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta=", theta_best)

X_new = np.array([[0], [0.5], [1], [1.5], [2]])
X_new_b = np.c_[np.ones((len(X_new),1)), X_new]
y_predict = X_new_b.dot(theta_best)
print("Predict=", y_predict)

plt.plot(X_new, y_predict, 'r-')
plt.show()

#
# Sub-Section: ...
#

#
# Sub-Section: ...
#

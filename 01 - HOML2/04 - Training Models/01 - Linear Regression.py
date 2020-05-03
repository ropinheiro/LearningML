# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 4 - Training Models
        Section: Linear Regression

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np
import matplotlib.pyplot as plt

#
# Linear Regression
# (vanilla Python with Numpy)
#
print("--------------------------------------")
print("Linear Regression without Scikit-Learn")
print("--------------------------------------")
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100,1)

plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])

#
# Calculating the Normal Equation
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
# Linear Regression
# (now with Scikit-Learn)
#
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("-----------------------------------")
print("Linear Regression with Scikit-Learn")
print("-----------------------------------")
print("Intercept=", lin_reg.intercept_)
print("Coeficients=", lin_reg.coef_)
print("Predict=", lin_reg.predict(X_new))

# Note: The Scikit-Learn's LinearRegression uses a different formula to
#       calculate the theta. Instead of Normal Equation, it uses what
#       is called the Singular Value Decomposition (SVD).
#       This is better than Normal Equation for a number of reasons.
#       Learn more in HOML2, Chapter 4, pages 116-117.

#
# Sub-Section: ...
#

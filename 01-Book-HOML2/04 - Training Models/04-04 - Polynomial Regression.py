# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 4 - Training Models
        Section: Polynomial Regression

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np
import matplotlib.pyplot as plt

#
# Polynomial Regression
#
print("---------------------")
print("Polynomial Regression")
print("---------------------")

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)
# y = (0.5 x^2) + (1 x^1) + (2 x^0) + noise

plt.plot(X, y, 'b.')
plt.axis([-3,3,0,10])

# Transform the training data, adding the square of each feature
# in the training set as a new feature.
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias = False)
X_poly = poly_features.fit_transform(X)
print("X[0]=", X[0])
print("X_poly[0]=", X_poly[0])

# Now, fit a LinearRegression model to this extended training data
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Intercept=", lin_reg.intercept_)
print("Coeficients=", lin_reg.coef_)

# So, which y would be predicted, using the same X values?
y_predict = lin_reg.coef_[0][1] * X**2 + lin_reg.coef_[0][0] * X + lin_reg.intercept_
plt.plot(X, y_predict, 'rx')

plt.show()

# Note: beware of the combinational explosion of the number of features
#       that can happen by using PolynomialFeatures. See p.130.

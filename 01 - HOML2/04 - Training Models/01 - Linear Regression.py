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
print("-----------------------------------")
print("Linear Regression with Scikit-Learn")
print("-----------------------------------")

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Intercept=", lin_reg.intercept_)
print("Coeficients=", lin_reg.coef_)
print("Predict=", lin_reg.predict(X_new))

# Note: The Scikit-Learn's LinearRegression uses a different formula to
#       calculate the theta. Instead of Normal Equation, it uses what
#       is called the Singular Value Decomposition (SVD).
#       This is better than Normal Equation for a number of reasons.
#       Learn more in HOML2, Chapter 4, pages 116-117.

print("----------------------------------")
print("Linear Regression's Learning Curve")
print("----------------------------------")

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y, style):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    if (style==1):
        plt.plot(np.sqrt(train_errors), "r-x", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-x", linewidth=3, label="val")
    if (style==2):
        plt.plot(np.sqrt(train_errors), "m-", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "c-", linewidth=3, label="val")

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, 1)

# Note: interpreting the chart goes this way. We see red line, that is how
#       the model fits the training data as more and more instances are
#       used. At first, with 1 or 2 instances, it fits perfectly (that's why
#       we start at zero, meaning "no error"). Then as more instances are
#       added to the training data, error increases as the defined line of
#       the (linear...) model sees those additional points to be far from it.
#       At a given moment, adding new training data instances makes no much
#       difference, as the resulting RMSE (Root Mean Sequare Error) stabilizes
#       around some value. Now the blue line, that is how the model fits the
#       validation data. In the beginning, because few instances are not enough
#       to properly generalize the model, the error is stratosferic. Then, as
#       more and more instances are used and the model's line starts reducing
#       the overall RMSE, adding new training data, again, makes no difference
#       and starts stabilizing around the same value than the red line. Now,
#       because both lines stabilize around the same high (error) value, this
#       means that the model is underfitting: the model must be more complex
#       if we want to fix this.

print("---------------------------")
print("Polynomial's Learning Curve")
print("---------------------------")

# NOTE: learn from 04 - Polynomial Regression before this one.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y, 2)
plt.axis([0,80,0,2])

# Note: interpreting the chart is similar than previous one, with some subtle
#       differences. Now, there is a clear gap between the lines: the magenta
#       one usually stabilizing at a lower point than the cyan one, meaning
#       that the models fits way better the trainind data than the validation
#       data. This means that the model is overfitting. To fix that you can
#       either increase the training data so that it generalizes better and
#       the cyan line reachs the magenta line.
# -*- coding: utf-8 -*-
"""
HOML2 - Chapter 5 - Support Vector Machines
        Section: Soft Margin Classification

@author: rmp / rpinheiro / ropinheiro
"""

import numpy as np

print("------------------------")
print("Iris Plants - Linear SVM")
print("------------------------")

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
print("Keys = ", list(iris.keys()))

X = iris["data"][:, (2, 3)] # petal length, petal width

# The validation below sets 1 if it is Iris virginica, else 0.
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])

svm_clf.fit(X, y)

print("Predict = ", svm_clf.predict([[5.5, 1.7]]))

# -*- coding: utf-8 -*-
"""
HOML 2 - Chapter 10 - Artificial Neural Networks with Keras

Regression MLP with the Sequential API

@author: RMP
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

print('Training shape:')
print(X_train_full.shape)
print('Training data type:')
print(X_train_full.dtype)

# Multi-layer Perceptrion (MLP) with 1 hidden layer, since the dataset
# is quite noisy, with a fewer neurons, to avoid overfitting.
model = keras.models.Sequential([
    layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    layers.Dense(1)
])

model.summary()

# Compile the model
model.compile(loss="mean_squared_error",
              optimizer="sgd")

# Train the model
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))


# Evaluate the model in the test set to estimate the
# generalization error
mse_test = model.evaluate(X_test, y_test)

# Predict
X_new = X_test[:3]
y_pred = model.predict(X_new)
y_pred


# -*- coding: utf-8 -*-
"""
HOML 2 - Chapter 10 - Artificial Neural Networks with Keras

Complex Models using the Functional API

@author: RMP
"""

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import activations
# from tensorflow.keras import losses
# from tensorflow.keras import optimizers
# from tensorflow.keras import metrics


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# import pandas as pd
# import numpy as np
# import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

print('Training shape:')
print(X_train_full.shape)
print('Training data type:')
print(X_train_full.dtype)

# ============================================================================
# Multi-layer Perceptrion (MLP) with 2 hidden layers.
# ============================================================================
print('-------------------------------------------------')
print('MLP - 2 hidden layers - deep only - single output')
print('-------------------------------------------------')
input_ = layers.Input(shape=X_train.shape[1:])
hidden1 = layers.Dense(30, activation="relu")(input_)
hidden2 = layers.Dense(30, activation="relu")(hidden1)
concat = layers.concatenate([input_, hidden2])
output = layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])

model.summary()

# Compile the model
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

X_new = X_test[:3]

# Train the model
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)

# ============================================================================
# Same Multi-layer Perceptrion (MLP) with 2 hidden layers,
# but now with some features going the wide part.
# ============================================================================
print('-------------------------------------------------')
print('MLP - 2 hidden layers - deep+wide - single output')
print('-------------------------------------------------')
input_A = layers.Input(shape=[5], name="wide_input")
input_B = layers.Input(shape=[6], name="deep_input")
hidden1 = layers.Dense(30, activation="relu")(input_B)
hidden2 = layers.Dense(30, activation="relu")(hidden1)
concat = layers.concatenate([input_A, hidden2])
output = layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

model.summary()

# Compile the model
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# Train the model
history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))


# ============================================================================
# Same Multi-layer Perceptrion (MLP) with 2 hidden layers,
# but now with an axuiliary output only for the deep part.
# ============================================================================
print('----------------------------------------------------')
print('MLP - 2 hidden layers - deep+wide - auxiliary output')
print('----------------------------------------------------')
input_A = layers.Input(shape=[5], name="wide_input")
input_B = layers.Input(shape=[6], name="deep_input")
hidden1 = layers.Dense(30, activation="relu")(input_B)
hidden2 = layers.Dense(30, activation="relu")(hidden1)
concat = layers.concatenate([input_A, hidden2])
output = layers.Dense(1, name="main_output")(concat)
aux_output = layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],
                           outputs=[output, aux_output])

# Compile the model
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

# Train the model
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

# Evaluate the model in the test set to estimate the generalization error
total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])



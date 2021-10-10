# -*- coding: utf-8 -*-
"""
HOML 2 - Chapter 10 - Artificial Neural Networks with Keras

TensorBoard

@author: RMP
"""

# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import activations
# from tensorflow.keras import losses
# from tensorflow.keras import optimizers
# from tensorflow.keras import metrics


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# import pandas as pd
# import numpy as np
import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

keras.backend.clear_session()

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])


# To start the TensorBoard server, open an Anaconda console and type:
# > conda activate phd
# > cd "C:\Work\Public\LearningML\01 - HOML2\10 - Keras and Tensorflow"
# > tensorboard --logdir=./my_logs --port=6006
# Then point a browser to http://localhost:6006
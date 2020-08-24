# -*- coding: utf-8 -*-
"""
HOML 2 - Chapter 10 - Artificial Neural Networks with Keras

Image Classifier with the Sequential API

@author: RMP
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('Tensorflow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__)


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print('Training shape:')
print(X_train_full.shape)
print('Training data type:')
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print('First training sample: ' + class_names[y_train[0]])

# Multi-layer Perceptrion (MLP) with 2 hidden layers
# 1 - First layer just flattens input into a 1D array.
#     Data matrix is reshapped with (-1, 1) - The -1 here lets Python
#     infer the first dimension while the second is forced to 1.
#     E.g. if matrix is 100x3 it will infer 300, reshapping to (300, 1).
#     You must specify each input example's shape, hence the parameter
#     inpuy_shape being passed with seach image shape (28,28). 
# 2 - Dense hidden layers with 300 and 100 neurons + ReLU activation.
#     Each dense layer manages its own weight matrix, with all the connection
#     weights between the neurons and their inputs. It also manages a vector
#     of bias terms (one per neuron).
# 3 - The last layer is a dense output layer with 10 neurons, one per class,
#     using the softmax activation function, because classes are exclusive.
model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=[28,28]))              #1
model.add(layers.Dense(300, activation=activations.relu))   #2
model.add(layers.Dense(100, activation=activations.relu))   #2
model.add(layers.Dense(10, activation=activations.softmax)) #3

# Note: alternatively, you could pass all layers to Sequential this way:
#       model = keras.models.Sequential( [ layer1, layer2, ... ] )

model.summary()

# Not working, complains with this error, althoug everything is installed:
# ('Failed to import pydot. You must `pip install pydot` and install graphviz
#  (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.')

# keras.utils.plot_model(model)

hidden1 = model.layers[1]
print('Hidden 1 layer name: ' + hidden1.name)
weights, biases = hidden1.get_weights()
print('Weights:')
print(weights)     # will appear randomly initialized
print('Biases:')
print(biases)      # will appear initialized to zeros

# Compile the model
model.compile(loss=losses.sparse_categorical_crossentropy,
              optimizer=optimizers.SGD(),
              metrics=[metrics.sparse_categorical_accuracy])

# Train the model
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# Show training vs validation loss and accuracy
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
plt.show()

# Evaluate the model in the test set to estimate the
# generalization error
model.evaluate(X_test, y_test)

# Predict
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]
y_new = y_test[:3]
y_new


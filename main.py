import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading dataset
mnist_dataset = tf.keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist_dataset.load_data()

# Normalizing pixels (not digits) in dataset to values in range [0, 1]
train_X = tf.keras.utils.normalize(train_X, axis=1)
test_y = tf.keras.utils.normalize(test_X, axis=1)

# Creating the model
ann_model = tf.keras.models.Sequential()
ann_model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Creates an array (not a matrix) of size 784
ann_model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer
ann_model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer
ann_model.add(tf.keras.layers.Dense(10, activation='softmax')) # This represents our output layer and softmax means all 10 values add up to 1. Gives us probability for each digit to be the right answer

# Compiling the model
ann_model.compile(Optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the model, i.e. training the model
ann_model.fit(train_X, train_y, epochs=3) # Epochs specify how many times the ANN will see the same data.

# Saving the model
ann_model.save('HWDRANN.model')
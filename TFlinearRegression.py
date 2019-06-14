#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:15:09 2019

Tensor

@author: mazeyarmoeini
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Creates a line
x  = np.arange(-100,101)
linearFun = lambda t: 2*t + 5
y = np.array([linearFun(xi) for xi in x])


#builds a single preceptron layar
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])


#compiles the model with loss function and optimizer
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))


#Trains the model
history = model.fit(x, y, epochs=30, verbose=False)
print("Finished training the model")


#plots the model
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#predicts new weight
print(model.predict([1000]))

#prints out the weights of the model
weights = model.get_weights()
print("These are the layer variables: {}".format(weights[0][0][0],weights()[1][0]))
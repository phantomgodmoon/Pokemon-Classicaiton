# -*- coding: utf-8 -*-


# COMP2211 PA2 MLP (Multilayer Perceptron) vs. CNN (Convolutional Neural Network) Submission Template

# Provided: notebook bootstrapping
# Keras Models
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import numpy as np

## Task 1.1: Build an MLP Model

# TODO: Build a simple MLP with 2 Dense Layers.

def build_mlp_model():

  model = keras.Sequential(name='mlp')

  ### START YOUR CODE HERE
  model = keras.Sequential(name='mlp')

  ### START YOUR CODE HERE
  model.add(Dense(units=256, input_dim=784))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.ReLU())
  model.add(Dense(units=10, activation='softmax'))
  ### END YOUR CODE HERE

  return model

## Task 1.2: Compile an MLP Model

# TODO: define the optimizer, loss function, learning rate, and metrics

def compile_mlp_model(mlp_model):

  ### START YOUR CODE HERE
  mlp_optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
  mlp_loss=tf.keras.losses.CategoricalCrossentropy()
  mlp_metrics=['accuracy']
  ### END YOUR CODE HERE

  mlp_model.compile(optimizer=mlp_optimizer, loss=mlp_loss, metrics=mlp_metrics)
  return mlp_model


## Task 1.3: Model Training

# TODO: use model.fit() and input correct data, label, batch_size, epochs parameter

def train_model(model, train_epochs, bs, x_train, y_train):

    ### START YOUR CODE HERE
    return model.fit(x=x_train,y=y_train,epochs=train_epochs,batch_size=bs
      
    )
    ### END YOUR CODE HERE


## Task 1.4: Model Evaluation

# TODO: use model.evaluate() to test a given model with given test set

def test_model(model, x_test, y_test):

  ### START YOUR CODE HERE
  return model.evaluate(x=x_test,y=y_test,batch_size=64

   )
  ### END YOUR CODE HERE

## Task 2.1 Build a CNN model

# TODO: Build a simple CNN with 2 Dense Layers.

def build_cnn_model():

  model = keras.Sequential(name='cnn')

  ### START YOUR CODE HERE
  model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1),kernel_initializer=tf.keras.initializers.HeUniform()))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(units=100, activation='relu',kernel_initializer=tf.keras.initializers.HeUniform()))
  model.add(Dense(units=10, activation='softmax'))
  ### END YOUR CODE HERE

  return model


## Task 2.2: Compile a CNN Model

# TODO: define the optimizer, loss function, learning rate, and metrics

def compile_cnn_model(cnn_model):

  ### START YOUR CODE HERE
  cnn_optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  cnn_loss=tf.keras.losses.CategoricalCrossentropy()
  cnn_metrics=['accuracy']
  ### END YOUR CODE HERE

  cnn_model.compile(optimizer=cnn_optimizer, loss=cnn_loss, metrics=cnn_metrics)
  return cnn_model


## Task 2.3: Data Reshaping for CNN Training

# TODO: reshape the data for CNN training

def reshape_data(X_train, X_test):

  ### START YOUR CODE HERE
  X_train_reshape=X_train.reshape(60000,28,28,1)
  ### END YOUR CODE HERE
  X_test_reshape=X_test.reshape(10000,28,28,1)
  ### END YOUR CODE HERE

  return X_train_reshape, X_test_reshape

